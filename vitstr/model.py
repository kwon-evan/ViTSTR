"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import math
import yaml
from argparse import Namespace
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import pytorch_lightning as pl
import numpy as np
from nltk.metrics.distance import edit_distance

from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

from vitstr.modules.transformation import TPS_SpatialTransformerNetwork
from vitstr.modules.vitstr import create_vitstr, load_pretrained

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TokenLabelConverter(object):
    """Convert between text-label and text-index"""

    def __init__(self, opt):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        self.SPACE = "[s]"
        self.GO = "[GO]"
        self.list_token = [self.GO, self.SPACE]
        self.character = self.list_token + list(opt.character)

        self.dict = {word: i for i, word in enumerate(self.character)}
        self.batch_max_length = opt.batch_max_length + len(self.list_token)

    def encode(self, text):
        """convert text-label into text-index."""
        length = [
            len(s) + len(self.list_token) for s in text
        ]  # +2 for [GO] and [s] at end of sentence.
        batch_text = torch.LongTensor(len(text), self.batch_max_length).fill_(
            self.dict[self.GO]
        )
        for i, t in enumerate(text):
            txt = [self.GO] + list(t) + [self.SPACE]
            txt = [self.dict[char] for char in txt]
            batch_text[i][: len(txt)] = torch.LongTensor(
                txt
            )  # batch_text[:, 0] = [GO] token
        return batch_text.to(device)

    def decode(self, text_index, length):
        """convert text-index into text-label."""
        texts = []
        for index, l in enumerate(length):
            text = "".join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class Model(pl.LightningModule):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt

        """ Transformation """
        self.Transformation = TPS_SpatialTransformerNetwork(
            F=self.opt.num_fiducial,
            I_size=(self.opt.imgH, self.opt.imgW),
            I_r_size=(self.opt.imgH, self.opt.imgW),
            I_channel_num=3 if self.opt.rgb else 1,
        )

        self.vitstr = create_vitstr(
            num_tokens=len(self.opt.character) + 2,
            model=self.opt.model_name,
            load_pretrained=self.opt.load_pretrained,
        )

        self.converter = TokenLabelConverter(self.opt)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # ignore [GO]

        self.save_hyperparameters(self.opt)

    # NOTE: args "text" & "is_train" may useless
    def forward(self, input, text, is_train=True, seqlen=25):
        """Transformation stage"""
        input = self.Transformation(input)

        prediction = self.vitstr(input, seqlen=seqlen)
        return prediction

    def training_step(self, batch, batch_idx):
        # train part
        image, labels = batch

        target = self.converter.encode(labels)
        preds = self(image, text=target, seqlen=self.converter.batch_max_length)
        cost = self.criterion(
            preds.view(-1, preds.shape[-1]), target.contiguous().view(-1)
        )

        self.log("train-loss", cost, prog_bar=True, batch_size=self.opt.batch_size)
        return cost

    def validation_step(self, batch, batch_idx):
        image, labels = batch
        batch_size = image.size(0)

        # For max length prediction
        target = self.converter.encode(labels)

        preds = self(image, text=target, seqlen=self.converter.batch_max_length)
        _, preds_index = preds.topk(1, dim=-1, largest=True, sorted=True)
        preds_index = preds_index.view(-1, self.converter.batch_max_length)
        cost = self.criterion(
            preds.contiguous().view(-1, preds.shape[-1]),
            target.contiguous().view(-1),
        )

        length_for_pred = torch.IntTensor(
            [self.converter.batch_max_length - 1] * batch_size
        )
        preds_str = self.converter.decode(preds_index[:, 1:], length_for_pred)

        self.log("val-loss", cost, prog_bar=True, batch_size=self.opt.batch_size)
        return preds, preds_str, labels

    def validation_epoch_end(self, outputs):
        preds = []
        preds_str = []
        labels = []
        n_correct = 0
        norm_ED = 0

        for pred, pred_str, label in outputs:
            preds.extend(pred)
            preds_str.extend(pred_str)
            labels.extend(label)

        preds = torch.stack(preds)

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        print("=" * 15 * 3)
        print(f"{'predict':15s}{'ground truth':15s}{'confidence':15s}")
        print("=" * 15 * 3)
        for i, (gt, pred, pred_max_prob) in enumerate(
            zip(labels, preds_str, preds_max_prob)
        ):
            pred_EOS = pred.find("[s]")
            pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            pred_max_prob = pred_max_prob[:pred_EOS]

            if pred == gt:
                n_correct += 1

            # ICDAR2019 Normalized Edit Distance
            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list.append(confidence_score)

            if i < 10:
                print(f"{pred:15s}{gt:15s}{confidence_score:.4f}")

        accuracy = n_correct / float(len(preds)) * 100
        norm_ED = norm_ED / float(len(preds))  # ICDAR2019 Normalized Edit Distance

        print("-" * 15 * 3)
        print(
            f"Accuracy: {accuracy:.4f}, Norm_ED: {norm_ED:.4f}, Confidence: {sum(confidence_score_list)/len(confidence_score_list):.4f}"
        )

        self.log_dict(
            {"val-acc": accuracy, "val-ned": norm_ED},
            prog_bar=True,
            batch_size=self.opt.batch_size,
        )

    def test_step(self, batch, batch_idx):
        image, labels = batch
        batch_size = image.size(0)
        n_correct = 0
        norm_ED = 0

        # For max length prediction
        target = self.converter.encode(labels)

        preds = self(image, text=target, seqlen=self.converter.batch_max_length)
        _, preds_index = preds.topk(1, dim=-1, largest=True, sorted=True)
        preds_index = preds_index.view(-1, self.converter.batch_max_length)
        cost = self.criterion(
            preds.contiguous().view(-1, preds.shape[-1]),
            target.contiguous().view(-1),
        )

        length_for_pred = torch.IntTensor(
            [self.converter.batch_max_length - 1] * batch_size
        )
        preds_str = self.converter.decode(preds_index[:, 1:], length_for_pred)

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            pred_EOS = pred.find("[s]")
            pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            pred_max_prob = pred_max_prob[:pred_EOS]

            if pred == gt:
                n_correct += 1

            # ICDAR2019 Normalized Edit Distance
            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list.append(confidence_score)

        accuracy = n_correct / float(len(preds)) * 100
        norm_ED = norm_ED / float(len(preds))  # ICDAR2019 Normalized Edit distance

        self.log_dict(
            {"test-loss": cost, "test-accuracy": accuracy, "test-norm_ED": norm_ED},
            prog_bar=True,
            batch_size=self.opt.batch_size,
        )

    def predict_step(self, batch, batch_idx):
        image, image_path_list = batch
        labels = [
            image_name.split(".")[0].split("-")[0] for image_name in image_path_list
        ]
        batch_size = image.size(0)

        # For max length prediction
        target = self.converter.encode(labels)

        preds = self(image, text=target, seqlen=self.converter.batch_max_length)
        _, preds_index = preds.topk(1, dim=-1, largest=True, sorted=True)
        preds_index = preds_index.view(-1, self.converter.batch_max_length)

        length_for_pred = torch.IntTensor(
            [self.converter.batch_max_length - 1] * batch_size
        )
        preds_str = self.converter.decode(preds_index[:, 1:], length_for_pred)

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)

        predicts = []

        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            pred_EOS = pred.find("[s]")
            pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            pred_max_prob = pred_max_prob[:pred_EOS]

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])

            predicts.append((gt, pred, confidence_score))

        return predicts

    def imread(self, image, device=device):
        """
        Read Texts in PIL Image.

        Args:
            image: PIL Image to Read
            device: torch.device

        Returns:
            predict: predicted string
            confidence: model's confidence_score
            inference_time: inference_time in ms
        """
        from PIL import Image
        import torchvision.transforms as transforms
        import time

        start_time = time.time()

        totensor = transforms.ToTensor()

        if not self.opt.rgb:
            image = image.convert("L")
        image = image.resize((self.opt.imgW, self.opt.imgH), Image.BICUBIC)
        image = totensor(image).to(device)
        image.sub_(0.5).div_(0.5)
        image = image.unsqueeze(0)

        text_for_pred = torch.LongTensor(1, self.opt.batch_max_length + 1).fill_(0)

        # For max length prediction
        preds = self(image, text=text_for_pred, seqlen=self.converter.batch_max_length)
        _, preds_index = preds.topk(1, dim=-1, largest=True, sorted=True)
        preds_index = preds_index.view(-1, self.converter.batch_max_length)

        length_for_pred = torch.IntTensor([self.converter.batch_max_length - 1])
        preds_str = self.converter.decode(preds_index[:, 1:], length_for_pred)

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)

        pred = None
        confidence_score = None
        for pred, pred_max_prob in zip(preds_str, preds_max_prob):
            pred_EOS = pred.find("[s]")
            pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            pred_max_prob = pred_max_prob[:pred_EOS]

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])

        end_time = time.time()

        return pred.upper(), confidence_score.item(), (end_time - start_time) * 1000

    def configure_optimizers(self):
        filtered_parameters = []
        params_num = []
        num_iter = math.ceil(len(os.listdir(self.opt.train_data)) / self.opt.batch_size)

        for p in filter(lambda p: p.requires_grad, self.parameters()):
            filtered_parameters.append(p)
            params_num.append(np.prod(p.size()))
        print("Trainable params num : ", sum(params_num))

        if self.opt.adam:
            optimizer = optim.Adam(
                filtered_parameters, lr=self.opt.lr, betas=(self.opt.beta1, 0.999)
            )
        else:
            optimizer = optim.Adadelta(
                filtered_parameters, lr=self.opt.lr, rho=self.opt.rho, eps=self.opt.eps
            )

        scheduler = None
        if self.opt.scheduler:
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=num_iter,
                T_mult=1,
                eta_min=0.00001,
            )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def load_ViTSTR(yaml_path: str) -> Tuple[Model, Namespace]:
    """
    Load ViTSTR model from yaml file
    Args:
        yaml_path(str): path to yaml file
    Returns:
        model: loaded or created ViTSTR model
        opt: Namespace object with model parameters
    """

    # load Configuration
    with open(yaml_path, "r") as f:
        opt = Namespace(**yaml.safe_load(f))
        print(f"Configuration file loaded from {yaml_path}")

    # set experiment name if not specified
    if not opt.exp_name:
        opt.exp_name = f"ViTSTR-Seed{opt.manualSeed}"

    # Seed and GPU settings
    pl.seed_everything(opt.manualSeed)
    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    # Load model from checkpoint if specified in yaml file else create new model
    if opt.saved_model:
        if not os.path.exists(opt.saved_model):
            raise FileNotFoundError(f"Model file {opt.saved_model} not found")

        model = Model.load_from_checkpoint(opt.saved_model, opt=opt)
        print(f"Model loaded from {opt.saved_model}")
    else:
        model = Model(opt)
        print(f"Model created with {yaml_path}")

    # model warm-up with dummy tensor
    model.to(device)
    dummy_image = torch.rand(1, 3 if opt.rgb else 1, opt.imgH, opt.imgH).to(device)
    dummy_text = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)
    model(dummy_image, dummy_text)

    return model, opt
