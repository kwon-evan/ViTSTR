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

import torch
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
from nltk.metrics.distance import edit_distance
from glob import glob

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from vitstr import ViTSTR


class Model(pl.LightningModule):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt

        model = ViTSTR(opt)

        self.Transformation = model.Transformation
        self.vitstr = model.vitstr

        self.converter = model.converter
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # ignore [GO]

        self.save_hyperparameters(self.opt)

    def forward(self, input, seqlen=25):
        """Transformation stage"""
        input = self.Transformation(input)
        prediction = self.vitstr(input, seqlen=seqlen)
        return prediction

    def training_step(self, batch, batch_idx):
        # train part
        image, labels = batch

        target = self.converter.encode(labels).to(next(self.parameters()).device)
        preds = self(image, seqlen=self.converter.batch_max_length)
        cost = self.criterion(
            preds.view(-1, preds.shape[-1]), target.contiguous().view(-1)
        )

        self.log("train-loss", cost, prog_bar=True, batch_size=self.opt.batch_size)
        return cost

    def validation_step(self, batch, batch_idx):
        image, labels = batch
        batch_size = image.size(0)

        # For max length prediction
        target = self.converter.encode(labels).to(next(self.parameters()).device)

        preds = self(image, seqlen=self.converter.batch_max_length)
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

            if i < 20:
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

        preds = self(image, seqlen=self.converter.batch_max_length)
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

        preds = self(image, seqlen=self.converter.batch_max_length)
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

    def configure_optimizers(self):
        filtered_parameters = []
        params_num = []
        train_size = len(glob(os.path.join(self.opt.data_root, "*", "train", "*.jpg")))
        num_iter = math.ceil(train_size / self.opt.batch_size)

        for p in filter(lambda p: p.requires_grad, self.parameters()):
            filtered_parameters.append(p)
            params_num.append(np.prod(p.size()))
        print("Trainable params num : ", sum(params_num))

        if self.opt.adam:
            optimizer = optim.Adam(
                filtered_parameters,
                lr=self.opt.lr,
                betas=(self.opt.beta1, 0.999),
            )
        else:
            optimizer = optim.Adadelta(
                filtered_parameters,
                lr=self.opt.lr,
                rho=self.opt.rho,
                eps=self.opt.eps,
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
