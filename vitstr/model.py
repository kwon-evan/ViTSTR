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
from time import process_time
from typing import Union
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from vitstr.modules.transformation import TPS_SpatialTransformerNetwork
from vitstr.modules.vitstr import create_vitstr
from vitstr.config import ModelConfig

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    @property
    def device(self):
        return next(self.parameters()).device

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
        return batch_text.to(self.device)

    def decode(self, text_index, length):
        """convert text-index into text-label."""
        texts = []
        for index, l in enumerate(length):
            text = "".join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class ViTSTR(nn.Module):
    def __init__(self, opt: Union[ModelConfig, Namespace] = ModelConfig()):
        super(ViTSTR, self).__init__()
        self.opt = opt

        """ Transformation """
        self.Transformation = TPS_SpatialTransformerNetwork(
            F=self.opt.num_fiducial,
            I_size=self.opt.img_size,
            I_r_size=self.opt.img_size,
            I_channel_num=3 if self.opt.rgb else 1,
        )

        self.vitstr = create_vitstr(
            num_tokens=len(self.opt.character) + 2,
            model=self.opt.model_name,
            pretrained=self.opt.pretrained,
        )

        self.character = self.opt.character
        self.converter = TokenLabelConverter(self.opt)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input, seqlen=25):
        """Transformation stage"""
        input = self.Transformation(input)

        prediction = self.vitstr(input, seqlen=seqlen)
        return prediction

    def imread(self, image):
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

        totensor = transforms.ToTensor()

        if not self.opt.rgb:
            image = image.convert("L")
        image = image.resize(self.opt.img_size, Image.BICUBIC)
        image = totensor(image).to(self.device)
        image.sub_(0.5).div_(0.5)
        image = image.unsqueeze(0)

        # For max length prediction
        preds = self(image, seqlen=self.converter.batch_max_length)
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

        return pred.upper(), confidence_score.item()

    @staticmethod
    def load_from(
        path: str,
        opt: Union[ModelConfig, Namespace] = ModelConfig(),
        device: torch.device = torch.device("cpu"),
    ):
        if opt.kor:
            opt.character = "0123456789가강거경계고관광구금기김나남너노누다대더도동두등라러로루리마머명모무문미바배버보부북사산서소수시아악안양어연영오용우울원육이인자작저전조주중지차천초추충카타파평포하허호홀히"
        model = ViTSTR(opt)
        model.load_state_dict(torch.load(path, map_location=device)["state_dict"])
        model.to(device)
        model.eval()

        # warm-up
        model.imread(Image.new("RGB", (100, 32), (255, 255, 255)))

        return model
