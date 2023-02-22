import os
import random
import string
from argparse import Namespace
import yaml
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import pytorch_lightning as pl
import pandas as pd
from PIL import Image

from vitstr import Model

warnings.filterwarnings(action="ignore")


def predict(opt):
    if opt.saved_model == "" or os.path.exists(opt.saved_model):
        assert f"{opt.saved_model} is not exist!"

    model = Model.load_from_checkpoint(opt.saved_model, opt=opt)
    model.eval().to(device)
    # model.freeze()
    print(f"model loaded from checkpoint {opt.saved_model}")

    print(model.hparams)

    IMAGE_FOLDER = "demo_images/"
    # IMAGE_FOLDER = opt.test_data

    result = {
        "img_names": [],
        "labels": [],
        "preds": [],
        "confs": [],
        "times": [],
        "acc": [],
    }

    for img_name in os.listdir(IMAGE_FOLDER):
        img = Image.open(f"{IMAGE_FOLDER}/{img_name}")
        label = img_name.split(".")[0].split("-")[0]

        pred, conf, time = model.imread(img, device)
        result["img_names"].append(img_name)
        result["labels"].append(label)
        result["preds"].append(pred)
        result["confs"].append(conf)
        result["times"].append(time)
        result["acc"].append(label == pred)

    df = pd.DataFrame(result)
    print(df)


if __name__ == "__main__":
    """load configuration"""
    with open("config.yaml", "r") as f:
        opt = yaml.safe_load(f)
        opt = Namespace(**opt)

    """ Seed and GPU setting """
    pl.seed_everything(opt.mamualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predict(opt)
