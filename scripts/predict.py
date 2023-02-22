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
import numpy as np
import pandas as pd

from vitstr import Model
from vitstr import DataModule

warnings.filterwarnings(action="ignore")


def predict(opt):
    if opt.saved_model == "" or os.path.exists(opt.saved_model):
        assert f"{opt.saved_model} is not exist!"

    dm = DataModule(opt)
    model = Model.load_from_checkpoint(opt.saved_model, opt=opt)
    model.eval()
    print(f"model loaded from checkpoint {opt.saved_model}")

    print(model.hparams)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=opt.num_gpu,
        precision=16,
    )

    predict_result = trainer.predict(model, dm)

    predict_result = [pred for pred in predict_result[0]]
    predict_df = pd.DataFrame(
        [
            (
                img_name,
                img_name.split(".jpg")[0].split("-")[0],
                pred.upper(),
                conf.item(),
            )
            for img_name, pred, conf in predict_result
        ],
        columns=["img_name", "label", "pred", "conf"],
    )
    predict_df["correct"] = predict_df.apply(lambda x: x.label == x.pred, axis=1)
    predict_df.to_csv("predict_result.csv", index=False)
    failures = predict_df.loc[predict_df["correct"] == False]
    failures.to_csv("predict_failures.csv", index=False)
    print("Accuracy:", (len(predict_df) - len(failures)) / len(predict_df) * 100)


if __name__ == "__main__":
    """load configuration"""
    with open("config.yaml", "r") as f:
        opt = yaml.safe_load(f)
        opt = Namespace(**opt)

    """ Seed and GPU setting """
    pl.seed_everything(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    predict(opt)
