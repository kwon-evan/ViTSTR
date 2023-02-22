import os
from argparse import Namespace
import yaml
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import pytorch_lightning as pl

from vitstr import Model
from vitstr import DataModule

warnings.filterwarnings(action="ignore")


def test(opt):
    if opt.saved_model == "" or os.path.exists(opt.saved_model):
        assert f"{opt.saved_model} is not exist!"

    dm = DataModule(opt)
    print("Loding Saved:", opt.saved_model)
    print(os.path.exists(opt.saved_model))
    model = Model.load_from_checkpoint(opt.saved_model, opt=opt)
    model.eval()
    print(f"model loaded from checkpoint {opt.saved_model}")

    print(model.hparams)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=opt.num_gpu,
        precision=16,
    )

    test_result = trainer.test(model, dm)

    print(test_result)


if __name__ == "__main__":
    """load configuration"""
    with open("config.yaml", "r") as f:
        opt = yaml.safe_load(f)
        opt = Namespace(**opt)

    if not opt.exp_name:
        opt.exp_name = f"ViTSTR-Seed{opt.manualSeed}"
        # print(opt.exp_name)

    """ Seed and GPU setting """
    pl.seed_everything(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True

    opt.num_gpu = torch.cuda.device_count()

    test(opt)
