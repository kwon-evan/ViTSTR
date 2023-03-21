import yaml
import torch
import torch.backends.cudnn as cudnn
from argparse import Namespace
import pytorch_lightning as pl
import pandas as pd

from components.datamodule import DataModule
from components.pl_wrapper import Model
from vitstr import ModelConfig


def predict(opt):
    cfg = ModelConfig()
    # Merge config with opt
    for k, v in vars(cfg).items():
        if k not in vars(opt):
            setattr(opt, k, v)
    if opt.kor:
        kor_character = "0123456789가강거경계고관광구금기김나남너노누다대더도동두등라러로루리마머명모무문미바배버보부북사산서소수시아악안양어연영오용우울원육이인자작저전조주중지차천초추충카타파평포하허호홀히"
        opt.character = kor_character
    print(opt)

    dm = DataModule(opt)
    model = Model.load_from_checkpoint(opt.saved_model, opt=opt).eval()
    model.freeze()

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
    opt = Namespace(**yaml.safe_load(open("scripts/components/config.yaml", "r")))

    """ Seed and GPU setting """
    pl.seed_everything(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    predict(opt)
