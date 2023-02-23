import pytorch_lightning as pl
import pandas as pd

from vitstr import DataModule, load_ViTSTR


def predict(model, opt):
    dm = DataModule(opt)
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
    model, opt = load_ViTSTR("config.yaml")
    model.eval().freeze()

    predict(model, opt)
