import os

import pandas as pd
from PIL import Image

from vitstr import load_ViTSTR


def predict(model, opt):
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

        pred, conf, time = model.imread(img)
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
    model, opt = load_ViTSTR("config.yaml")
    model.eval().freeze()

    predict(model, opt)
