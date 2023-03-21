import os
import yaml
import torch
from argparse import Namespace
from vitstr import ViTSTR, ModelConfig
from PIL import Image
from glob import glob
from rich.progress import track

if __name__ == "__main__":
    """load configuration"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved_model = Namespace(
        **yaml.safe_load(open("scripts/components/config.yaml", "r"))
    ).saved_model
    cfg = ModelConfig(kor=False)

    """load model"""
    model = ViTSTR.load_from(saved_model, opt=cfg).to(device).eval()

    """inference"""
    paths = glob("demo_images/*.jpg")

    acc = []
    print(f"{'gt':<10} {'pred':<10} {'conf':<10} {'acc':<10}")
    for path in track(paths):
        gt = os.path.basename(path).split(".")[0].split("-")[0]
        image = Image.open(path).convert("RGB")
        pred, conf = model.imread(image)

        acc.append(gt == pred)
        print(f"{gt:<10} {pred:<10} {conf:<10.4f} {gt==pred}")
    print("Accuracy: ", sum(acc) / len(acc))
