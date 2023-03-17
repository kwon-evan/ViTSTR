import os
import yaml
import torch
from argparse import Namespace
from vitstr import ViTSTR
from PIL import Image
from glob import glob
from rich.progress import track

if __name__ == "__main__":
    """load configuration"""
    opt = Namespace(**yaml.safe_load(open("scripts/components/config.yaml", "r")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """load model"""
    model = ViTSTR.load_from(opt.saved_model).to(device).eval()

    """inference"""
    paths = glob("img_plate/*.jpg")

    for path in track(paths):
        image = Image.open(path).convert("RGB")
        pred, conf = model.imread(image)

        # if img exists, add extra numbering to pred
        if os.path.exists(f"img_plate_read/{pred}.jpg"):
            i = 1
            while os.path.exists(f"img_plate_read/{pred}-{i}.jpg"):
                i += 1
            pred = f"{pred}_{i}"

        image.save(f"img_plate_read/{pred}.jpg", "JPEG")
