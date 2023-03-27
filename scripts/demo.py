import os
import yaml
import torch
from argparse import Namespace
from vitstr import ViTSTR, ModelConfig
from PIL import Image
from glob import glob
from time import time
from rich.progress import track
from rich import print


def print_result(results):
    # print results
    print("\n-----results-----")
    print(f"{'gt':^10} {'pred':^10} {'conf':^10} {'time':^10} {'acc':^10}")
    for gt, pred, conf, inf_time, acc in results:
        print(f"{gt:>10} {pred:>10} {conf:>10.4f} {inf_time:>8.4f}ms {acc}")

    # print statistics
    acc = [acc for _, _, _, _, acc in results]
    times = [inf_time for _, _, _, inf_time, _ in results]
    print("\n-----Accuracy-----")
    print(
        f"correct: {sum(acc)}/{len(acc)}, "
        + f"incorrect: {len(acc) - sum(acc)}/{len(acc)}"
    )
    print(f"accuracy: {sum(acc) / len(acc) * 100:.2f} %")
    print("\n-----inference time-----")
    print(f"mean: {sum(times) / len(times):.4f} ms")
    print(f"max: {max(times):.4f} ms")
    print(f"min: {min(times):.4f} ms")


if __name__ == "__main__":
    """load configuration"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved_model = Namespace(
        **yaml.safe_load(open("scripts/components/config-kor.yaml", "r"))
    ).saved_model
    cfg = ModelConfig(kor=True)

    """load model"""
    model = ViTSTR.load_from(saved_model, opt=cfg, device=device)
    print("Model Loaded!")

    """inference"""
    paths = glob("demo_images_kor/*.jpg")

    results = []
    for path in track(paths):
        gt = os.path.basename(path).split(".")[0].split("-")[0]
        image = Image.open(path)
        start = time()
        pred, conf = model.imread(image)
        end = time()
        inf_time = (end - start) * 1000

        results.append((gt, pred, conf, inf_time, gt == pred))
    print_result(results)
