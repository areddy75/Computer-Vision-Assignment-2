"""Evaluate the best checkpoint on a labelled folder or predict an unlabelled folder.

Examples
  python predict.py --data data/test                       # labelled: data/test/<class>/*.jpg -> accuracy
  python predict.py --data data/test2 --out preds.csv      # unlabelled: data/test2/*.jpg -> CSV
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from models import Net, load_img, preprocess


def pick_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


@torch.no_grad()
def predict(model, X, mean, std, device, tta=True, bs=256):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        x = preprocess(X[i:i + bs].to(device), mean, std)
        p = model(x).softmax(1)
        if tta: p = p + model(torch.flip(x, dims=[3])).softmax(1)   # horizontal-flip TTA
        out.append(p.cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/best_model.pt")
    ap.add_argument("--data", required=True, help="folder of class subfolders (labelled) or of images (unlabelled)")
    ap.add_argument("--out", default=None, help="optional CSV of predictions")
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args()

    device = pick_device()
    ck = torch.load(args.ckpt, map_location="cpu")
    classes = ck["classes"]
    model = Net(len(classes)); model.load_state_dict(ck["state_dict"]); model.to(device)

    root = Path(args.data)
    subdirs = sorted(d for d in root.iterdir() if d.is_dir())
    if subdirs:   # labelled
        files, y = [], []
        for d in subdirs:
            for f in sorted(d.glob("*.jpg")):
                files.append(f); y.append(classes.index(d.name))
        y = torch.tensor(y)
    else:
        files, y = sorted(root.glob("*.jpg")), None

    X = torch.stack([load_img(f) for f in files])
    pred = predict(model, X, ck["mean"], ck["std"], device, tta=not args.no_tta).argmax(1)
    print(f"device {device} | {len(files)} images")
    if y is not None:
        acc = (pred == y).float().mean().item()
        print(f"accuracy: {acc * 100:.2f}% ({int((pred == y).sum())}/{len(y)})")
    if args.out:
        pd.DataFrame({"file": [str(f.relative_to(root)) for f in files],
                      "pred": [classes[i] for i in pred]}).to_csv(args.out, index=False)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
