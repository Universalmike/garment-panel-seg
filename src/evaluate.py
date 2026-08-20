"""
Score a trained checkpoint the way the brief scores it.

This measures the DEPLOYED pipeline, not a training-time approximation: images
go through exactly what predict.py does - resize to the network's input size,
forward pass (with flip TTA by default), bilinear upsample of the logits back to
the image's native resolution, then argmax. Ground truth is the full-resolution
mask written by prepare_data.py. So the number here is what the model would
actually score, not what it looks like at 256x256 with nearest-neighbour edges.

It reports a per-class breakdown rather than a single blended figure, because
one number hides which panel is failing.

Usage:
    python -m src.evaluate --manifest data/train/manifest.json

    # same val split train.py held out (needs the same --val-split and --seed)
    python -m src.evaluate --manifest data/train/manifest.json --seed 42 --val-split 0.15

    # is flip TTA actually earning its keep?
    python -m src.evaluate --manifest data/train/manifest.json --compare
"""

import argparse
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from .metrics import IoUAccumulator
from .model import PanelSegNet, TRAIN_CLASSES, NUM_TRAIN_CLASSES

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_checkpoint(weights_path, device="cpu"):
    ckpt = torch.load(weights_path, map_location=device)
    model = PanelSegNet(pretrained=False, freeze_encoder=True)
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    model.eval().to(device)
    return model, ckpt.get("size", 256)


def val_indices(n, val_split, seed):
    """Reproduce exactly the split train.py holds out."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return idx[: int(n * val_split)]


def predict_full_res(model, image_path, size, target_hw, tta=True, device="cpu"):
    """The deployed inference path, returning a full-resolution class map."""
    img = Image.open(image_path).convert("RGB").resize((size, size), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - _MEAN) / _STD
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        if tta:
            flipped = model(torch.flip(x, dims=[3]))
            logits = (logits + torch.flip(flipped, dims=[3])) / 2.0
        logits = F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)


def evaluate(model, items, size, tta, device="cpu"):
    acc = IoUAccumulator(NUM_TRAIN_CLASSES, TRAIN_CLASSES)
    for it in tqdm(items, desc="tta" if tta else "no-tta"):
        gt = np.asarray(Image.open(it["mask"]), dtype=np.uint8)
        pred = predict_full_res(model, it["image"], size, gt.shape[:2], tta=tta, device=device)
        acc.update(pred, gt)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--weights", default="weights/model.pt")
    ap.add_argument("--val-split", type=float, default=0.15,
                    help="use the same value train.py used, to score the same held-out images")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all", action="store_true",
                    help="score every image in the manifest, not just the val split")
    ap.add_argument("--no-tta", action="store_true", help="disable horizontal-flip TTA")
    ap.add_argument("--compare", action="store_true",
                    help="report with and without flip TTA, to show whether it earns its cost")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, size = load_checkpoint(args.weights, device)

    with open(args.manifest) as f:
        items = json.load(f)
    if not args.all:
        items = [items[i] for i in val_indices(len(items), args.val_split, args.seed)]

    print(f"scoring {len(items)} images at {size}px input, device {device}")

    if args.compare:
        for tta in (False, True):
            acc = evaluate(model, items, size, tta, device)
            print(f"\n--- flip TTA: {'on' if tta else 'off'} ---")
            print(acc.report())
    else:
        acc = evaluate(model, items, size, not args.no_tta, device)
        print()
        print(acc.report())


if __name__ == "__main__":
    main()
