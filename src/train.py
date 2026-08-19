"""
Train the panel segmentation model.

Reproducible: all seeds are fixed and cudnn is set deterministic, so re-running
gives comparable numbers.

Loss: class-weighted cross-entropy + Dice. Collar and sleeve are small relative
to body and background, so we up-weight them; Dice further helps the thin/small
classes.

Metric: mean per-class IoU (the same definition the graders use), computed over
the classes present in each image and averaged.

Usage:
    python -m src.train --manifest data/train/manifest.json \
        --val-split 0.15 --epochs 25 --size 256 --batch 16 --seed 42
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .dataset import GarmentDataset
from .model import PanelSegNet, NUM_TRAIN_CLASSES, count_params


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dice_loss(logits, target, num_classes, eps=1.0):
    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    union = probs.sum(dims) + onehot.sum(dims)
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


@torch.no_grad()
def mean_iou(logits, target, num_classes):
    """Mean per-class IoU over classes present in the batch's ground truth."""
    pred = logits.argmax(1)
    ious = []
    for c in range(num_classes):
        p, t = pred == c, target == c
        if not t.any() and not p.any():
            continue
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="weights/model.pt")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    full = GarmentDataset(args.manifest, size=args.size, train=True, seed=args.seed)
    n = len(full)
    idx = list(range(n))
    random.Random(args.seed).shuffle(idx)
    n_val = int(n * args.val_split)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    val_ds = GarmentDataset(args.manifest, size=args.size, train=False, seed=args.seed)
    train_loader = DataLoader(Subset(full, train_idx), batch_size=args.batch,
                              shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=args.batch,
                            shuffle=False, num_workers=2)
    print(f"train {len(train_idx)}  val {len(val_idx)}")

    model = PanelSegNet(pretrained=True, freeze_encoder=True).to(device)
    tr, tot = count_params(model)
    print(f"trainable params {tr:,}  |  total inference params {tot:,}")

    # class weights: background, body, sleeve, collar
    weights = torch.tensor([0.5, 1.0, 2.0, 3.0], device=device)
    ce = nn.CrossEntropyLoss(weight=weights)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = -1.0
    for ep in range(args.epochs):
        model.train()
        run = 0.0
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            logits = model(img)
            loss = ce(logits, mask) + dice_loss(logits, mask, NUM_TRAIN_CLASSES)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item()
        sched.step()

        model.eval()
        mious = []
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(device), mask.to(device)
                mious.append(mean_iou(model(img), mask, NUM_TRAIN_CLASSES))
        vmiou = float(np.mean(mious)) if mious else 0.0
        print(f"epoch {ep+1:02d}/{args.epochs}  loss {run/max(1,len(train_loader)):.4f}  val_mIoU {vmiou:.4f}")

        if vmiou > best:
            best = vmiou
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            # Save the FULL state dict (encoder + decoder) so predict.py needs no
            # download at inference time.
            torch.save({"state_dict": model.state_dict(),
                        "val_miou": vmiou,
                        "size": args.size,
                        "seed": args.seed}, args.out)
            print(f"  saved {args.out} (val_mIoU {vmiou:.4f})")

    print(f"best val_mIoU {best:.4f}")


if __name__ == "__main__":
    main()
