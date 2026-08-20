"""
Train the panel segmentation model.

Reproducible: all seeds are fixed and cudnn is set deterministic, so re-running
gives comparable numbers.

Loss: class-weighted cross-entropy + Dice. Collar and sleeve are small relative
to body and background, so we up-weight them; Dice further helps the thin/small
classes.

Metric: the brief's own definition - per-class IoU over the classes present in
each IMAGE, averaged, then averaged over images, with background excluded. See
src/metrics.py for why those two details matter. Model selection uses the
panels-only number, so we do not pick a checkpoint that merely got good at
predicting background.

Mixing in synthetic data: --synth-manifest adds procedurally generated garment
renders (see src/synth.py) to the TRAINING set only. Validation stays purely real
Fashionpedia, deliberately - synthetic images are far easier than photographs, so
letting them into the val split would inflate the number and make it
incomparable to the previous run. The question worth answering is whether adding
render-style data costs us anything on real photos, and that needs the val set
held fixed.

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
from torch.utils.data import ConcatDataset, DataLoader, Subset

from .dataset import GarmentDataset
from .metrics import IoUAccumulator
from .model import PanelSegNet, TRAIN_CLASSES, NUM_TRAIN_CLASSES, count_params


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="real data; this is what gets split into train and val")
    ap.add_argument("--synth-manifest", default=None,
                    help="optional extra manifest (src.synth output), added to TRAINING only")
    ap.add_argument("--synth-frac", type=float, default=1.0,
                    help="fraction of the synthetic manifest to use, 0-1")
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

    train_parts = [Subset(full, train_idx)]
    n_synth = 0
    if args.synth_manifest:
        synth = GarmentDataset(args.synth_manifest, size=args.size, train=True, seed=args.seed)
        keep = list(range(len(synth)))
        random.Random(args.seed + 1).shuffle(keep)
        keep = keep[: int(len(keep) * max(0.0, min(1.0, args.synth_frac)))]
        train_parts.append(Subset(synth, keep))
        n_synth = len(keep)

    train_set = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    train_loader = DataLoader(train_set, batch_size=args.batch,
                              shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=args.batch,
                            shuffle=False, num_workers=2)

    if n_synth:
        share = n_synth / (len(train_idx) + n_synth)
        print(f"train {len(train_idx)} real + {n_synth} synthetic "
              f"({share:.0%} synthetic)  |  val {len(val_idx)} real only")
    else:
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
    best_report = None
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
        acc = IoUAccumulator(NUM_TRAIN_CLASSES, TRAIN_CLASSES)
        with torch.no_grad():
            for img, mask in val_loader:
                pred = model(img.to(device)).argmax(1)
                acc.update(pred, mask)
        vmiou = acc.miou()                          # panels only: the brief's metric
        vmiou_bg = acc.miou(include_background=True)
        avg_loss = run / max(1, len(train_loader))
        print(f"epoch {ep+1:02d}/{args.epochs}  loss {avg_loss:.4f}  "
              f"val_mIoU(panels) {vmiou:.4f}  val_mIoU(incl bg) {vmiou_bg:.4f}")

        if vmiou > best:
            best = vmiou
            best_report = acc.report()
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            # Save the FULL state dict (encoder + decoder) so predict.py needs no
            # download at inference time.
            torch.save({"state_dict": model.state_dict(),
                        "val_miou": vmiou,
                        "val_miou_with_background": vmiou_bg,
                        "per_class_iou": {k: v[0] for k, v in acc.per_class().items()},
                        "size": args.size,
                        "seed": args.seed}, args.out)
            print(f"  saved {args.out} (val_mIoU {vmiou:.4f})")

    print(f"best val_mIoU (panels only) {best:.4f}")
    if best_report:
        print()
        print(best_report)


if __name__ == "__main__":
    main()
