"""
Dataset for image/mask pairs listed in a manifest.json (see prepare_data.py).

Augmentation is deliberately conservative and geometry-aware:
  - We DO use horizontal flips, but a flip swaps left and right. Since the model
    predicts a single generic "sleeve" class (not left/right), flipping is safe
    and free extra data. Left/right is resolved later from geometry, so nothing
    downstream is confused by it.
  - Colour jitter helps a little with the photo->render domain gap, but we keep
    it mild so we do not destroy fabric cues the model may rely on.
"""

import json
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class GarmentDataset(Dataset):
    def __init__(self, manifest_path, size=256, train=True, seed=42):
        with open(manifest_path) as f:
            self.items = json.load(f)
        self.size = size
        self.train = train
        random.seed(seed)

    def __len__(self):
        return len(self.items)

    def _augment(self, img, mask):
        # horizontal flip
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        # mild colour jitter on the image only
        if random.random() < 0.5:
            from PIL import ImageEnhance
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
            img = ImageEnhance.Color(img).enhance(random.uniform(0.85, 1.15))
        return img, mask

    def __getitem__(self, i):
        it = self.items[i]
        img = Image.open(it["image"]).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        mask = Image.open(it["mask"]).resize((self.size, self.size), Image.NEAREST)

        if self.train:
            img, mask = self._augment(img, mask)

        img = np.asarray(img, dtype=np.float32) / 255.0
        # ImageNet normalisation (encoder was trained with it)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = torch.from_numpy(img.transpose(2, 0, 1))

        mask = torch.from_numpy(np.asarray(mask, dtype=np.int64))
        return img, mask


def denormalize(img_tensor):
    """Undo ImageNet normalisation -> uint8 RGB array (for visual checks)."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = img_tensor.numpy().transpose(1, 2, 0) * std + mean
    return np.clip(x * 255, 0, 255).astype(np.uint8)
