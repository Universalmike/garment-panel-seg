"""
The evaluation metric, written to match the brief's definition exactly.

The brief says:

    "for each panel, the area where your mask and ours overlap, divided by the
     total area the two cover between them, averaged across the panels present
     in that image"

Two details in that sentence do real work, and an earlier version of this repo
got both of them wrong:

1. "in that image" - the average is per IMAGE, then across images. Computing IoU
   over a whole batch instead lets one large, easy image dominate and dilutes
   small-class errors, because with batch=16 essentially every class is present
   somewhere in the batch.

2. "each panel" - background is not a panel. The brief also says it avoids
   metrics that "reward a model for finding nothing", and averaging in the easy
   background class does exactly that: background IoU is routinely >0.9 and
   lifts the mean regardless of whether the panels were found. So
   `per_image_miou` excludes background by default, and reports the
   background-inclusive number separately for comparison.

Classes absent from an image's ground truth are skipped for that image, per
"panels present in that image". A class predicted but absent from ground truth
still costs us, indirectly: those pixels are stolen from whichever class should
have had them, lowering that class's IoU.
"""

import numpy as np
import torch

BACKGROUND_IDX = 0


def _to_numpy(a):
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def _as_batch(pred, target):
    pred, target = _to_numpy(pred), _to_numpy(target)
    if pred.ndim == 2:
        pred, target = pred[None], target[None]
    return pred, target


def image_class_ious(pred, target, num_classes):
    """
    IoU per class for ONE image (2-D arrays of class indices).

    Returns {class_index: iou} containing only the classes present in the ground
    truth - the set the brief averages over.
    """
    pred, target = _to_numpy(pred), _to_numpy(target)
    ious = {}
    for c in range(num_classes):
        t = target == c
        if not t.any():
            continue
        p = pred == c
        union = int(np.logical_or(p, t).sum())
        inter = int(np.logical_and(p, t).sum())
        ious[c] = (inter / union) if union else 0.0
    return ious


def per_image_miou(pred, target, num_classes, include_background=False):
    """
    The brief's metric: mean over images of (mean IoU over the classes present in
    that image's ground truth). Background excluded unless asked for.

    Accepts a single image (H, W) or a batch (B, H, W).
    """
    pred, target = _as_batch(pred, target)
    scores = []
    for i in range(pred.shape[0]):
        ious = image_class_ious(pred[i], target[i], num_classes)
        if not include_background:
            ious.pop(BACKGROUND_IDX, None)
        if ious:
            scores.append(float(np.mean(list(ious.values()))))
    return float(np.mean(scores)) if scores else 0.0


class IoUAccumulator:
    """
    Accumulates IoU across a dataset so we can report a per-class breakdown, not
    just one blended number. A single mean hides the thing we most want to know:
    collar is small and thin and is where a light decoder struggles.
    """

    def __init__(self, num_classes, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self._sums = np.zeros(num_classes, dtype=np.float64)
        self._counts = np.zeros(num_classes, dtype=np.int64)
        self._image_means = []           # panels only
        self._image_means_with_bg = []

    def update(self, pred, target):
        pred, target = _as_batch(pred, target)
        for i in range(pred.shape[0]):
            ious = image_class_ious(pred[i], target[i], self.num_classes)
            for c, v in ious.items():
                self._sums[c] += v
                self._counts[c] += 1
            if ious:
                self._image_means_with_bg.append(float(np.mean(list(ious.values()))))
            panels = {c: v for c, v in ious.items() if c != BACKGROUND_IDX}
            if panels:
                self._image_means.append(float(np.mean(list(panels.values()))))

    def per_class(self):
        """{class_name: (mean_iou, n_images_the_class_appeared_in)}."""
        out = {}
        for c in range(self.num_classes):
            n = int(self._counts[c])
            out[self.class_names[c]] = (float(self._sums[c] / n) if n else float("nan"), n)
        return out

    def miou(self, include_background=False):
        vals = self._image_means_with_bg if include_background else self._image_means
        return float(np.mean(vals)) if vals else 0.0

    def report(self):
        """Human-readable summary, ready to paste into the README."""
        lines = ["per-class IoU (mean over images where the class is in ground truth):"]
        for name, (iou, n) in self.per_class().items():
            lines.append(f"  {name:<12} {iou:.4f}   ({n} images)")
        lines.append("")
        lines.append(f"mean per-class IoU, panels only  (the brief's metric) : {self.miou():.4f}")
        lines.append(f"mean per-class IoU, incl. background                  : {self.miou(True):.4f}")
        return "\n".join(lines)
