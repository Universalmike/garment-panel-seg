"""
Tests for the evaluation metric.

The point of these is to pin down the two things the brief's wording actually
requires, both of which are easy to get subtly wrong and neither of which shows
up as a crash:

  1. The average is per IMAGE, not over a whole batch. A batch-aggregated IoU
     lets one big image swamp a small one; the brief says "averaged across the
     panels present in that image".
  2. Background is not a panel and is excluded by default.
"""

import numpy as np

from src.metrics import IoUAccumulator, image_class_ious, per_image_miou

NUM_CLASSES = 4  # background, body, sleeve, collar


def _tiny_pair():
    """
    Two 10x10 images that a batch-aggregated metric scores very differently to a
    per-image one.

      image A: a 2x2 patch of class 1, predicted perfectly       -> class1 IoU 1.0
      image B: an 8x8 patch of class 1, missed completely        -> class1 IoU 0.0

    Per image: (1.0 + 0.0) / 2 = 0.5
    Aggregated over the batch: inter 4 / union 68 = 0.059, because B's larger
    area dominates. The two numbers disagree by an order of magnitude.
    """
    gt_a = np.zeros((10, 10), np.uint8)
    gt_a[0:2, 0:2] = 1
    pred_a = gt_a.copy()

    gt_b = np.zeros((10, 10), np.uint8)
    gt_b[0:8, 0:8] = 1
    pred_b = np.zeros((10, 10), np.uint8)  # predicts all background

    return np.stack([pred_a, pred_b]), np.stack([gt_a, gt_b])


def test_average_is_per_image_not_per_batch():
    pred, gt = _tiny_pair()
    assert per_image_miou(pred, gt, NUM_CLASSES) == 0.5

    # what the old batch-aggregated version would have produced, for contrast
    inter = np.logical_and(pred == 1, gt == 1).sum()
    union = np.logical_or(pred == 1, gt == 1).sum()
    assert abs(inter / union - 0.0588) < 0.001


def test_background_excluded_by_default():
    pred, gt = _tiny_pair()
    panels_only = per_image_miou(pred, gt, NUM_CLASSES)
    with_bg = per_image_miou(pred, gt, NUM_CLASSES, include_background=True)
    # image A scores 1.0 either way; image B's easy background (IoU 0.36) lifts
    # the inclusive number well above the panels-only one
    assert panels_only == 0.5
    assert abs(with_bg - 0.59) < 1e-9
    assert with_bg > panels_only


def test_classes_absent_from_ground_truth_are_skipped():
    gt = np.zeros((8, 8), np.uint8)
    gt[0:4, 0:4] = 1               # only background and class 1 present
    pred = gt.copy()
    ious = image_class_ious(pred, gt, NUM_CLASSES)
    assert set(ious) == {0, 1}, "collar/sleeve are absent and must not be scored"
    assert ious[1] == 1.0


def test_false_positive_on_an_absent_class_still_costs_us():
    """
    A class we predict but that isn't in the ground truth is not scored directly
    (it isn't "present in that image"), but it steals pixels from a class that
    IS present, so it must lower the score.
    """
    gt = np.zeros((8, 8), np.uint8)
    gt[0:4, 0:4] = 1
    clean = gt.copy()
    hallucinating = gt.copy()
    hallucinating[0:2, 0:2] = 3    # invent some collar inside the body region

    assert per_image_miou(hallucinating, gt, NUM_CLASSES) < per_image_miou(clean, gt, NUM_CLASSES)


def test_accumulator_reports_per_class_and_counts():
    pred, gt = _tiny_pair()
    acc = IoUAccumulator(NUM_CLASSES, ["background", "body", "sleeve", "collar"])
    acc.update(pred, gt)

    per_class = acc.per_class()
    body_iou, body_n = per_class["body"]
    assert body_n == 2                       # class 1 present in both images
    assert abs(body_iou - 0.5) < 1e-9        # mean of 1.0 and 0.0

    sleeve_iou, sleeve_n = per_class["sleeve"]
    assert sleeve_n == 0 and np.isnan(sleeve_iou)

    assert acc.miou() == 0.5
    assert acc.miou(include_background=True) > acc.miou()
    assert "collar" in acc.report()


def test_perfect_prediction_scores_one():
    gt = np.zeros((16, 16), np.uint8)
    gt[2:8, 2:8] = 1
    gt[10:14, 2:6] = 2
    gt[0:2, 0:4] = 3
    assert per_image_miou(gt.copy(), gt, NUM_CLASSES) == 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all metric tests passed")
