"""
Tests that earn their place: they check the two things the brief tests hardest.

  1. Determinism   - same inputs -> byte-identical output.
  2. Targeting     - "left_sleeve" fills the left sleeve and nothing else;
                     "right_sleeve" fills the right sleeve. Left != right.
  3. Absent panel  - asking for a panel that isn't there returns the image
                     unchanged and does not raise.

We build a tiny synthetic mask so the test needs no model and no dataset.
"""

import numpy as np

from src.panels import PANEL_IDS, to_output_mask, TRAIN_IDX
from src.fabric import apply_fabric


def _synthetic_scene(h=64, w=64):
    """A 4-class prediction: body centre, a sleeve blob on each side, collar top."""
    pred = np.zeros((h, w), dtype=np.uint8)
    pred[16:56, 24:40] = TRAIN_IDX["body"]      # centre body
    pred[8:16, 26:38] = TRAIN_IDX["collar"]     # collar strip near top
    pred[24:44, 4:16] = TRAIN_IDX["sleeve"]     # LEFT sleeve blob
    pred[24:44, 48:60] = TRAIN_IDX["sleeve"]    # RIGHT sleeve blob
    return pred


def test_left_right_targeting():
    mask = to_output_mask(_synthetic_scene())
    # geometry split should produce both sleeves in the right image halves
    left = mask == PANEL_IDS["left_sleeve"]
    right = mask == PANEL_IDS["right_sleeve"]
    assert left.any() and right.any()
    # left sleeve centroid must be on the image-left of the right sleeve
    lx = np.where(left)[1].mean()
    rx = np.where(right)[1].mean()
    assert lx < rx, f"left sleeve ({lx:.1f}) should be left of right sleeve ({rx:.1f})"


def test_fill_lands_only_in_named_panel():
    mask = to_output_mask(_synthetic_scene())
    img = np.full((64, 64, 3), 200, np.uint8)          # grey garment
    swatch = np.full((8, 8, 3), np.array([255, 0, 0]), np.uint8)  # red fabric

    out = apply_fabric(img, mask, "left_sleeve", swatch)
    changed = np.any(out != img, axis=-1)
    left_region = mask == PANEL_IDS["left_sleeve"]

    # every changed pixel is inside the left sleeve, and the whole left sleeve changed
    assert np.array_equal(changed, left_region)
    # the right sleeve was untouched
    assert not changed[mask == PANEL_IDS["right_sleeve"]].any()


def test_determinism():
    mask = to_output_mask(_synthetic_scene())
    img = np.full((64, 64, 3), 128, np.uint8)
    swatch = np.random.RandomState(0).randint(0, 255, (5, 5, 3), np.uint8)
    a = apply_fabric(img, mask, "right_sleeve", swatch)
    b = apply_fabric(img, mask, "right_sleeve", swatch)
    assert np.array_equal(a, b)


def test_absent_panel_is_noop():
    mask = to_output_mask(_synthetic_scene())  # has no back_body
    img = np.full((64, 64, 3), 90, np.uint8)
    swatch = np.full((4, 4, 3), 10, np.uint8)
    out = apply_fabric(img, mask, "back_body", swatch)  # absent -> unchanged
    assert np.array_equal(out, img)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
