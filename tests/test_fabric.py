"""
Tests that earn their place: they check the two things the brief tests hardest.

  1. Determinism   - same inputs -> byte-identical output.
  2. Targeting     - "left_sleeve" fills the left sleeve and nothing else;
                     "right_sleeve" fills the right sleeve. Left != right.
  3. Absent panel  - asking for a panel that isn't there returns the image
                     unchanged and does not raise.
  4. Occlusion     - a sleeve broken into several pieces by drape keeps all of
                     its pixels and still lands on the correct side. The
                     held-out set explicitly includes an occluded panel.

We build a tiny synthetic mask so the test needs no model and no dataset.
"""

import numpy as np

from src.panels import (PANEL_IDS, to_output_mask, split_sleeves,
                        resolve_panel_name, get_panel_region, TRAIN_IDX)
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


def _occluded_scene(h=64, w=64):
    """Like the scene above, but drape splits the right sleeve into two blobs."""
    pred = np.zeros((h, w), dtype=np.uint8)
    pred[16:56, 24:40] = TRAIN_IDX["body"]      # centre body
    pred[24:44, 4:16] = TRAIN_IDX["sleeve"]     # LEFT sleeve, one piece
    pred[24:32, 48:60] = TRAIN_IDX["sleeve"]    # RIGHT sleeve, upper piece
    pred[36:44, 48:60] = TRAIN_IDX["sleeve"]    # RIGHT sleeve, lower piece
    return pred


def test_occluded_sleeve_keeps_every_piece():
    pred = _occluded_scene()
    mask = to_output_mask(pred)

    sleeve_px = pred == TRAIN_IDX["sleeve"]
    labelled = (mask == PANEL_IDS["left_sleeve"]) | (mask == PANEL_IDS["right_sleeve"])
    # no sleeve pixel may be silently dropped to background
    assert np.array_equal(sleeve_px, labelled), (
        f"{int(sleeve_px.sum() - labelled.sum())} sleeve pixels were lost")

    # both right-hand blobs must be on the right, the single left blob on the left
    right = mask == PANEL_IDS["right_sleeve"]
    assert right[24:32, 48:60].all() and right[36:44, 48:60].all()
    assert (mask[24:44, 4:16] == PANEL_IDS["left_sleeve"]).all()


def test_offcentre_garment_still_splits():
    """Both sleeves left of the image centre, and no body to anchor on."""
    sleeve = np.zeros((64, 64), dtype=bool)
    sleeve[20:40, 4:10] = True    # centroid x ~6.5
    sleeve[20:40, 14:20] = True   # centroid x ~16.5
    left, right = split_sleeves(sleeve, body_mask=None)
    assert left.any() and right.any(), "widest-gap fallback should still split these"
    assert np.where(left)[1].mean() < np.where(right)[1].mean()


def test_panel_names_tolerate_realistic_spellings():
    """
    The mask contract is driven by someone else's string. Failing an integration
    on "Left Sleeve" would be a self-inflicted wound, so spacing, case and
    hyphen/underscore are all normalised.
    """
    for spelling in ["left_sleeve", "Left_Sleeve", "left sleeve", "LEFT SLEEVE",
                     "left-sleeve", "  Left Sleeve  "]:
        assert resolve_panel_name(spelling) == "left_sleeve", spelling

    mask = to_output_mask(_synthetic_scene())
    canonical = get_panel_region(mask, "left_sleeve")
    assert np.array_equal(get_panel_region(mask, "Left Sleeve"), canonical)

    img = np.full((64, 64, 3), 200, np.uint8)
    swatch = np.full((8, 8, 3), np.array([255, 0, 0]), np.uint8)
    assert np.array_equal(apply_fabric(img, mask, "Left-Sleeve", swatch),
                          apply_fabric(img, mask, "left_sleeve", swatch))


def test_unknown_panel_name_still_raises():
    """
    Absent and unknown are different failures. An absent panel returns an empty
    mask (tested above); a name we do not recognise is a caller bug, and staying
    silent about it would look exactly like a model that found nothing.
    """
    mask = to_output_mask(_synthetic_scene())
    for bad in ["sleeve", "left_arm", "", None, 3]:
        try:
            resolve_panel_name(bad)
        except KeyError:
            continue
        raise AssertionError(f"expected KeyError for {bad!r}")


def test_absent_panel_is_noop_whatever_the_spelling():
    mask = to_output_mask(_synthetic_scene())  # has no back_body
    img = np.full((64, 64, 3), 90, np.uint8)
    swatch = np.full((4, 4, 3), 10, np.uint8)
    assert np.array_equal(apply_fabric(img, mask, "Back Body", swatch), img)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
