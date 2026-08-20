"""
Tests for the synthetic render generator.

Synthetic training data is only useful if it is trustworthy, and the ways it goes
wrong are quiet: a mask that drifts off its image, a label id that should not
exist, a "varied" distribution that turns out to emit the same garment every
time. None of those raise. So they get pinned here.

The distribution assertions matter as much as the correctness ones. The brief's
held-out set contains a garment with an absent panel (sleeveless, collarless), so
the generator has to actually produce those - if it silently stopped, training
would quietly lose the one signal that teaches "sometimes there is no sleeve".
"""

import numpy as np
from scipy import ndimage

from src.synth import CLASS_BODY, CLASS_COLLAR, CLASS_SLEEVE, NUM_CLASSES, make_sample

SIZE = 128  # small, so the suite stays fast


def _sample(i):
    return make_sample(1000 + i, SIZE)


def test_same_seed_gives_identical_output():
    a_img, a_lab = make_sample(7, SIZE)
    b_img, b_lab = make_sample(7, SIZE)
    assert np.array_equal(a_img, b_img)
    assert np.array_equal(a_lab, b_lab)


def test_different_seeds_give_different_garments():
    _, a = make_sample(1, SIZE)
    _, b = make_sample(2, SIZE)
    assert not np.array_equal(a, b)


def test_shapes_and_label_ids_are_valid():
    img, lab = _sample(0)
    assert img.shape == (SIZE, SIZE, 3) and img.dtype == np.uint8
    assert lab.shape == (SIZE, SIZE) and lab.dtype == np.uint8
    assert set(np.unique(lab)).issubset(set(range(NUM_CLASSES)))


def test_every_sample_has_a_body():
    """A garment with no body is a generator bug, not a valid garment."""
    for i in range(12):
        _, lab = _sample(i)
        assert (lab == CLASS_BODY).sum() > 0, f"sample {i} has no body"


def test_absent_panels_actually_occur():
    """
    The held-out set includes a sleeveless or collarless garment. If the
    generator stops producing them the model never learns that a panel can be
    legitimately absent, so this asserts both variants show up.
    """
    n = 60
    sleeveless = collarless = 0
    for i in range(n):
        _, lab = _sample(i)
        if not (lab == CLASS_SLEEVE).any():
            sleeveless += 1
        if not (lab == CLASS_COLLAR).any():
            collarless += 1
    assert 1 <= sleeveless <= n * 0.6, f"sleeveless share looks wrong: {sleeveless}/{n}"
    assert 1 <= collarless <= n * 0.8, f"collarless share looks wrong: {collarless}/{n}"


def test_sleeves_stay_attached_to_the_body():
    """
    A sleeve that floats free of the shoulder is a rendering artefact, and it
    would teach the model that sleeves are detached rectangles. Two sleeves means
    at most two connected components.
    """
    for i in range(30):
        _, lab = _sample(i)
        sleeve = lab == CLASS_SLEEVE
        if not sleeve.any():
            continue
        n_comp = ndimage.label(sleeve)[1]
        assert n_comp <= 2, f"sample {i}: sleeves split into {n_comp} components"


def test_labels_are_varied_not_a_fixed_template():
    """Guards against the geometry randomisation silently collapsing."""
    areas = []
    for i in range(15):
        _, lab = _sample(i)
        areas.append((lab == CLASS_BODY).sum())
    assert np.std(areas) > 0.05 * np.mean(areas), "body size barely varies between samples"


def test_mask_and_image_agree_on_where_the_garment_is():
    """
    The garment must be visually distinguishable from the backdrop wherever the
    mask says there is garment. If the mask drifted off the image this fails.
    """
    for i in range(8):
        img, lab = _sample(i)
        grey = img.mean(axis=2)
        inside, outside = lab > 0, lab == 0
        if inside.sum() < 50 or outside.sum() < 50:
            continue
        # backdrops are light, garments are coloured: the two must differ clearly
        assert abs(grey[inside].mean() - grey[outside].mean()) > 12, (
            f"sample {i}: mask and image do not agree on the garment location")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all synth tests passed")
