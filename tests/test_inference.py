"""
Tests for the inference path in predict.py, using a stub network so they need no
checkpoint and no dataset.

What these pin down:

  1. The mask comes back at the image's NATIVE resolution, because the logits are
     upsampled before argmax rather than the class map after it.
  2. The flip-TTA bookkeeping is right. This is the classic place to introduce a
     silent bug - flip the input on one axis and the output back on another, and
     you get a plausible-looking mask that is quietly wrong. We test it with a
     network that is exactly equivariant to a horizontal flip, for which TTA must
     be a no-op. If the un-flip were wrong, it would not be.
  3. Inference is deterministic with TTA on, since TTA is part of what the
     graders will run.
"""

import numpy as np
import torch
import torch.nn as nn

from predict import predict_mask
from src.panels import PANEL_IDS


class FlipEquivariantStub(nn.Module):
    """
    Class scores that depend only on each pixel's own value, never on where it
    sits. Such a network commutes with a horizontal flip, so averaging a normal
    and a mirrored pass must return exactly the normal pass.
    """

    def forward(self, x):
        b, _, h, w = x.shape
        v = x.mean(dim=1, keepdim=True)
        logits = torch.zeros(b, 4, h, w, dtype=x.dtype)
        logits[:, 0:1] = -v      # background wins on dark pixels
        logits[:, 2:3] = v       # sleeve wins on bright pixels
        return logits


class LeftHalfStub(nn.Module):
    """Ignores its input and always calls the image-left half 'sleeve'."""

    def forward(self, x):
        b, _, h, w = x.shape
        logits = torch.zeros(b, 4, h, w, dtype=x.dtype)
        logits[:, 0] = 1.0
        logits[:, 2, :, : w // 2] = 5.0
        return logits


def _input(h=16, w=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, h, w, generator=g) * 2 - 1


def test_mask_is_returned_at_native_resolution():
    x = _input(16, 16)
    mask = predict_mask(FlipEquivariantStub(), x, (48, 60))
    assert mask.shape == (48, 60), "logits must be upsampled to the image size"
    assert mask.dtype == np.uint8


def test_tta_is_a_noop_for_a_flip_equivariant_network():
    x = _input(16, 16)
    m = FlipEquivariantStub()
    with_tta = predict_mask(m, x, (32, 32), tta=True)
    without_tta = predict_mask(m, x, (32, 32), tta=False)
    assert np.array_equal(with_tta, without_tta), "flip/un-flip bookkeeping is wrong"


def test_tta_symmetrises_a_one_sided_network():
    """
    Sanity check that TTA is actually doing something: a network that only ever
    fires on the image-left half should, once mirrored and averaged, fire on
    both halves. Without this, the test above could pass on a TTA that silently
    did nothing at all.
    """
    x = _input(16, 16)
    m = LeftHalfStub()
    without_tta = predict_mask(m, x, (32, 32), tta=False)
    with_tta = predict_mask(m, x, (32, 32), tta=True)

    # Look well clear of the midline: bilinear upsampling of the logits smears
    # the decision boundary a few pixels either side of it, which is the whole
    # point of upsampling before argmax rather than after.
    sleeves = (PANEL_IDS["left_sleeve"], PANEL_IDS["right_sleeve"])
    far_right = slice(24, None)
    assert not np.isin(without_tta[:, far_right], sleeves).any(), "stub should ignore the right half"
    assert np.isin(with_tta[:, far_right], sleeves).all(), "TTA should have mirrored it across"


def test_inference_is_deterministic_with_tta_on():
    x = _input(16, 16)
    m = FlipEquivariantStub()
    a = predict_mask(m, x, (40, 40), tta=True)
    b = predict_mask(m, x, (40, 40), tta=True)
    assert np.array_equal(a, b)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all inference tests passed")
