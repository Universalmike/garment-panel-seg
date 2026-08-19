"""
Part 2: deterministic fabric fill.

Given a garment image, a predicted panel mask, a panel name, and a fabric swatch,
paste the fabric into exactly that panel. No model, no randomness: the same
inputs always give the same output.

The whole point of the exercise is targeting: apply_fabric(..., "left_sleeve")
must colour the left sleeve, every time. Which pixels are "left_sleeve" is
already decided in panels.py by geometry, so this function just has to fill the
region it is handed, deterministically.

Flat tiling only (as the brief allows). Making it drape with the render's
shading is Part 3, and is described but not implemented.
"""

import numpy as np
from PIL import Image

from .panels import get_panel_region


def _tile_swatch(swatch_rgb, h, w):
    """Tile a swatch to cover an (h, w) canvas, anchored at the top-left.

    Deterministic: fixed origin, integer tiling, no interpolation choices that
    depend on anything but the inputs.
    """
    sh, sw = swatch_rgb.shape[:2]
    reps_y = int(np.ceil(h / sh))
    reps_x = int(np.ceil(w / sw))
    tiled = np.tile(swatch_rgb, (reps_y, reps_x, 1))
    return tiled[:h, :w]


def apply_fabric(image, mask, panel_name, swatch, alpha=1.0):
    """
    Fill one named panel of `image` with `swatch`.

    Parameters
    ----------
    image : PIL.Image or np.ndarray (H, W, 3), uint8 RGB
    mask  : np.ndarray (H, W) uint8, the 6-id output mask from predict.py
    panel_name : one of front_body / back_body / left_sleeve / right_sleeve / collar
    swatch : PIL.Image or np.ndarray (h, w, 3), uint8 RGB fabric tile
    alpha  : blend strength in [0, 1]. 1.0 = fully replace the panel pixels.

    Returns
    -------
    np.ndarray (H, W, 3) uint8 : the composited image.

    Behaviour on an absent panel: the region is empty, so the image is returned
    unchanged. No exception is raised.
    """
    img = np.asarray(image, dtype=np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img[..., :3].copy()

    swatch_rgb = np.asarray(swatch, dtype=np.uint8)[..., :3]
    mask = np.asarray(mask)

    region = get_panel_region(mask, panel_name)  # all-False if absent
    if not region.any():
        return img  # nothing to fill, unchanged

    h, w = img.shape[:2]
    tiled = _tile_swatch(swatch_rgb, h, w).astype(np.float32)

    a = float(alpha)
    base = img.astype(np.float32)
    blended = (1.0 - a) * base + a * tiled
    img[region] = np.clip(blended[region], 0, 255).astype(np.uint8)
    return img


def load_rgb(path):
    """Small helper: load any image file as an (H, W, 3) uint8 RGB array."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
