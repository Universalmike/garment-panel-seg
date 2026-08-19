"""
Worked example for the Part 2 function `apply_fabric`.

It does not need a trained model: it builds a small synthetic garment scene,
fills the left sleeve with one fabric and the collar with another, and saves the
result so you can eyeball the targeting.

    python example_apply_fabric.py
    # writes example_before.png and example_after.png
"""

import numpy as np
from PIL import Image

from src.panels import to_output_mask, TRAIN_IDX
from src.fabric import apply_fabric


def build_scene(h=256, w=256):
    pred = np.zeros((h, w), np.uint8)
    pred[40:210, 90:166] = TRAIN_IDX["body"]     # body
    pred[24:44, 96:160] = TRAIN_IDX["collar"]    # collar
    pred[70:180, 20:70] = TRAIN_IDX["sleeve"]    # left sleeve
    pred[70:180, 186:236] = TRAIN_IDX["sleeve"]  # right sleeve
    return to_output_mask(pred)


def checker(color_a, color_b, tile=16, size=64):
    sw = np.zeros((size, size, 3), np.uint8)
    for y in range(size):
        for x in range(size):
            sw[y, x] = color_a if ((x // tile) + (y // tile)) % 2 == 0 else color_b
    return sw


def main():
    mask = build_scene()
    garment = np.full((256, 256, 3), 210, np.uint8)  # plain grey garment

    Image.fromarray(garment).save("example_before.png")

    denim = checker((40, 70, 130), (30, 55, 110))
    stripe = checker((200, 60, 60), (230, 230, 230))

    out = apply_fabric(garment, mask, "left_sleeve", denim)
    out = apply_fabric(out, mask, "right_sleeve", denim)
    out = apply_fabric(out, mask, "collar", stripe)
    out = apply_fabric(out, mask, "front_body", checker((90, 150, 90), (70, 130, 70)))

    Image.fromarray(out).save("example_after.png")
    print("wrote example_before.png and example_after.png")


if __name__ == "__main__":
    main()
