"""
predict.py - garment image -> single-channel panel mask PNG.

Mask index mapping (also in README and src/panels.py):
    0 background   1 front_body   2 back_body
    3 left_sleeve  4 right_sleeve  5 collar

Runs on CPU. Prints per-image latency and the hardware string.

Examples
--------
    # write a mask for one image
    python predict.py --image shirt.jpg --out shirt_mask.png

    # ask only for one panel; if the model didn't predict it you still get a
    # valid (possibly empty) mask, and the program does not raise.
    python predict.py --image tank_top.jpg --out mask.png --panel left_sleeve
"""

import argparse
import platform
import time

import numpy as np
import torch
from PIL import Image

from src.model import PanelSegNet, NUM_TRAIN_CLASSES
from src.panels import PANEL_IDS, to_output_mask, get_panel_region

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model(weights_path, device="cpu"):
    model = PanelSegNet(pretrained=False, freeze_encoder=True)
    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval().to(device)
    size = ckpt.get("size", 256) if isinstance(ckpt, dict) else 256
    return model, size


def preprocess(image_path, size):
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    small = img.resize((size, size), Image.BILINEAR)
    x = np.asarray(small, dtype=np.float32) / 255.0
    x = (x - _MEAN) / _STD
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)
    return x, (orig_h, orig_w)


def predict_mask(model, x, orig_hw):
    with torch.no_grad():
        logits = model(x)
    pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)  # 4-class
    # upsample class map back to original size with nearest (labels, not values)
    pred_img = Image.fromarray(pred).resize((orig_hw[1], orig_hw[0]), Image.NEAREST)
    pred = np.asarray(pred_img, dtype=np.uint8)
    return to_output_mask(pred)  # 6-id output mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="weights/model.pt")
    ap.add_argument("--panel", default=None,
                    help="optional: emit only this panel's mask (still valid if absent)")
    args = ap.parse_args()

    device = "cpu"  # inference on CPU per the brief
    model, size = load_model(args.weights, device)

    x, orig_hw = preprocess(args.image, size)

    t0 = time.perf_counter()
    mask = predict_mask(model, x, orig_hw)
    dt = (time.perf_counter() - t0) * 1000

    if args.panel is not None:
        if args.panel not in PANEL_IDS:
            raise KeyError(f"unknown panel '{args.panel}'. valid: {list(PANEL_IDS)}")
        region = get_panel_region(mask, args.panel)  # all-False if absent -> empty mask
        out = np.where(region, PANEL_IDS[args.panel], 0).astype(np.uint8)
        if not region.any():
            print(f"note: panel '{args.panel}' is absent; writing an empty mask.")
        mask = out

    Image.fromarray(mask).save(args.out)

    present = sorted({int(v) for v in np.unique(mask) if v != 0})
    names = [k for k, v in PANEL_IDS.items() if v in present]
    print(f"wrote {args.out}")
    print(f"panels present: {names if names else '(none)'}")
    print(f"latency: {dt:.1f} ms/image on CPU ({platform.processor() or platform.machine()})")


if __name__ == "__main__":
    main()
