"""
predict.py - garment image -> single-channel panel mask PNG.

Mask index mapping (also in README and src/panels.py):
    0 background   1 front_body   2 back_body
    3 left_sleeve  4 right_sleeve  5 collar

Runs on CPU. Prints per-image latency and the hardware string.

Two inference details worth knowing about:

  - The logits are upsampled to the image's native resolution BEFORE argmax, not
    after. Taking argmax at 256x256 and then resizing the class map with
    nearest-neighbour throws away all sub-block boundary information and leaves
    visibly stepped edges - which costs IoU exactly where thin classes like
    collar live.

  - Horizontal-flip test-time augmentation is ON by default (--no-tta disables
    it). It was measured on the held-out split before being turned on; see
    USE_TTA_BY_DEFAULT below.

    It is worth knowing that TTA is only *possible* here because the network is
    deliberately left/right agnostic: it predicts a single generic "sleeve"
    class, so a mirrored pass carries identical label semantics and the two can
    simply be averaged. A network trained to emit left_sleeve and right_sleeve
    directly could not do this at all. Left/right is resolved afterwards from
    geometry, untouched by TTA. It costs one extra forward pass and stays fully
    deterministic.

Examples
--------
    # write a mask for one image
    python predict.py --image shirt.jpg --out shirt_mask.png

    # ask only for one panel; if the model didn't predict it you still get a
    # valid (possibly empty) mask, and the program does not raise.
    python predict.py --image tank_top.jpg --out mask.png --panel left_sleeve
"""

import argparse
import os
import platform
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.model import PanelSegNet, NUM_TRAIN_CLASSES
from src.panels import PANEL_IDS, to_output_mask, get_panel_region, resolve_panel_name

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ---- TTA DEFAULT -----------------------------------------------------------
# ON, because it was measured rather than assumed. On the 111-image held-out
# split (python -m src.evaluate --compare):
#
#                        TTA off    TTA on
#     body                0.5993    0.6134
#     sleeve              0.5125    0.5304
#     collar              0.1669    0.1606
#     mIoU (panels only)  0.5188    0.5330
#
# It costs one extra forward pass, roughly doubling latency, and buys +0.014
# mean IoU - carried by body and sleeve, very slightly negative on collar. The
# brief sets no latency threshold, only asks that we report it, so the accuracy
# is worth the milliseconds.
#
# Worth recording that this was nearly shipped the other way. A probe on crude
# synthetic renders had TTA making things much worse, on the reasoning that
# averaging two disagreeing near-empty predictions lands on empty. That probe was
# simply not representative of the real distribution. The lesson is the obvious
# one: measure on held-out data, not on whatever is convenient to generate.
#
# Set to False (or pass --no-tta) to halve latency at a small accuracy cost.
USE_TTA_BY_DEFAULT = True


def load_model(weights_path, device="cpu"):
    if not os.path.exists(weights_path):
        raise SystemExit(
            f"""no checkpoint at '{weights_path}'.
The trained weights ship in weights/model.pt. If you are working from a fresh
clone that does not have them, train one with the commands in the README section
'How to reproduce training', or point --weights at a checkpoint you already have."""
        )
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


def predict_mask(model, x, orig_hw, tta=USE_TTA_BY_DEFAULT):
    """
    Image tensor -> 6-id output mask at the image's native resolution.

    Order matters here: average the (optionally flip-augmented) logits, upsample
    them bilinearly to full resolution, and only then take argmax. Argmax first
    would quantise the decision at 256x256 and leave the upsampler nothing to
    work with but blocky labels.
    """
    with torch.no_grad():
        logits = model(x)
        if tta:
            flipped = model(torch.flip(x, dims=[3]))
            logits = (logits + torch.flip(flipped, dims=[3])) / 2.0
        logits = F.interpolate(logits, size=orig_hw, mode="bilinear", align_corners=False)
    pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)  # 4-class, full res
    return to_output_mask(pred)  # 6-id output mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="weights/model.pt")
    ap.add_argument("--panel", default=None,
                    help="optional: emit only this panel's mask (still valid if absent)")
    ap.add_argument("--tta", dest="tta", action="store_true", default=None,
                    help="enable horizontal-flip TTA (one extra forward pass)")
    ap.add_argument("--no-tta", dest="tta", action="store_false",
                    help="disable horizontal-flip TTA")
    args = ap.parse_args()

    device = "cpu"  # inference on CPU per the brief
    model, size = load_model(args.weights, device)

    x, orig_hw = preprocess(args.image, size)

    tta = USE_TTA_BY_DEFAULT if args.tta is None else args.tta
    t0 = time.perf_counter()
    mask = predict_mask(model, x, orig_hw, tta=tta)
    dt = (time.perf_counter() - t0) * 1000

    if args.panel is not None:
        panel = resolve_panel_name(args.panel)       # raises only on an unknown name
        region = get_panel_region(mask, panel)       # all-False if absent -> empty mask
        out = np.where(region, PANEL_IDS[panel], 0).astype(np.uint8)
        if not region.any():
            print(f"note: panel '{panel}' is absent; writing an empty mask.")
        mask = out

    Image.fromarray(mask).save(args.out)

    present = sorted({int(v) for v in np.unique(mask) if v != 0})
    names = [k for k, v in PANEL_IDS.items() if v in present]
    print(f"wrote {args.out}")
    print(f"panels present: {names if names else '(none)'}")
    hw = platform.processor() or platform.machine()
    print(f"latency: {dt:.1f} ms/image on CPU ({hw}), flip TTA {'on' if tta else 'off'}")


if __name__ == "__main__":
    main()
