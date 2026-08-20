# Garment Panel Segmentation + Deterministic Fabric Fill

Two things, matching the brief:

1. **A small segmentation model** that takes one garment image and predicts a
   pixel mask for each visible panel.
2. **A deterministic fill function** that pastes a fabric swatch into a *named*
   panel — the same inputs always give the same output, and `"left_sleeve"`
   always means the left sleeve.

Part 3 (shading-aware fill) is written up separately in `DESIGN_NOTE.md`. No code
for it, as requested.

---

## Output mask format

`predict.py` writes a single-channel 8-bit PNG. Pixel values are panel ids:

| id | panel        |
|----|--------------|
| 0  | background   |
| 1  | front_body   |
| 2  | back_body    |
| 3  | left_sleeve  |
| 4  | right_sleeve |
| 5  | collar       |

A panel the model does not predict simply never appears in the mask. Asking for
an absent panel returns an empty mask and does **not** raise (see below).

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

Python 3.10+ works. CPU is enough for inference; a GPU only speeds up training.

## Run

**Predict a mask:**
```bash
python predict.py --image path/to/shirt.jpg --out shirt_mask.png
```

**Ask for one panel only** (still valid, and empty if the model didn't find it):
```bash
python predict.py --image tank_top.jpg --out mask.png --panel left_sleeve
```

**Fill a panel with fabric** (Part 2). Importable function, with a runnable
example:
```bash
python example_apply_fabric.py       # writes example_before.png / example_after.png
```
```python
from src.fabric import apply_fabric, load_rgb
import numpy as np
from PIL import Image

img    = load_rgb("shirt.jpg")
mask   = np.asarray(Image.open("shirt_mask.png"))   # from predict.py
swatch = load_rgb("denim.png")
out    = apply_fabric(img, mask, "left_sleeve", swatch)
Image.fromarray(out).save("filled.png")
```

**Run the tests:**
```bash
python -m tests.test_fabric
```

---

## Architecture and why

**Frozen MobileNetV2 encoder + a small trainable UNet decoder.**

- The encoder is a pretrained MobileNetV2, **frozen**. It gives general
  shape/edge/texture features for free. Because it is frozen, it does not count
  against the trainable-parameter budget.
- The decoder is a UNet-style up-path built from **depthwise-separable convs**
  (cheap on parameters) that fuses encoder skip features at each scale.

Why this shape and not a big backbone: the 2M cap is the whole point of the
exercise. You cannot bolt on a large trainable backbone and coast. Freezing a
compact pretrained encoder and training only a light decoder is the honest way
to get good features under that constraint. Depthwise-separable convs let the
decoder be reasonably wide without blowing the budget.

### Parameter count

| | count |
|---|---|
| **Trainable (decoder)** | **491,564** |
| Frozen encoder | 1,811,712 |
| **Total at inference** | **2,303,276** |

Trainable is well under the 2,000,000 cap. Reported by `python src/model.py`.

---

## Label set and why

The network is trained on **four** classes:

`background`, `body`, `sleeve`, `collar`

It is **not** trained to tell left sleeve from right sleeve. Up close the two
sleeves are near pixel-identical, so asking a small model to distinguish them
from local texture is a losing game (the brief hints at exactly this). Instead:

- the model predicts one generic `sleeve` class, then
- `predict.py` splits it into connected pieces and labels them **left/right by
  horizontal position** (centroid x). This is deterministic geometry, not
  learned guessing.

Mapping from the 4 training classes to the 6 output ids:

- `body`  → `front_body`
- `collar` → `collar`
- `sleeve` → split into `left_sleeve` / `right_sleeve` by geometry
- `back_body` is **never produced** — see *What's broken*.

### Which "left"?

The brief is deliberately ambiguous about *whose* left. We default to
**image-left** (viewer's perspective): `left_sleeve` is the sleeve on the left of
the picture. It is unambiguous from the pixels and matches how someone pointing
at the image would describe it. If your held-out set uses the **wearer's** left
(image right), flip one flag: `LEFT_IS_IMAGE_LEFT = False` in `src/panels.py`.
Nothing else changes.

---

## Data source and licence

**Fashionpedia** (`instances_attributes_train2020.json`), the dataset the brief
suggested. It has garment-part instance annotations (sleeve, collar, and the
main garments) at roughly the granularity we need. `src/prepare_data.py` reads
the category names from the annotation file and builds our 4-class masks from
them, so we are not depending on fragile hardcoded category ids.

Licence: Fashionpedia images are Creative Commons (the set is filtered to CC
licences), and the annotations are released under CC BY 4.0 by the Fashionpedia
authors. Suitable for training and for this assessment. Confirm the current
licence text on the Fashionpedia download page before any production use.

### The domain gap (called out honestly)

Fashionpedia is **photographs of worn garments** — real people, real lighting,
occlusion, folds. Your production images are **synthetic 3D renders of garments
with no body inside**. That is a real distribution shift and the model will be
weaker on the render style than the numbers on photos suggest. We did not try to
close the gap (the brief says it is not scored); the honest read is that the
geometry-based left/right logic transfers cleanly, while the learned masks will
need render-style data or light fine-tuning to be production-grade. Mild colour
jitter in training is the only nod to robustness here.

---

## Reproducibility

All seeds are fixed (`--seed`, default 42) and cudnn is set deterministic in
`src/train.py`. Re-running training gives comparable numbers. `predict.py` and
`apply_fabric` have no randomness at all.

---
## Results (validation set)

Trained on the Fashionpedia validation split (631 train / 111 val images) due to
the 24h window. Best mean IoU over the 4 training classes: 0.572.

Honest reading of that number:
- It includes the background class, which is easy and lifts the average. Per-panel
  IoU on body/sleeve/collar is lower; body is strongest, collar (small and thin)
  is weakest.
- Training on the full train2020 set (~45k images) instead of the val split would
  be the biggest quality lever and is the first thing I'd do with more time.

---
## How to reproduce training

```bash
# 1. get Fashionpedia (images + instances_attributes_train2020.json)
# 2. build 4-class masks (subset for speed)
python -m src.prepare_data \
    --ann  instances_attributes_train2020.json \
    --imgs train \
    --out  data/train \
    --max-images 4000 --require-sleeve --seed 42

# 3. train (saves best checkpoint to weights/model.pt)
python -m src.train --manifest data/train/manifest.json \
    --epochs 25 --size 256 --batch 16 --seed 42
```

The checkpoint stores the **full** model (encoder + decoder), so `predict.py`
needs no weight download at inference time. It is a few MB, safely under 100 MB,
and is committed.

---

## Latency / hardware

`predict.py` runs on CPU and prints per-image latency. On the machine used
during development (x86_64, 256×256 input) a forward pass plus post-processing
was roughly **0.5 s/image**. Report your own machine's number from the line
`predict.py` prints.

---

## What I'd do next with two more days

- **Render-style data.** Generate or collect a small set of the actual 3D-render
  style and fine-tune (or at least validate) on it. That is the single biggest
  quality lever given the domain gap.
- **Panel-aware evaluation harness.** A held-out scorer that reports IoU *per
  panel*, plus a targeted left/right accuracy number (does `left_sleeve` land on
  the correct side), rather than one averaged figure.
- **Better collar/small-class recall** via harder augmentation and possibly a
  boundary-weighted loss; thin classes are where a light decoder struggles most.
- **Occlusion handling** for overlapping pieces, since the held-out set includes
  a partially occluded panel.

## What I know is broken

- **`back_body` is never predicted.** Worn-garment photos almost never show the
  back, so there is no signal to learn it. The id is reserved so the contract
  stays stable, but we do not hallucinate a back panel we cannot see.
- **Left/right is image-space, not wearer-space, by default** (one-flag switch,
  documented above). If the grader's convention differs, the sleeve targeting
  test will invert until the flag is flipped.
- **Single sleeve edge case:** if only one sleeve is visible, it is assigned by
  which half of the image it falls in. A sleeve that crosses the centre line is
  a weak spot.
- **Domain gap** on renders (above): learned masks will be softer on the
  production style than on photos.

---

## Use of AI tools

I used an AI assistant (Claude) for: scaffolding the repo structure, drafting
boilerplate (dataset loader, training loop, argument parsing), and tightening the
README/design-note wording. The architecture decisions (frozen encoder + light
decoder to fit the cap), the label-set choice, and the geometry-based left/right
approach are mine; I directed those and can walk through any part of the code in
the follow-up. Parameter counts, tests, and the end-to-end run were verified
before submission.
