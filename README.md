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

Panel names are normalised before lookup, so `"left_sleeve"`, `"Left Sleeve"`
and `"left-sleeve"` are the same panel. Two failure modes are deliberately kept
distinct: an **absent** panel (valid name, model didn't find it) returns an empty
mask silently, while an **unknown** name raises `KeyError`. Swallowing a typo
would look exactly like a model that found nothing, which is the least useful
thing this code could do.

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
python predict.py --image tank_top.jpg --out mask.png --panel "Left Sleeve"   # same thing
```

**Optional flip TTA** (off by default — see below):
```bash
python predict.py --image shirt.jpg --out mask.png --tta
```

Two things `predict.py` does that are worth knowing:

- **Logits are upsampled to the image's native resolution before `argmax`,** not
  after. Taking `argmax` at 256×256 and then resizing the class map with
  nearest-neighbour quantises every boundary to an 8-pixel block on a 2048px
  image. Boundaries are most of the IoU on thin classes.
- **Horizontal-flip TTA is implemented and tested, but off by default.** It is
  only *possible* because the network is left/right agnostic — a model emitting
  `left_sleeve`/`right_sleeve` directly could not average a mirrored pass at all.
  It is off because we have not yet measured it on held-out data, and shipping an
  unmeasured accuracy change as the default is how you lose points you thought
  you were gaining. `python -m src.evaluate --compare` decides it; if TTA wins,
  flip `USE_TTA_BY_DEFAULT` in `predict.py`.

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
pytest -q          # 19 tests across three suites
```

- `tests/test_fabric.py` - targeting, determinism, absent panels, occlusion,
  off-centre garments, and panel-name spelling.
- `tests/test_metrics.py` - pins the two subtleties in the brief's metric
  definition (per-image averaging, background excluded).
- `tests/test_inference.py` - the inference path, using stub networks so it needs
  no checkpoint. Includes a flip-equivariant network for which TTA must be a
  no-op, which is what catches a wrong un-flip.

CI runs all of this on every push (`.github/workflows/tests.yml`), plus the
parameter-cap assertion and both worked examples, against the committed
checkpoint. That is the "clone, install, infer without emailing you" claim,
checked rather than asserted.

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
- `src/panels.py` splits it into connected pieces and labels each one **left or
  right by horizontal position**, comparing the piece's centroid x against the
  centroid of the predicted body (the garment's midline, which beats the image
  centre when the garment is off to one side of the frame). This is
  deterministic geometry, not learned guessing.

Every piece above a small speckle threshold is assigned, rather than only the
two largest — occlusion routinely breaks one sleeve into several blobs, and
dropping them would delete real sleeve pixels. If the split line somehow puts
every piece on one side, we fall back to cutting at the widest gap between
centroids.

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

**Fashionpedia**, the dataset the brief suggested. It has garment-part instance
annotations (sleeve, collar, and the main garments) at roughly the granularity we
need. `src/prepare_data.py` reads the category names from the annotation file and
builds our 4-class masks from them, so we are not depending on fragile hardcoded
category ids.

We trained on the **val2020** split (`instances_attributes_val2020.json`, 3,200
images) rather than train2020 (~45k). That was a deliberate trade inside the 24h
window: train2020 is a 20GB+ download, and getting the whole pipeline trained,
verified and documented end-to-end mattered more than the last few IoU points.
The scripts are split-agnostic - point `--ann`/`--imgs` at train2020 and nothing
else changes.

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

**A probe, and it is not encouraging.** We ran the trained model over a handful
of quickly-generated flat-shaded garment images — solid panels, plain
background, a simple lighting ramp. The model predicted background across
essentially the whole garment, at ~0.99 mean confidence, scoring near zero on
every panel. That is *not* a measurement of the production render style: these
were crude synthetic shapes, far simpler than a real 3D render, and simplicity
can be as out-of-distribution as complexity. But it points at the gap being a
step change rather than a gentle softening of the masks, and it is the first
thing we would check with real render samples in hand. See "what I'd do next".

---

## Reproducibility

All seeds are fixed (`--seed`, default 42) and cudnn is set deterministic in
`src/train.py`. Re-running training gives comparable numbers. `predict.py` and
`apply_fabric` have no randomness at all.

---

## Evaluating a checkpoint

```bash
python -m src.evaluate --manifest data/train/manifest.json          # val split
python -m src.evaluate --manifest data/train/manifest.json --compare  # with/without TTA
```

To run it on Kaggle against the shipped checkpoint, use the **Evaluation**
section at the bottom of `garmentimage-training.ipynb` — it re-clones the repo,
pulls Fashionpedia back down, rebuilds the identical manifest with seed 42 so the
val split is the genuine held-out set, and runs the comparison. No GPU and no
retraining needed.

`src/evaluate.py` scores the **deployed** pipeline - resize, forward pass,
bilinear upsample of the logits to the image's native resolution, argmax -
against the full-resolution ground-truth masks, and prints a per-class
breakdown rather than one blended figure.

It implements the brief's metric definition literally, which differs from what
this repo measured during training in two ways that both matter:

| | training metric (before) | `src/evaluate.py` (now) |
|---|---|---|
| averaging | over the whole **batch** | per **image**, then across images |
| background | included | **excluded** - it is not a panel |

Both differences inflate the number. Batch aggregation lets one large image
swamp a small one and dilutes small-class errors, since with `batch=16` nearly
every class appears somewhere in the batch. And background IoU is routinely
above 0.9, so averaging it in rewards a model for finding nothing - which the
brief explicitly says it does not want. `tests/test_metrics.py` has a worked
case where the two definitions differ by an order of magnitude.

---
## Results (validation set)

Trained on the Fashionpedia val2020 split (631 train / 111 val images) for 30
epochs at 256×256, seed 42. The full training log is in
`garmentimage-training.ipynb`.

**0.572** is the number that run reported — but it was computed with the *old*
metric: aggregated over batches, and with background included. Both of those
flatter the model, so treat 0.572 as an upper bound rather than the score.
`src/evaluate.py` reports the brief's actual definition (per-image, panels only)
for the shipped checkpoint. Run it to get the honest per-panel figures; the
training log's number is left here unedited so the two are comparable.

Honest reading of that number:
- It includes the background class, which is easy and lifts the average. Per-panel
  IoU on body/sleeve/collar is lower; body is strongest, collar (small and thin)
  is weakest.
- Training on the full train2020 set (~45k images) instead of the val split would
  be the biggest quality lever and is the first thing I'd do with more time.

---
## How to reproduce training

These are the exact commands behind the numbers reported above. The full run,
with output, is preserved in `garmentimage-training.ipynb` (Kaggle, one T4).

```bash
# 1. get Fashionpedia val2020 from the official CVDF mirror.
#    The image zip covers val and test together and unpacks to ./test (3,200 jpgs).
wget -O instances_attributes_val2020.json \
  https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json
wget -O val_test2020.zip \
  https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip
unzip -q val_test2020.zip

# 2. build 4-class masks -> 742 image/mask pairs.
#    --max-images 4000 is not binding here: only 742 images in this split carry
#    a sleeve annotation, and --require-sleeve drops the rest.
python -m src.prepare_data \
    --ann  instances_attributes_val2020.json \
    --imgs test \
    --out  data/train \
    --max-images 4000 --require-sleeve --seed 42

# 3. train -> best checkpoint to weights/model.pt (631 train / 111 val)
python -m src.train --manifest data/train/manifest.json \
    --epochs 30 --size 256 --batch 16 --seed 42 \
    --out weights/model.pt
```

To train on the full set instead, swap in `instances_attributes_train2020.json`
and `train2020.zip`, and point `--imgs` at the unpacked folder. Nothing else
changes.

The checkpoint stores the **full** model (encoder + decoder), so `predict.py`
needs no weight download at inference time. It is a few MB, safely under 100 MB,
and is committed.

---

## Latency / hardware

`predict.py` runs on CPU and prints per-image latency on every call.

Measured on an Intel Core i5-7200U (2 cores, 2 torch threads, Windows), median
of 9 runs after warm-up. The network input is always 256×256; the output
resolution varies because the logits are upsampled to the image's native size
before argmax, and that upsample is not free:

| output resolution | TTA off (default) | TTA on |
|---|---|---|
| 256 × 256 | **368 ms** | 848 ms |
| 512 × 512 | **482 ms** | 862 ms |
| 1024 × 1365 (typical photo) | **754 ms** | 1190 ms |
| 2048 × 2048 | **1756 ms** | 2006 ms |

For reference, the Kaggle CPU used during training measured 373 ms/image at
512×512 output under the older nearest-neighbour path.

Upsampling logits rather than the class map costs roughly 250 ms at photo
resolution. That buys sub-block boundary accuracy, which is where IoU is won or
lost on thin classes like collar, so it is a trade worth making — but it is a
real trade and it is stated rather than hidden.

---

## What I'd do next with two more days

- **Render-style training data, generated rather than collected.** The brief
  permits synthetic data, and procedurally drawing garment templates — front
  *and* back, random colourways, subtle shading ramps, random scale and rotation
  — yields pixel-perfect labels for all five panels essentially for free. It is
  procedural drawing, not a generative model, so it stays inside the constraint.
  This is the single biggest lever available: it attacks the domain gap, and it
  is the only route to `back_body` that does not require someone to hand-label
  back views. Mixed 50/50 with Fashionpedia, this is what I would spend the next
  day on.

- **Train on train2020.** 631 training images is very little. The full split is
  ~45k; attaching it as a Kaggle input dataset avoids the 20GB download entirely.
- **A left/right accuracy number.** Panel targeting is 25% of the score and this
  repo still has no measurement of it. Fashionpedia does not label which sleeve
  is which, but it can be derived: for images whose ground truth has exactly two
  sleeve instances, the instance with the smaller centroid x *is* the image-left
  sleeve by definition. Scoring our assignment against that gives a real number
  for the thing that matters most. (Per-panel IoU is now covered by
  `src/evaluate.py`.)
- **Better collar/small-class recall** via harder augmentation and possibly a
  boundary-weighted loss; thin classes are where a light decoder struggles most.
- **Occlusion grouping.** Sleeve pieces are currently assigned to a side one by
  one. The next step is grouping blobs that belong to the same arm (proximity
  plus adjacency to the same body edge) so a fragment on the wrong side of the
  split line is pulled back to its own sleeve.

## What I know is broken

- **`back_body` is never predicted.** Worn-garment photos almost never show the
  back, so there is no signal to learn it. The id is reserved so the contract
  stays stable, but we do not hallucinate a back panel we cannot see.
- **Left/right is image-space, not wearer-space, by default** (one-flag switch,
  documented above). If the grader's convention differs, the sleeve targeting
  test will invert until the flag is flipped.
- **Single sleeve edge case:** if only one sleeve is visible it is assigned by
  which side of the *body centroid* it falls on (image centre if no body was
  predicted). A sleeve straddling that line is still a weak spot, and a garment
  with no predicted body at all falls back to the cruder image-centre test.
- **Sleeve pieces are assigned independently.** Occlusion that breaks a sleeve
  into several blobs is handled - every blob above a 1% speckle threshold is
  kept and assigned by side, so no sleeve pixels are silently dropped - but a
  blob landing on the wrong side of the split line joins the wrong sleeve. There
  is no grouping step that says "these two blobs are one arm".
- **Domain gap** on renders (above): our own probe suggests the model may find
  almost nothing on render-style inputs, not merely produce softer masks. That is
  the largest known risk in this submission and it is measured, not guessed at.

- **The headline 0.572 is measured with the wrong metric.** It aggregates over
  batches and counts background as a panel; both inflate it. `src/evaluate.py`
  computes the brief's definition, and the honest per-panel numbers will be
  lower. The old figure is left in place rather than quietly restated.

- **Flip TTA is unmeasured**, so it ships disabled. On the synthetic probe above
  it made things worse — averaging two near-empty disagreeing predictions lands
  on empty — but that probe is not representative enough to conclude from.

---

## Use of AI tools

I used an AI assistant (Claude, via Claude Code) throughout, and used it heavily.
Being specific, since the brief asks:

- **Mine, directed by me:** the architecture decision (freeze a compact
  pretrained encoder, train only a light depthwise-separable decoder, to fit the
  2M cap), the label-set choice (train on a generic `sleeve` class rather than
  left/right), and the decision to resolve left/right from geometry afterwards.
  These are the choices the brief says it reads closely, and they are the ones I
  made.
- **Scaffolding and boilerplate:** repo structure, dataset loader, training loop,
  argument parsing, and the first draft of the README and design note.
- **Review that changed the code:** I had it audit the repo against the brief. It
  found that the sleeve splitter kept only the two largest connected components
  (silently dropping pixels when occlusion fragments a sleeve — the held-out set
  includes an occluded panel), that `predict.py` took `argmax` before upsampling,
  and that the training metric did not match the brief's stated definition. It
  wrote the fixes and the tests that pin them, including `src/metrics.py` and
  `src/evaluate.py`, under my direction.
- **Measurement:** the latency table, the parameter counts, the domain-gap probe,
  and the decision to ship flip TTA *disabled* pending a real measurement rather
  than assume it helps.

I have read everything in this repo and can walk through any of it. Where the
numbers are uncertain or the approach is a known weak point, the README says so
rather than rounding up.
