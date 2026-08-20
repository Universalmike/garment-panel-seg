"""
Procedurally generated garment renders, to close the photo -> render domain gap.

WHY THIS EXISTS
---------------
Fashionpedia is photographs of worn garments: real people, real scenes, real
occlusion. Production images are synthetic 3D renders of a garment on an
invisible form against a clean backdrop. Probing the Fashionpedia-trained model
on render-style inputs had it predicting background across the whole garment at
~0.99 confidence - the gap looks like a step change, not a softening.

The brief permits synthetic data we generate ourselves, and this is procedural
drawing plus a shading model: no diffusion, no hosted image model, nothing
generative in the sense the brief rules out. It is the same thing a renderer
does, done cheaply in 2D.

WHAT IT PRODUCES
----------------
(image, mask) pairs in exactly the format prepare_data.py emits, so they drop
straight into GarmentDataset and the existing training loop:

    0 background   1 body   2 sleeve   3 collar

Deliberately included in the distribution, because the brief says its held-out
set contains them:
  - sleeveless garments (a requested sleeve panel is genuinely absent)
  - collarless garments (ditto for collar)
  - garments where one sleeve is partly occluded by the body silhouette

WHAT MAKES IT MORE THAN COLOURED RECTANGLES
-------------------------------------------
Flat polygons would teach the network flat polygons, which is a different
out-of-distribution problem rather than a solution. Each sample gets:
  - antialiased silhouettes, built by drawing at 3x and box-filtering down
  - directional lighting, so one side of the garment is lit and the other is not
  - low-frequency fold shading from smoothed noise, at two octaves
  - per-panel ambient occlusion, which darkens seams where sleeve meets body and
    the inside of the silhouette edge
  - woven fabric texture at pixel scale
  - varied colourways, proportions, sleeve length and droop, collar size
  - varied framing: scale, translation and rotation, applied to the geometry
    rather than by resampling, so the labels stay exact
  - a soft contact shadow on the backdrop

Usage:
    python -m src.synth --out data/synth --n 4000 --seed 42
    python -m src.synth --out /tmp/peek --n 24 --seed 7 --contact-sheet sheet.png
"""

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from tqdm import tqdm

CLASS_BG, CLASS_BODY, CLASS_SLEEVE, CLASS_COLLAR = 0, 1, 2, 3
NUM_CLASSES = 4

SUPERSAMPLE = 3  # draw at 3x, box-filter down: cheap, exact antialiasing


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def _rotate(points, cx, cy, deg):
    """Rotate points about (cx, cy). Applied to geometry, never to the raster,
    so the label mask never picks up interpolation error."""
    a = np.deg2rad(deg)
    ca, sa = np.cos(a), np.sin(a)
    return [(cx + (x - cx) * ca - (y - cy) * sa,
             cy + (x - cx) * sa + (y - cy) * ca) for x, y in points]


def _body_polygon(rng, cx, top_y, hem_y, shoulder_w, hem_w):
    """A torso outline with a slight waist, rather than a trapezoid."""
    n = 8
    left, right = [], []
    waist = rng.uniform(-0.06, 0.03)
    for i in range(n + 1):
        t = i / n
        y = top_y + t * (hem_y - top_y)
        w = shoulder_w + (hem_w - shoulder_w) * t
        w *= 1.0 + waist * np.sin(np.pi * t)
        jitter = 1.0 + rng.normal(0, 0.004)
        left.append((cx - w / 2 * jitter, y))
        right.append((cx + w / 2 * jitter, y))
    return left + right[::-1]


def _sleeve_polygon(rng, shoulder, sign, length, upper_w, cuff_w, droop_deg):
    """A tapered tube hanging off one shoulder point."""
    theta = np.deg2rad(droop_deg)
    d = np.array([sign * np.cos(theta), np.sin(theta)])
    perp = np.array([-d[1], d[0]])
    start = np.array(shoulder, dtype=float)
    end = start + d * length

    n = 6
    bow_amt = length * rng.uniform(0.02, 0.09)   # sleeves hang, they are not rulers
    droop_curve = rng.uniform(-1.0, 1.0)

    def centre_at(t):
        c = start + d * (length * t)
        return c + perp * (np.sin(np.pi * t) * bow_amt * droop_curve)

    def width_at(t):
        # taper is not linear: the bicep is full, the forearm narrows faster
        return upper_w + (cuff_w - upper_w) * (t ** 1.35)

    outer, inner = [], []
    for i in range(n + 1):
        t = i / n
        c, w = centre_at(t), width_at(t)
        outer.append(tuple(c + perp * (w / 2)))
        inner.append(tuple(c - perp * (w / 2)))

    # rounded cuff, so the sleeve ends in a hem rather than a guillotine cut
    cap = []
    end_c, end_w = centre_at(1.0), width_at(1.0)
    for k in range(1, 6):
        ang = np.pi * k / 6
        v = perp * np.cos(ang) + d * np.sin(ang)
        cap.append(tuple(end_c + v * (end_w / 2)))

    return outer + cap + inner[::-1]


def _garment_geometry(rng, size):
    """All panel polygons for one garment, in output-pixel coordinates."""
    s = size
    scale = rng.uniform(0.62, 0.82)
    cx = s * rng.uniform(0.44, 0.56)
    cy = s * rng.uniform(0.46, 0.54)
    rot = rng.normal(0, 4.0)

    shoulder_w = s * scale * rng.uniform(0.42, 0.56)
    body_len = s * scale * rng.uniform(0.62, 0.90)
    hem_w = shoulder_w * rng.uniform(0.88, 1.18)
    top_y = cy - body_len * rng.uniform(0.42, 0.52)
    hem_y = top_y + body_len

    body = _body_polygon(rng, cx, top_y, hem_y, shoulder_w, hem_w)

    # neck opening, cut out of the body; the collar sits around it
    neck_w = shoulder_w * rng.uniform(0.26, 0.38)
    neck_h = neck_w * rng.uniform(0.45, 0.85)
    neck = [cx - neck_w / 2, top_y - neck_h * 0.35, cx + neck_w / 2, top_y + neck_h * 0.65]

    has_collar = rng.random() < 0.72
    collar = None
    if has_collar:
        pad = neck_w * rng.uniform(0.16, 0.34)
        collar = [neck[0] - pad, neck[1] - pad * rng.uniform(0.5, 1.2),
                  neck[2] + pad, neck[3] + pad * rng.uniform(0.4, 1.0)]

    has_sleeves = rng.random() < 0.80
    sleeves = []
    if has_sleeves:
        length = shoulder_w * rng.uniform(0.30, 0.95)     # cap sleeve .. long sleeve
        upper_w = shoulder_w * rng.uniform(0.20, 0.30)
        cuff_w = upper_w * rng.uniform(0.62, 0.95)
        droop = rng.uniform(16, 74)
        # one sleeve may sit tighter to the body, partly hiding behind it
        for sign in (-1, 1):
            sy = top_y + shoulder_w * rng.uniform(0.03, 0.10)
            # Start the sleeve INSIDE the body edge. The sleeve cap genuinely
            # overlaps the shoulder on a real garment, and starting flush with
            # the edge leaves the sleeve floating detached whenever droop or
            # body taper pulls the two apart.
            sx = cx + sign * shoulder_w / 2 * rng.uniform(0.70, 0.88)
            this_droop = droop + rng.normal(0, 6)
            this_len = length * rng.uniform(0.9, 1.1)
            sleeves.append(_sleeve_polygon(rng, (sx, sy), sign, this_len,
                                           upper_w, cuff_w, this_droop))

    geo = {"body": body, "neck": neck, "collar": collar, "sleeves": sleeves}
    for key in ("body",):
        geo[key] = _rotate(geo[key], cx, cy, rot)
    geo["sleeves"] = [_rotate(p, cx, cy, rot) for p in geo["sleeves"]]
    geo["_rot"] = rot
    geo["_centre"] = (cx, cy)
    return geo


def _ellipse_polygon(box, rot, centre, n=48):
    """Ellipses have to become polygons so they can be rotated with everything else."""
    x0, y0, x1, y1 = box
    rx, ry = (x1 - x0) / 2, (y1 - y0) / 2
    ex, ey = (x0 + x1) / 2, (y0 + y1) / 2
    pts = [(ex + rx * np.cos(2 * np.pi * i / n), ey + ry * np.sin(2 * np.pi * i / n))
           for i in range(n)]
    return _rotate(pts, centre[0], centre[1], rot)


# --------------------------------------------------------------------------
# rasterisation
# --------------------------------------------------------------------------

def _rasterise_labels(geo, size):
    """
    Per-class coverage at output resolution, computed by drawing each panel at
    SUPERSAMPLE resolution and box-filtering down. Coverage (not a hard label) is
    what gives antialiased edges; the label map is the argmax of it.
    """
    S = SUPERSAMPLE
    big = size * S
    cx, cy = geo["_centre"]
    rot = geo["_rot"]

    def draw(polys, box_polys=()):
        im = Image.new("L", (big, big), 0)
        d = ImageDraw.Draw(im)
        for p in polys:
            d.polygon([(x * S, y * S) for x, y in p], fill=255)
        for p in box_polys:
            d.polygon([(x * S, y * S) for x, y in p], fill=0)
        return np.asarray(im, dtype=np.float32) / 255.0

    neck_poly = _ellipse_polygon(geo["neck"], rot, (cx, cy))

    # Body is solid: the neck opening is not cut away, because what shows through
    # a neck hole on an invisible form is the inside back of the garment, lit by
    # nothing much. We keep the opening as a separate mask so make_sample can
    # shade it like an interior instead of like a lit outer surface.
    body = draw([geo["body"]])
    neck = draw([neck_poly]) * body
    sleeves = draw(geo["sleeves"]) if geo["sleeves"] else np.zeros_like(body)
    collar = (draw([_ellipse_polygon(geo["collar"], rot, (cx, cy))],
                   box_polys=[neck_poly])
              if geo["collar"] is not None else np.zeros_like(body))

    def down(a):
        return a.reshape(size, S, size, S).mean(axis=(1, 3))

    cov = np.stack([down(body), down(sleeves), down(collar)])  # (3, h, w)
    neck_cov = down(neck)

    # Painting order matches prepare_data.py: body, then collar, then sleeve on
    # top, so a part always wins over the body it sits on.
    labels = np.zeros((size, size), np.uint8)
    labels[cov[0] > 0.5] = CLASS_BODY
    labels[cov[2] > 0.5] = CLASS_COLLAR
    labels[cov[1] > 0.5] = CLASS_SLEEVE

    alpha = np.clip(cov.max(axis=0), 0.0, 1.0)
    return labels, alpha, neck_cov


# --------------------------------------------------------------------------
# shading
# --------------------------------------------------------------------------

def _shading(rng, labels, size):
    """
    Turn a flat label map into something that reads as lit cloth: directional
    light, two octaves of fold noise, and per-panel ambient occlusion so seams
    and silhouette edges darken.
    """
    garment = labels > 0

    # directional light across the garment
    ang = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    ramp = np.cos(ang) * xx + np.sin(ang) * yy
    ramp = (ramp - ramp.min()) / max(1e-6, np.ptp(ramp))
    directional = 0.82 + 0.36 * ramp

    # folds: smoothed noise, two scales
    def octave(sigma, amp):
        n = rng.normal(0, 1, (size, size)).astype(np.float32)
        n = ndimage.gaussian_filter(n, sigma=sigma)
        n /= max(1e-6, np.abs(n).max())
        return amp * n

    folds = octave(size * 0.045, 1.0) + octave(size * 0.014, 0.45)
    folds = 1.0 + rng.uniform(0.10, 0.22) * folds

    # per-panel AO: distance from each panel's own boundary, so sleeve/body seams
    # darken as well as the outer silhouette
    ao = np.zeros((size, size), np.float32)
    for c in (CLASS_BODY, CLASS_SLEEVE, CLASS_COLLAR):
        m = labels == c
        if m.any():
            d = ndimage.distance_transform_edt(m).astype(np.float32)
            ao[m] = np.clip(d[m] / (size * rng.uniform(0.035, 0.07)), 0, 1)
    ao = 0.70 + 0.30 * ao

    shade = directional * folds * ao
    shade[~garment] = 1.0
    return np.clip(shade, 0.25, 1.6).astype(np.float32)


def _albedo(rng, labels, size):
    """Base colour per panel, with occasional contrast trim."""
    def colour():
        h = rng.random()
        base = np.array([0.5 + 0.45 * np.cos(2 * np.pi * (h + p)) for p in (0.0, 0.33, 0.67)])
        base = np.clip(base, 0.05, 0.95)
        sat = rng.uniform(0.25, 1.0)
        grey = base.mean()
        base = grey + (base - grey) * sat
        return np.clip(base * rng.uniform(0.35, 0.95), 0.04, 0.96) * 255.0

    main = colour()
    trim = colour() if rng.random() < 0.25 else main
    sleeve_c = colour() if rng.random() < 0.12 else main

    alb = np.zeros((size, size, 3), np.float32)
    alb[labels == CLASS_BODY] = main
    alb[labels == CLASS_COLLAR] = trim
    alb[labels == CLASS_SLEEVE] = sleeve_c

    # woven texture at pixel scale
    weave = rng.normal(0, 1, (size, size)).astype(np.float32)
    weave = ndimage.gaussian_filter(weave, 0.6)
    weave /= max(1e-6, np.abs(weave).max())
    alb *= (1.0 + rng.uniform(0.02, 0.06) * weave)[..., None]
    return alb


def _backdrop(rng, size):
    """Clean studio backdrop: near-neutral, gentle gradient, slight vignette."""
    base = rng.uniform(198, 246)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    ang = rng.uniform(0, 2 * np.pi)
    grad = np.cos(ang) * xx + np.sin(ang) * yy
    grad = (grad - grad.min()) / max(1e-6, np.ptp(grad))
    bg = base + rng.uniform(-14, 14) * grad

    r = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
    bg *= 1.0 - rng.uniform(0.0, 0.16) * np.clip(r / 0.7, 0, 1) ** 2

    tint = np.array([rng.uniform(0.98, 1.02) for _ in range(3)], np.float32)
    return np.clip(bg[..., None] * tint, 0, 255).astype(np.float32)


# --------------------------------------------------------------------------
# one sample
# --------------------------------------------------------------------------

def make_sample(seed, size=384):
    """
    Deterministic: the same seed always produces the same (image, mask) pair.

    Returns (rgb uint8 (size, size, 3), labels uint8 (size, size)).
    """
    rng = np.random.default_rng(seed)

    for _ in range(8):  # retry if the roll produced a degenerate garment
        geo = _garment_geometry(rng, size)
        labels, alpha, neck_cov = _rasterise_labels(geo, size)
        if (labels == CLASS_BODY).sum() > size * size * 0.02:
            break

    bg = _backdrop(rng, size)

    # contact shadow on the backdrop, offset from the garment silhouette
    sil = (alpha > 0.5).astype(np.float32)
    if sil.any():
        sh = ndimage.gaussian_filter(sil, size * rng.uniform(0.02, 0.05))
        sh = np.roll(sh, int(size * rng.uniform(0.005, 0.03)), axis=0)
        bg *= (1.0 - rng.uniform(0.10, 0.30) * np.clip(sh, 0, 1))[..., None]

    alb = _albedo(rng, labels, size)
    shade = _shading(rng, labels, size)

    # the neck interior is the far side of the garment, seen from inside: same
    # fabric, deep in shadow, and with the fold detail flattened out
    interior = neck_cov > 0.5
    if interior.any():
        shade[interior] = rng.uniform(0.22, 0.42)
        shade = ndimage.gaussian_filter(shade, 0.8)

    garment = np.clip(alb * shade[..., None], 0, 255)

    a = alpha[..., None]
    img = garment * a + bg * (1 - a)

    # a touch of sensor noise so edges are not unnaturally clean
    img += rng.normal(0, rng.uniform(0.6, 2.2), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)

    # labels must not survive where the garment is transparent
    labels = labels.copy()
    labels[alpha <= 0.5] = CLASS_BG
    return img, labels


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def contact_sheet(seed, n=24, size=384, cols=6, cell=180):
    """A grid of samples with their masks, for eyeballing fidelity."""
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell * 2), (24, 24, 24))
    palette = np.array([[24, 24, 24], [60, 130, 220], [235, 150, 40], [200, 70, 70]], np.uint8)
    for i in range(n):
        img, lab = make_sample(seed * 100003 + i, size)
        r, c = divmod(i, cols)
        sheet.paste(Image.fromarray(img).resize((cell, cell), Image.BILINEAR),
                    (c * cell, r * cell * 2))
        sheet.paste(Image.fromarray(palette[lab]).resize((cell, cell), Image.NEAREST),
                    (c * cell, r * cell * 2 + cell))
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--contact-sheet", default=None,
                    help="also write a grid of samples + masks here, to eyeball fidelity")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "_masks"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "_images"), exist_ok=True)

    manifest = []
    stats = np.zeros(NUM_CLASSES, np.int64)
    for i in tqdm(range(args.n), desc="synth"):
        img, lab = make_sample(args.seed * 100003 + i, args.size)
        ip = os.path.join(args.out, "_images", f"synth_{i:06d}.png")
        mp = os.path.join(args.out, "_masks", f"synth_{i:06d}.png")
        Image.fromarray(img).save(ip)
        Image.fromarray(lab).save(mp)
        manifest.append({"image": ip, "mask": mp})
        stats += np.bincount(lab.ravel(), minlength=NUM_CLASSES)

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    total = stats.sum()
    print(f"wrote {len(manifest)} pairs -> {args.out}/manifest.json")
    for name, c in zip(["background", "body", "sleeve", "collar"], stats):
        print(f"  {name:<11} {c / total:6.2%} of pixels")

    if args.contact_sheet:
        contact_sheet(args.seed, n=24, size=args.size).save(args.contact_sheet)
        print("contact sheet ->", args.contact_sheet)


if __name__ == "__main__":
    main()
