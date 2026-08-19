"""
Turn Fashionpedia annotations into 4-class panel masks (background/body/sleeve/
collar) that we can train on.

Fashionpedia is COCO-format instance segmentation with fine-grained apparel
parts. We do not hardcode category ids (they are easy to get wrong); instead we
read the category names from the annotation file and match by name:

    body   <- any main upper-body garment (shirt, jacket, dress, ...)
    sleeve <- the "sleeve" part
    collar <- the "collar" part

Painting order matters: body first, then sleeve and collar on top, so a part
always wins over the body it sits on.

Output: one 8-bit PNG per image, same basename, pixel value in {0,1,2,3}.

    python -m src.prepare_data \
        --ann  /path/instances_attributes_train2020.json \
        --imgs /path/train \
        --out  data/train \
        --max-images 4000 --seed 42
"""

import argparse
import json
import os
import random

import numpy as np
from PIL import Image
from pycocotools import mask as cocomask
from tqdm import tqdm

# Name tokens we treat as "body" (upper-body garments). Matched case-insensitively
# as substrings of the Fashionpedia category name.
UPPER_GARMENT_TOKENS = [
    "shirt", "blouse", "top", "t-shirt", "sweatshirt", "sweater", "cardigan",
    "jacket", "vest", "coat", "dress", "jumpsuit", "cape",
]
SLEEVE_TOKENS = ["sleeve"]
COLLAR_TOKENS = ["collar"]

CLASS_BG, CLASS_BODY, CLASS_SLEEVE, CLASS_COLLAR = 0, 1, 2, 3


def _match(name, tokens):
    name = name.lower()
    return any(t in name for t in tokens)


def build_category_maps(categories):
    """Return sets of category ids for body / sleeve / collar."""
    body, sleeve, collar = set(), set(), set()
    for c in categories:
        cid, name = c["id"], c["name"]
        if _match(name, SLEEVE_TOKENS):
            sleeve.add(cid)
        elif _match(name, COLLAR_TOKENS):
            collar.add(cid)
        elif _match(name, UPPER_GARMENT_TOKENS):
            body.add(cid)
    return body, sleeve, collar


def ann_to_mask(ann, h, w):
    """Decode one COCO annotation (polygon or RLE) to a boolean mask."""
    seg = ann.get("segmentation")
    if seg is None:
        return None
    if isinstance(seg, list):  # polygons
        rles = cocomask.frPyObjects(seg, h, w)
        rle = cocomask.merge(rles)
    elif isinstance(seg["counts"], list):  # uncompressed RLE
        rle = cocomask.frPyObjects(seg, h, w)
    else:  # compressed RLE
        rle = seg
    return cocomask.decode(rle).astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="Fashionpedia instances json")
    ap.add_argument("--imgs", required=True, help="folder of source images")
    ap.add_argument("--out", required=True, help="output folder for mask PNGs")
    ap.add_argument("--max-images", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--require-sleeve", action="store_true",
                    help="only keep images that contain a sleeve (denser signal)")
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "_masks"), exist_ok=True)

    print("loading annotations...")
    with open(args.ann) as f:
        data = json.load(f)

    body_ids, sleeve_ids, collar_ids = build_category_maps(data["categories"])
    print(f"category ids -> body:{sorted(body_ids)} sleeve:{sorted(sleeve_ids)} "
          f"collar:{sorted(collar_ids)}")

    anns_by_img = {}
    for a in data["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    images = {im["id"]: im for im in data["images"]}
    img_ids = list(images.keys())
    random.shuffle(img_ids)

    kept = 0
    manifest = []
    for img_id in tqdm(img_ids):
        if kept >= args.max_images:
            break
        im = images[img_id]
        h, w = im["height"], im["width"]
        anns = anns_by_img.get(img_id, [])
        if not anns:
            continue

        canvas = np.zeros((h, w), dtype=np.uint8)
        has_sleeve = False
        # body first
        for a in anns:
            if a["category_id"] in body_ids:
                m = ann_to_mask(a, h, w)
                if m is not None:
                    canvas[m] = CLASS_BODY
        # collar and sleeve on top
        for a in anns:
            if a["category_id"] in collar_ids:
                m = ann_to_mask(a, h, w)
                if m is not None:
                    canvas[m] = CLASS_COLLAR
        for a in anns:
            if a["category_id"] in sleeve_ids:
                m = ann_to_mask(a, h, w)
                if m is not None:
                    canvas[m] = CLASS_SLEEVE
                    has_sleeve = True

        if not (canvas > 0).any():
            continue
        if args.require_sleeve and not has_sleeve:
            continue

        src = os.path.join(args.imgs, im["file_name"])
        if not os.path.exists(src):
            continue

        stem = os.path.splitext(os.path.basename(im["file_name"]))[0]
        Image.fromarray(canvas).save(os.path.join(args.out, "_masks", stem + ".png"))
        manifest.append({"image": src, "mask": os.path.join(args.out, "_masks", stem + ".png")})
        kept += 1

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {kept} mask/image pairs -> {args.out}/manifest.json")


if __name__ == "__main__":
    main()
