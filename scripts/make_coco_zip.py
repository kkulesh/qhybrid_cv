"""Make a dataset ZIP for the web app's «Набір даних» page from COCO object crops.

Every annotated object is cut out by its box (with a small margin) and saved as
<class>/<image>_<k>.jpg, so the ZIP has the same class-folder layout as mnist_pryklad.zip.

Two label formats are accepted:
  * COCO JSON (official val2017 / train2017):
      python scripts/make_coco_zip.py --images data/coco/val2017 \
          --ann data/coco/annotations/instances_val2017.json --out coco_pryklad.zip
  * YOLO txt labels (Ultralytics coco128 / coco8):
      python scripts/make_coco_zip.py --images data/coco128/images/train2017 \
          --yolo-labels data/coco128/labels/train2017 --out coco128_pryklad.zip
"""
import argparse
import io
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image

# the 80 COCO classes in the YOLO (contiguous) order
COCO80 = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
          "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
          "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
          "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
          "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
          "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
          "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
          "teddy bear", "hair drier", "toothbrush"]


def boxes_coco_json(images, ann):
    """{image path: [(class name, x, y, w, h)]} from a COCO instances_*.json (boxes in pixels)."""
    d = json.loads(Path(ann).read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    files = {im["id"]: Path(images) / im["file_name"] for im in d["images"]}
    out = defaultdict(list)
    for a in d["annotations"]:
        if not a.get("iscrowd"):
            out[files[a["image_id"]]].append((cats[a["category_id"]], *a["bbox"]))
    return out


def boxes_yolo(images, labels):
    """The same from YOLO txt labels (class cx cy w h, normalised)."""
    out = defaultdict(list)
    for txt in sorted(Path(labels).glob("*.txt")):
        img = next((p for p in (Path(images) / (txt.stem + e) for e in (".jpg", ".png")) if p.is_file()), None)
        if img is None:
            continue
        with Image.open(img) as im:
            W, H = im.size
        for line in txt.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                c, cx, cy, w, h = line.split()[:5]
                w, h = float(w) * W, float(h) * H
                out[img].append((COCO80[int(c)], float(cx) * W - w / 2, float(cy) * H - h / 2, w, h))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--ann", help="COCO instances_*.json")
    ap.add_argument("--yolo-labels", help="folder of YOLO txt labels")
    ap.add_argument("--classes", nargs="*", help="class names; default: the --top most frequent")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--per-class", type=int, default=150, help="at most this many crops per class")
    ap.add_argument("--min-size", type=int, default=32, help="skip objects smaller than this (pixels, shorter side)")
    ap.add_argument("--max-side", type=int, default=256, help="shrink crops to this longest side")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if bool(a.ann) == bool(a.yolo_labels):
        ap.error("give exactly one of --ann or --yolo-labels")

    boxes = boxes_coco_json(a.images, a.ann) if a.ann else boxes_yolo(a.images, a.yolo_labels)
    objs = defaultdict(list)  # class -> [(image, k, box)]
    for img, bs in boxes.items():
        for k, (c, x, y, w, h) in enumerate(bs):
            if min(w, h) >= a.min_size:
                objs[c].append((img, k, (x, y, w, h)))
    classes = a.classes or sorted(objs, key=lambda c: -len(objs[c]))[:a.top]
    rng = random.Random(a.seed)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_STORED) as z:  # JPEGs do not compress further
        for c in classes:
            items = objs.get(c, [])
            rng.shuffle(items)
            items = items[:a.per_class]
            folder = c.replace(" ", "_")
            for img, k, (x, y, w, h) in items:
                m = 0.1  # keep a little context around the box
                with Image.open(img) as im:
                    im = im.convert("RGB")
                    crop = im.crop((max(0, x - m * w), max(0, y - m * h),
                                    min(im.width, x + w * (1 + m)), min(im.height, y + h * (1 + m))))
                crop.thumbnail((a.max_side, a.max_side))
                buf = io.BytesIO()
                crop.save(buf, "JPEG", quality=90)
                z.writestr(f"coco/{folder}/{img.stem}_{k}.jpg", buf.getvalue())
            print(f"{c:15s} {len(items):5d}")
    print("saved", a.out)


if __name__ == "__main__":
    main()
