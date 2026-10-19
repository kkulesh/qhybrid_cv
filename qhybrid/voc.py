"""PASCAL VOC -> object classification dataset.

Every annotated object is cropped from its image by the ground-truth bounding box (with a small
context margin) and saved in ImageFolder layout: <out>/<split>/<class>/<image>_<k>.jpg.
VOC2007 `trainval` becomes the training split and VOC2007 `test` the test split.
"""
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image

VOC_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
               "train", "tvmonitor"]


def parse_annotation(path):
    root = ET.parse(path).getroot()
    objs = []
    for o in root.iter("object"):
        bb = o.find("bndbox")
        objs.append(dict(
            name=o.find("name").text.strip(),
            difficult=int(o.findtext("difficult", "0")),
            truncated=int(o.findtext("truncated", "0")),
            box=tuple(int(float(bb.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax"))))
    return objs


def make_crops(voc_root, out, image_set, split, min_size=32, margin=0.1, skip_difficult=True,
               skip_truncated=False):
    """Crop all objects of one VOC image set. Returns a Counter of crops per class."""
    voc_root, out = Path(voc_root), Path(out)
    ids = (voc_root / "ImageSets" / "Main" / f"{image_set}.txt").read_text().split()
    counts = Counter()
    for img_id in ids:
        objs = parse_annotation(voc_root / "Annotations" / f"{img_id}.xml")
        img = None
        for k, o in enumerate(objs):
            if (skip_difficult and o["difficult"]) or (skip_truncated and o["truncated"]):
                continue
            x0, y0, x1, y1 = o["box"]
            w, h = x1 - x0, y1 - y0
            if min(w, h) < min_size:
                continue
            if img is None:
                img = Image.open(voc_root / "JPEGImages" / f"{img_id}.jpg").convert("RGB")
            dx, dy = int(margin * w), int(margin * h)
            crop = img.crop((max(0, x0 - dx), max(0, y0 - dy), min(img.width, x1 + dx), min(img.height, y1 + dy)))
            d = out / split / o["name"]
            d.mkdir(parents=True, exist_ok=True)
            crop.save(d / f"{img_id}_{k}.jpg", quality=95)
            counts[o["name"]] += 1
    return counts
