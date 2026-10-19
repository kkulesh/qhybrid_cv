"""Build the VOC object-classification dataset (ImageFolder layout) from a VOCdevkit directory."""
import argparse
import json
from pathlib import Path

from qhybrid.voc import VOC_CLASSES, make_crops

p = argparse.ArgumentParser()
p.add_argument("--voc", required=True, help="path to VOCdevkit/VOC2007")
p.add_argument("--out", required=True)
p.add_argument("--min-size", type=int, default=32)
a = p.parse_args()

stats = {}
for image_set, split in [("trainval", "train"), ("test", "test")]:
    c = make_crops(a.voc, a.out, image_set, split, a.min_size)
    stats[split] = {k: c[k] for k in VOC_CLASSES}
    print(split, sum(c.values()), "crops")
Path(a.out, "stats.json").write_text(json.dumps(stats, indent=1))
print(f"{'class':<12}{'train':>7}{'test':>7}")
for k in VOC_CLASSES:
    print(f"{k:<12}{stats['train'][k]:>7}{stats['test'][k]:>7}")
