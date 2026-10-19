"""Extract and cache frozen YOLO11-cls features for an ImageFolder dataset (train/ and test/)."""
import argparse
import time
from pathlib import Path

import numpy as np

from qhybrid.features import YoloFeatureExtractor, list_images

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--out", required=True)
p.add_argument("--weights", default="yolo11n-cls.pt")
p.add_argument("--train-per-class", type=int, default=500)
p.add_argument("--test-per-class", type=int, default=200)
p.add_argument("--imgsz", type=int, default=224)
a = p.parse_args()

ext = YoloFeatureExtractor(a.weights, a.imgsz)
out = {}
for split, per_class in [("train", a.train_per_class), ("test", a.test_per_class)]:
    paths, y, classes = list_images(Path(a.data) / split, per_class, seed=0)
    t0 = time.perf_counter()
    out[f"Z_{split}"], out[f"y_{split}"] = ext.extract(paths), y
    print(f"{split}: {len(paths)} images, {time.perf_counter() - t0:.1f}s")
out["classes"] = np.array(classes)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(a.out, **out)
print("saved", a.out, out["Z_train"].shape)
