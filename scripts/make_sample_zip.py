"""Make a small dataset ZIP for the web app's «Набір даних» page from a dataset kept as class folders
(<root>/<class>/*.jpg, e.g. Caltech-101, Caltech-256, Food-101 images/), with train/ and test/ folders.

    python scripts/make_sample_zip.py --root data/caltech101 --classes airplanes Motorbikes ... \
        --train 60 --test 20 --out caltech101_pryklad.zip
    python scripts/make_sample_zip.py --root data/food-101/images --top 10 --train 100 --test 30 \
        --out food101_pryklad.zip
"""
import argparse
import io
import random
import zipfile
from pathlib import Path

from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="folder with one sub-folder per class")
    ap.add_argument("--classes", nargs="*", help="class folder names; default: the --top largest")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--train", type=int, default=60, help="images per class in train/")
    ap.add_argument("--test", type=int, default=20, help="images per class in test/")
    ap.add_argument("--max-side", type=int, default=256, help="shrink images to this longest side")
    ap.add_argument("--name", help="top folder name inside the ZIP (default: --out without extension)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    folders = {p.name: sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT)
               for p in sorted(Path(a.root).iterdir()) if p.is_dir()}
    classes = a.classes or sorted(folders, key=lambda c: -len(folders[c]))[:a.top]
    top = a.name or Path(a.out).stem
    rng = random.Random(a.seed)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_STORED) as z:  # JPEGs do not compress further
        for c in classes:
            files = list(folders[c])
            rng.shuffle(files)
            if len(files) < a.train + a.test:
                print(f"warning: {c} has only {len(files)} images")
            k = min(a.test, len(files) // 4)
            for split, part in (("test", files[:k]), ("train", files[k:k + a.train])):
                for f in part:
                    with Image.open(f) as im:
                        im = im.convert("RGB")
                        im.thumbnail((a.max_side, a.max_side))
                        buf = io.BytesIO()
                        im.save(buf, "JPEG", quality=90)
                    z.writestr(f"{top}/{split}/{c}/{f.stem}.jpg", buf.getvalue())
            print(f"{c:20s} train {min(a.train, len(files) - k):4d}  test {k:3d}")
    print("saved", a.out)


if __name__ == "__main__":
    main()
