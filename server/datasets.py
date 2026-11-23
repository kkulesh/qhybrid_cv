"""Datasets of the web app. A dataset is a features file (frozen YOLO11 features, made by
scripts/extract_features.py or by an upload) plus a directory with its trained models.

Built-in: PASCAL VOC (features/voc.npz, models/voc, server/samples) and any other features/<name>.npz.
Uploaded: a ZIP of class folders, unpacked and turned into features by the server; kept in
datasets/<id>/ (features.npz, meta.json, samples/, models/).
"""
import json
import re
import shutil
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TEST_NAMES = {"test", "val", "valid", "validation"}
NAMES = {"voc": "PASCAL VOC 2007", "cifar10": "CIFAR-10", "fashion-mnist": "Fashion-MNIST"}
MAX_IMAGES = 50_000
SAMPLES_PER_CLASS = 2


@dataclass
class Dataset:
    id: str
    name: str
    features: Path
    models_dir: Path
    samples_dir: Path | None
    builtin: bool
    created: str | None = None


def discover(root, data_dir, voc_features, voc_models, voc_samples):
    """All datasets, VOC first."""
    out = {"voc": Dataset("voc", NAMES["voc"], Path(voc_features), Path(voc_models), Path(voc_samples), True)}
    for f in sorted((Path(root) / "features").glob("*.npz")):
        if f.stem not in out and f.resolve() != Path(voc_features).resolve():
            out[f.stem] = Dataset(f.stem, NAMES.get(f.stem, f.stem), f, Path(root) / "models" / f.stem, None, True)
    for meta in sorted(Path(data_dir).glob("*/meta.json")):
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):  # broken leftover of a failed upload: skip it, do not stop the server
            continue
        d = meta.parent
        out[m["id"]] = Dataset(m["id"], m["name"], d / "features.npz", d / "models", d / "samples", False, m.get("created"))
    return out


def info(ds: Dataset):
    """Counts read from the features file (np.load is lazy, so only the label arrays are read)."""
    base = {"id": ds.id, "name": ds.name, "builtin": ds.builtin, "created": ds.created}
    if not ds.features.is_file():
        return {**base, "available": False, "path": str(ds.features)}
    with np.load(ds.features) as d:
        y, yt = d["y_train"], d["y_test"]
        counts = np.bincount(y, minlength=len(d["classes"]))
        return {**base, "available": True, "classes": [str(c) for c in d["classes"]], "dim": int(d["Z_train"].shape[1]),
                "n_train": int(len(y)), "n_test": int(len(yt)), "min_per_class": int(counts.min()),
                "samples": bool(ds.samples_dir and any(ds.samples_dir.glob("*")))}


UK_LAT = dict(zip("абвгґдеєжзиіїйклмнопрстуфхцчшщьюя",
                  ["a", "b", "v", "h", "g", "d", "e", "ie", "zh", "z", "y", "i", "i", "i", "k", "l", "m", "n", "o", "p",
                   "r", "s", "t", "u", "f", "kh", "ts", "ch", "sh", "shch", "", "iu", "ia"]))


def slug(name):
    lat = "".join(UK_LAT.get(ch, ch) for ch in name.lower())
    s = re.sub(r"[^a-z0-9]+", "-", lat).strip("-")[:24] or "dataset"
    return f"{s}-{datetime.now():%m%d%H%M%S}"


def _safe_unzip(zip_path, dest, job):
    """Extract only image files, refusing paths that escape dest."""
    dest = Path(dest).resolve()
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if Path(n).suffix.lower() in IMG_EXT and not n.startswith("__MACOSX")
                 and not Path(n).name.startswith(".")]
        if not names:
            raise ValueError("В архіві немає зображень (JPEG, PNG, BMP, WebP).")
        if len(names) > MAX_IMAGES:
            raise ValueError(f"Забагато зображень: {len(names)}, максимум {MAX_IMAGES}.")
        for i, n in enumerate(names):
            target = (dest / n).resolve()
            if dest not in target.parents:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(n) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            if i % 200 == 0:
                job.check_cancel()
                job.emit("progress", stage="unzip", done=i + 1, total=len(names))
    return len(names)


def _images(d):
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT)


def _class_dirs(d):
    return {p.name: _images(p) for p in sorted(d.iterdir()) if p.is_dir() and _images(p)}


def find_layout(raw, test_share, seed=0):
    """Accepts <root>/train/<class>/*, <root>/test/<class>/* (test may be named val) or <root>/<class>/*,
    where <root> may be nested in wrapper folders. Without a test folder each class is split
    1 - test_share / test_share. Returns (classes, train [(path, label)], test [(path, label)], note)."""
    root = Path(raw)
    while True:
        subs = [p for p in root.iterdir() if p.is_dir()]
        if len(subs) == 1 and not _images(root) and subs[0].name.lower() not in {"train"}:
            root = subs[0]
        else:
            break
    dirs = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
    test_dir = next((dirs[n] for n in TEST_NAMES if n in dirs), None)
    if "train" in dirs and test_dir is not None:
        tr, te = _class_dirs(dirs["train"]), _class_dirs(test_dir)
        classes = sorted(set(tr) & set(te))
        if len(classes) < 2:
            raise ValueError("У папках train і test мають бути спільні класи (щонайменше 2).")
        train = [(p, i) for i, c in enumerate(classes) for p in tr[c]]
        test = [(p, i) for i, c in enumerate(classes) for p in te[c]]
        return classes, train, test, "розбиття з архіву (train/test)"
    cls = _class_dirs(dirs["train"]) if "train" in dirs else _class_dirs(root)
    cls = {c: f for c, f in cls.items() if len(f) >= 2}
    if len(cls) < 2:
        raise ValueError("Потрібно щонайменше 2 папки класів, у кожній щонайменше 2 зображення.")
    rng = np.random.default_rng(seed)
    classes, train, test = sorted(cls), [], []
    for i, c in enumerate(classes):
        f = [cls[c][j] for j in rng.permutation(len(cls[c]))]
        k = max(1, round(len(f) * test_share))
        test += [(p, i) for p in f[:k]]
        train += [(p, i) for p in f[k:]]
    return classes, train, test, f"випадкове розбиття {round((1 - test_share) * 100)}/{round(test_share * 100)} у кожному класі"


def run_import(job, zip_path, data_dir, extractor, lock, log):
    """Unpack the upload, extract YOLO11 features of every image, keep a few test images as samples."""
    p = job.params
    d = Path(data_dir) / p["dataset"]
    raw = d / "raw"
    try:
        raw.mkdir(parents=True, exist_ok=True)
        n = _safe_unzip(zip_path, raw, job)
        log.info("[набір %s] розпаковано %d зображень", p["dataset"], n)
        classes, train, test, note = find_layout(raw, p["test_share"])
        job.emit("layout", classes=classes, n_train=len(train), n_test=len(test), note=note)
        log.info("[набір %s] класів %d, навчання %d, тест %d (%s)", p["dataset"], len(classes), len(train), len(test), note)
        total, done, skipped, t0 = len(train) + len(test), 0, 0, time.perf_counter()
        out = {}
        for split, items in (("train", train), ("test", test)):
            Z, y = [], []
            for i in range(0, len(items), 32):
                job.check_cancel()
                xs, ys = [], []
                for path, label in items[i:i + 32]:
                    try:
                        with Image.open(path) as img:
                            xs.append(extractor.to_tensor(img)); ys.append(label)
                    except Exception:
                        skipped += 1
                if xs:
                    with lock:  # the extractor is shared with classification requests
                        Z.append(extractor.extract_tensor(torch.stack(xs)).numpy())
                    y += ys
                done += len(items[i:i + 32])
                job.emit("progress", stage="features", done=done, total=total, elapsed_s=round(time.perf_counter() - t0, 1))
            if not Z:
                raise ValueError(f"Не вдалося прочитати жодного зображення в частині {split}.")
            out[f"Z_{split}"], out[f"y_{split}"] = np.concatenate(Z).astype(np.float32), np.array(y)
        missing = [classes[c] for c in range(len(classes)) if not (out["y_train"] == c).any()]
        if missing:
            raise ValueError(f"Немає навчальних зображень для класів: {', '.join(missing)}.")
        np.savez_compressed(d / "features.npz", classes=np.array(classes), **out)
        samples = d / "samples"
        samples.mkdir(exist_ok=True)
        index = []
        for c, cname in enumerate(classes):
            for k, (path, _) in enumerate([t for t in test if t[1] == c][:SAMPLES_PER_CLASS]):
                fname = f"s{c:03d}_{k}{path.suffix.lower()}"
                shutil.copy(path, samples / fname)
                index.append({"name": fname, "label": cname})
        (samples / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
        meta = {"id": p["dataset"], "name": p["name"], "created": datetime.now().isoformat(timespec="seconds"),
                "split": note, "skipped": skipped}
        tmp = d / "meta.json.tmp"  # written whole, then renamed: a crash never leaves a half-written meta.json
        tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(d / "meta.json")
        log.info("[набір %s] ознаки YOLO11 обчислено за %.1f с, пропущено %d файлів", p["dataset"],
                 time.perf_counter() - t0, skipped)
        return {"dataset": p["dataset"], "classes": classes, "n_train": int(len(out["y_train"])),
                "n_test": int(len(out["y_test"])), "skipped": skipped, "split": note,
                "time_s": round(time.perf_counter() - t0, 1)}
    except BaseException:
        if not (d / "meta.json").exists():
            shutil.rmtree(d, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(raw, ignore_errors=True)
        Path(zip_path).unlink(missing_ok=True)
