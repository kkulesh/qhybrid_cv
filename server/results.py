"""Results of trained models and experiment runs for the «Результати» page.

Two sources, both already on disk (no database):
  * the registries of trained models (models/<dataset>/registry.json and datasets/<id>/models/registry.json):
    one row per model with test F1, accuracy, parameters and training time;
  * experiment grids (results/<run>/hybrid.csv and baselines.csv, written by scripts/run_experiments.py):
    every (encoding, qubits, gamma) or baseline, averaged over seeds, with inference time per image.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

RUN_NAMES = {"round1": "Раунд 1: CIFAR-10 і Fashion-MNIST", "voc_round1": "PASCAL VOC 2007"}
SKIP = ("_before_fixes",)  # kept for history; its numbers were produced before bug fixes and would mislead


def model_rows(datasets):
    out = []
    for ds in datasets.values():
        reg = ds.models_dir / "registry.json"
        if not reg.is_file():
            continue
        try:
            r = json.loads(reg.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for m in r.get("models", []):
            out.append({"dataset": ds.id, "dataset_name": ds.name, "id": m["id"], "kind": m["kind"],
                        "encoding": m.get("encoding"), "n": m.get("n"), "gamma": m.get("gamma"),
                        "layers": m.get("layers"), "params": m.get("params"), "seed": m.get("seed"),
                        "test_f1": m.get("test_f1"), "test_acc": m.get("test_acc"), "val_f1": m.get("val_f1"),
                        "train_s": m.get("train_s"), "custom": bool(m.get("custom")),
                        "f1_seeds": m.get("test_f1_all_seeds")})
    return out


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _agg(rows, key_fields, base):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in key_fields)].append(r)
    out = []
    for key, rs in groups.items():
        f1 = [x for x in (_num(r["f1"]) for r in rs) if x is not None]
        col = lambda c: [x for x in (_num(r.get(c)) for r in rs) if x is not None]  # noqa: E731
        avg = lambda c: round(mean(col(c)), 4) if col(c) else None  # noqa: E731
        seeds = sorted({int(_num(r.get("seed")) or 0) for r in rs if _num(r.get("seed")) is not None})
        out.append({**base(dict(zip(key_fields, key))), "seeds": len(rs), "seed_list": seeds,
                    "f1": round(mean(f1), 4) if f1 else None, "f1_std": round(pstdev(f1), 4) if len(f1) > 1 else 0.0,
                    "acc": avg("acc"), "ms_per_img": avg("ms_per_img"), "train_s": avg("train_s"),
                    "params": int(_num(rs[0].get("params")) or 0)})
    return out


def experiment_rows(results_dir):
    runs, rows = [], []
    for d in sorted(Path(results_dir).iterdir()) if Path(results_dir).is_dir() else []:
        if not d.is_dir() or d.name.endswith(SKIP):
            continue
        hy, bl = d / "hybrid.csv", d / "baselines.csv"
        if not hy.is_file() and not bl.is_file():
            continue
        runs.append({"id": d.name, "name": RUN_NAMES.get(d.name, d.name)})
        if hy.is_file():
            rows += _agg(_read(hy), ("dataset", "encoding", "n", "gamma"), lambda k, run=d.name: {
                "run": run, "dataset": k["dataset"], "kind": "hybrid", "model": k["encoding"],
                "n": int(k["n"]), "gamma": float(k["gamma"])})
        if bl.is_file():
            rows += _agg(_read(bl), ("dataset", "model", "n"), lambda k, run=d.name: {
                "run": run, "dataset": k["dataset"], "kind": "classical", "model": k["model"],
                "n": int(k["n"]), "gamma": None})
    return {"runs": runs, "rows": rows}
