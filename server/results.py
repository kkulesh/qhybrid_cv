"""Results of trained models and experiment runs for the «Результати» page.

With PostgreSQL (server/db.py) everything is read from the database: models, experiment grids, encoding-selection
runs and the classification log. Without it the page reads the files as before:
  * the registries of trained models (models/<dataset>/registry.json and datasets/<id>/models/registry.json);
  * experiment grids (results/<run>/hybrid.csv and baselines.csv, written by scripts/run_experiments.py).
Experiment rows are averaged over seeds per (encoding, qubits, gamma) or baseline in both cases.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

from server import db

RUN_NAMES = {"round1": "Раунд 1: CIFAR-10 і Fashion-MNIST", "voc_round1": "PASCAL VOC 2007"}
SKIP = ("_before_fixes",)  # kept for history; its numbers were produced before bug fixes and would mislead
HYBRID_COLS = {"dataset", "n", "encoding", "gamma", "seed", "params", "acc", "f1", "ms_per_img", "train_s"}
BASELINE_COLS = {"dataset", "model", "n", "seed", "params", "acc", "f1", "ms_per_img", "train_s"}


# ---------- trained models ----------

def _model_row(ds_id, ds_name, m):
    return {"dataset": ds_id, "dataset_name": ds_name, "id": m["id"], "kind": m["kind"],
            "encoding": m.get("encoding"), "n": m.get("n"), "gamma": m.get("gamma"),
            "layers": m.get("layers"), "params": m.get("params"), "seed": m.get("seed"),
            "test_f1": m.get("test_f1"), "test_acc": m.get("test_acc"), "val_f1": m.get("val_f1"),
            "train_s": m.get("train_s"), "custom": bool(m.get("custom")),
            "f1_seeds": m.get("test_f1_all_seeds")}


def read_registry(ds):
    reg = ds.models_dir / "registry.json"
    if not reg.is_file():
        return None
    try:
        return json.loads(reg.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def model_rows(datasets):
    if db.enabled():
        rows = db.query("SELECT m.*, d.name AS dataset_name FROM models m JOIN datasets d ON d.id = m.dataset_id "
                        "ORDER BY m.dataset_id, m.custom, m.created, m.id")
        return [_model_row(r["dataset_id"], r["dataset_name"], db.model_meta(r)) for r in rows]
    out = []
    for ds in datasets.values():
        r = read_registry(ds)
        out += [_model_row(ds.id, ds.name, m) for m in (r or {}).get("models", [])]
    return out


# ---------- experiment grids ----------

def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _int(v):
    x = _num(v)
    return None if x is None else int(x)


def read_run_csv(d):
    """Rows of results/<run>/hybrid.csv and baselines.csv in one typed format (also used by the import)."""
    rows = []
    for name, kind in (("hybrid.csv", "hybrid"), ("baselines.csv", "classical")):
        f = Path(d) / name
        if not f.is_file():
            continue
        with open(f, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                cols = HYBRID_COLS if kind == "hybrid" else BASELINE_COLS
                rows.append({"dataset": r["dataset"], "kind": kind,
                             "model": r["encoding"] if kind == "hybrid" else r["model"],
                             "n": _int(r.get("n")), "gamma": _num(r.get("gamma")) if kind == "hybrid" else None,
                             "seed": _int(r.get("seed")), "params": _int(r.get("params")), "acc": _num(r.get("acc")),
                             "f1": _num(r.get("f1")), "ms_per_img": _num(r.get("ms_per_img")),
                             "train_s": _num(r.get("train_s")),
                             "extra": {k: (_num(v) if _num(v) is not None else v) for k, v in r.items() if k not in cols}})
    return rows


def run_dirs(results_dir):
    """Experiment folders under results/ that hold hybrid.csv or baselines.csv."""
    p = Path(results_dir)
    return [d for d in sorted(p.iterdir()) if d.is_dir() and not d.name.endswith(SKIP)
            and ((d / "hybrid.csv").is_file() or (d / "baselines.csv").is_file())] if p.is_dir() else []


def aggregate(raw):
    """Mean over seeds per (run, dataset, kind, model, n, gamma)."""
    groups = defaultdict(list)
    for r in raw:
        groups[(r["run"], r["dataset"], r["kind"], r["model"], r["n"], r["gamma"])].append(r)
    out = []
    for (run, dataset, kind, model, n, gamma), rs in groups.items():
        col = lambda c: [r[c] for r in rs if r.get(c) is not None]  # noqa: E731
        avg = lambda c: round(mean(col(c)), 4) if col(c) else None  # noqa: E731
        f1 = col("f1")
        out.append({"run": run, "dataset": dataset, "kind": kind, "model": model, "n": n, "gamma": gamma,
                    "seeds": len(rs), "seed_list": sorted(set(col("seed"))),
                    "f1": avg("f1"), "f1_std": round(pstdev(f1), 4) if len(f1) > 1 else 0.0,
                    "acc": avg("acc"), "ms_per_img": avg("ms_per_img"), "train_s": avg("train_s"),
                    "params": int(rs[0].get("params") or 0)})
    return out


def experiment_rows(results_dir):
    if db.enabled():
        runs = db.query("SELECT id, name FROM experiment_runs ORDER BY id")
        raw = db.query("SELECT run_id AS run, dataset, kind, model, n, gamma, seed, params, acc, f1, ms_per_img, train_s "
                       "FROM experiments")
        return {"runs": runs, "rows": aggregate(raw)}
    runs, raw = [], []
    for d in run_dirs(results_dir):
        runs.append({"id": d.name, "name": RUN_NAMES.get(d.name, d.name)})
        raw += [{**r, "run": d.name} for r in read_run_csv(d)]
    return {"runs": runs, "rows": aggregate(raw)}


# ---------- database only: encoding selection runs and the classification log ----------

def selection_runs(limit=50):
    if not db.enabled():
        return None
    rows = db.query("SELECT s.id, s.created, s.dataset_id, d.name AS dataset_name, s.n_qubits, s.per_class, "
                    "s.sample_size, s.seed, s.best_encoding, s.best_gamma, s.best_kta, s.time_s, "
                    "jsonb_array_length(s.rows) AS configs "
                    "FROM selection_runs s JOIN datasets d ON d.id = s.dataset_id ORDER BY s.created DESC LIMIT %s",
                    (limit,))
    return [{**r, "created": r["created"].isoformat(timespec="seconds")} for r in rows]


def classifications(limit=100):
    if not db.enabled():
        return None
    rows = db.query("SELECT c.id, c.created, c.dataset_id, c.model_id, m.kind, m.encoding, m.n_qubits AS n, m.seed, "
                    "c.image_name, c.device, c.shots, c.predicted, c.confidence, c.timings_ms, c.compare "
                    "FROM classifications c LEFT JOIN models m ON m.dataset_id = c.dataset_id AND m.id = c.model_id "
                    "ORDER BY c.created DESC, c.id DESC LIMIT %s", (limit,))
    return [{**r, "created": r["created"].isoformat(timespec="seconds"),
             "total_ms": (r["timings_ms"] or {}).get("total")} for r in rows]
