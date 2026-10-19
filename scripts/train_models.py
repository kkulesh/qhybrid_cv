"""Train the models served by the web app and save them with a registry.

For every (encoding, qubits) the scale gamma with the best validation F1 in an experiment round is
used (results/<round>/hybrid.csv). A classical linear classifier on all 1280 YOLO11 features is
trained as the reference. Angle and ZZ circuits vary a lot between initialisations, so each model
is trained with several seeds on 90 % of the training features and the seed with the best F1 on the
held-out 10 % is kept; the test split is used only for the reported score.
Output: <out>/<id>.pt and <out>/registry.json.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--features", default="features/voc.npz")
p.add_argument("--results", default="results/voc_round1/hybrid.csv")
p.add_argument("--out", default="models/voc")
p.add_argument("--epochs", type=int, default=15)
p.add_argument("--layers", type=int, default=2)
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
p.add_argument("--workers", type=int, default=4)
a = p.parse_args()


def train_one(spec):
    import torch
    torch.set_num_threads(1)
    from qhybrid.encodings import make_encoding
    from qhybrid.models import HybridModel, pca_init
    from qhybrid.train import evaluate, n_params, train_model

    d = np.load(a.features)
    Z, y, Zte, yte = d["Z_train"].astype(np.float32), d["y_train"], d["Z_test"].astype(np.float32), d["y_test"]
    n_classes = len(d["classes"])
    perm = np.random.default_rng(123).permutation(len(Z))
    n_val = len(Z) // 10
    Zva, yva, Ztr, ytr = Z[perm[:n_val]], y[perm[:n_val]], Z[perm[n_val:]], y[perm[n_val:]]
    seed = spec["seed"]
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    if spec["kind"] == "classical":
        model = torch.nn.Linear(Ztr.shape[1], n_classes)
    else:
        enc = make_encoding(spec["encoding"], spec["n"])
        model = HybridModel(Ztr.shape[1], n_classes, enc, spec["gamma"], a.layers, pca_init(Ztr, enc.k, enc.whiten), seed)
    train_model(model, Ztr, ytr, epochs=a.epochs, seed=seed)
    v = evaluate(model, Zva, yva)
    r = evaluate(model, Zte, yte)
    torch.save(model.state_dict(), Path(a.out) / f"{spec['id']}.seed{seed}.pt")
    return {**spec, "layers": a.layers, "params": n_params(model), "val_f1": round(v["f1"], 4),
            "test_f1": round(r["f1"], 4), "test_acc": round(r["acc"], 4), "train_s": round(time.perf_counter() - t0, 1)}


if __name__ == "__main__":
    Path(a.out).mkdir(parents=True, exist_ok=True)
    h = pd.read_csv(a.results)
    g = h.groupby(["n", "encoding", "gamma"])["val_f1_last"].mean().reset_index()
    best = g.loc[g.groupby(["n", "encoding"])["val_f1_last"].idxmax()]
    specs = [{"id": "classical", "kind": "classical", "encoding": None, "n": None, "gamma": None}]
    specs += [{"id": f"{r.encoding}-{r.n}q", "kind": "hybrid", "encoding": r.encoding, "n": int(r.n), "gamma": float(r.gamma)}
              for r in best.itertuples()]
    jobs = [{**s, "seed": seed} for s in specs for seed in (a.seeds if s["kind"] == "hybrid" else [0])]
    jobs.sort(key=lambda s: (-(s["n"] or 0), s["encoding"] != "zz"))  # slow 8-qubit ZZ first
    with Pool(a.workers) as pool:
        runs = []
        for m in pool.imap_unordered(train_one, jobs):
            print(m, flush=True)
            runs.append(m)
    out = Path(a.out)
    models = []
    for s in specs:
        cand = [r for r in runs if r["id"] == s["id"]]
        best = max(cand, key=lambda r: r["val_f1"])
        (out / f"{s['id']}.seed{best['seed']}.pt").replace(out / f"{s['id']}.pt")
        for r in cand:
            (out / f"{s['id']}.seed{r['seed']}.pt").unlink(missing_ok=True)
        best["test_f1_all_seeds"] = sorted(r["test_f1"] for r in cand)
        models.append(best)
    classes = [str(c) for c in np.load(a.features)["classes"]]
    order = {"classical": 0, "amplitude": 1, "angle": 2, "zz": 3}
    models.sort(key=lambda m: (order[m["encoding"] or "classical"], m["n"] or 0))
    (Path(a.out) / "registry.json").write_text(json.dumps(
        {"dataset": Path(a.features).stem, "classes": classes, "feature_dim": 1280, "models": models}, indent=1, ensure_ascii=False))
