"""First experiment round: classical baselines, hybrid VQC grid, and method-A selection metrics.

Outputs (in --out): baselines.csv, hybrid.csv, metrics.csv. Rows are appended as runs finish,
so a partial run is still usable.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one BLAS thread per worker process
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import csv
import itertools
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--features", nargs="+", required=True)
p.add_argument("--out", default="results/round1")
p.add_argument("--qubits", type=int, nargs="+", default=[4, 6, 8])
p.add_argument("--encodings", nargs="+", default=["angle", "zz", "amplitude"])
p.add_argument("--gammas", type=float, nargs="+", default=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0])
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
p.add_argument("--layers", type=int, default=2)
p.add_argument("--readout", choices=["z", "zz"], default="z")
p.add_argument("--reupload", action="store_true")
p.add_argument("--train-per-class", type=int, default=500)
p.add_argument("--val-per-class", type=int, default=100)
p.add_argument("--select-per-class", type=int, default=50)
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--workers", type=int, default=4)
a = p.parse_args()


def load(path):
    """Fixed stratified train/val/selection split of the cached training features."""
    d = np.load(path)
    rng = np.random.default_rng(0)
    tr, va = [], []
    for c in np.unique(d["y_train"]):
        idx = rng.permutation(np.where(d["y_train"] == c)[0])
        tr += list(idx[:a.train_per_class])
        va += list(idx[a.train_per_class:a.train_per_class + a.val_per_class])
    tr, va = rng.permutation(tr), np.array(va)
    sel = np.concatenate([tr[d["y_train"][tr] == c][:a.select_per_class] for c in np.unique(d["y_train"])])
    Z, y = d["Z_train"].astype(np.float32), d["y_train"]
    return dict(Ztr=Z[tr], ytr=y[tr], Zva=Z[va], yva=y[va], Zsel=Z[sel], ysel=y[sel],
                Zte=d["Z_test"].astype(np.float32), yte=d["y_test"], n_classes=len(d["classes"]))


def append(path, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def job(spec):
    import torch
    torch.set_num_threads(1)
    from qhybrid.encodings import make_encoding
    from qhybrid.models import BottleneckMLP, HybridModel, pca_init
    from qhybrid.selection import encoding_metrics
    from qhybrid.train import evaluate, n_params, train_model

    kind, ds_path = spec[0], spec[1]
    D = load(ds_path)
    ds = Path(ds_path).stem
    out = Path(a.out)
    t0 = time.perf_counter()
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(D["Ztr"])
        clf = LogisticRegression(max_iter=3000).fit(sc.transform(D["Ztr"]), D["ytr"])
        t1 = time.perf_counter()
        pred = clf.predict(sc.transform(D["Zte"]))
        ms = 1000 * (time.perf_counter() - t1) / len(pred)
        append(out / "baselines.csv", dict(dataset=ds, model="logreg_1280", n=1280, seed=0,
               params=clf.coef_.size + clf.intercept_.size, acc=accuracy_score(D["yte"], pred),
               f1=f1_score(D["yte"], pred, average="macro"), ms_per_img=ms,
               train_s=t1 - t0))
    elif kind in ("mlp", "linear_bottleneck"):
        _, _, n, seed = spec
        torch.manual_seed(seed)
        pca = pca_init(D["Ztr"], n)
        if kind == "mlp":
            m = BottleneckMLP(D["Ztr"].shape[1], n, D["n_classes"], pca)
        else:
            m = torch.nn.Sequential(torch.nn.Linear(D["Ztr"].shape[1], n), torch.nn.Linear(n, D["n_classes"]))
        train_model(m, D["Ztr"], D["ytr"], epochs=a.epochs, seed=seed)
        r = evaluate(m, D["Zte"], D["yte"])
        append(out / "baselines.csv", dict(dataset=ds, model=kind, n=n, seed=seed, params=n_params(m),
               acc=r["acc"], f1=r["f1"], ms_per_img=r["ms_per_img"], train_s=time.perf_counter() - t0))
    elif kind == "metrics":
        _, _, n, enc_name, gamma = spec
        enc = make_encoding(enc_name, n)
        m = encoding_metrics(enc, D["Zsel"], D["ysel"], gamma)
        append(out / "metrics.csv", dict(dataset=ds, n=n, encoding=enc_name, gamma=gamma, **m,
               time_s=time.perf_counter() - t0))
    elif kind == "hybrid":
        _, _, n, enc_name, gamma, seed = spec
        enc = make_encoding(enc_name, n)
        pca = pca_init(D["Ztr"], enc.k, enc.whiten)
        m = HybridModel(D["Ztr"].shape[1], D["n_classes"], enc, gamma, a.layers, pca, seed, a.readout, a.reupload)
        hist = train_model(m, D["Ztr"], D["ytr"], epochs=a.epochs, seed=seed, val=(D["Zva"], D["yva"]))
        train_s = time.perf_counter() - t0
        r = evaluate(m, D["Zte"], D["yte"])
        append(out / "hybrid.csv", dict(dataset=ds, n=n, encoding=enc_name, gamma=gamma, seed=seed,
               params=n_params(m), vqc_params=m.vqc.weights.numel(), acc=r["acc"], f1=r["f1"],
               ms_per_img=r["ms_per_img"], train_s=train_s,
               val_f1_ep1=hist[0]["val_f1"], val_f1_ep2=hist[1]["val_f1"], val_f1_last=hist[-1]["val_f1"],
               loss_last=hist[-1]["loss"]))
    return spec


if __name__ == "__main__":
    Path(a.out).mkdir(parents=True, exist_ok=True)
    specs = []
    for f in a.features:
        specs.append(("logreg", f))
        for n, s in itertools.product(a.qubits, a.seeds):
            # linear bottleneck of width 2^n - 1 is the matching control for amplitude encoding
            specs += [("mlp", f, n, s), ("linear_bottleneck", f, n, s), ("linear_bottleneck", f, 2 ** n - 1, s)]
        for n, e, g in itertools.product(a.qubits, a.encodings, a.gammas):
            specs.append(("metrics", f, n, e, g))
    hybrid = [("hybrid", f, n, e, g, s) for f, n, e, g, s in
              itertools.product(a.features, a.qubits, a.encodings, a.gammas, a.seeds)]
    # heaviest (8-qubit ZZ) first so the pool does not end on one long straggler
    hybrid.sort(key=lambda s: (-s[2], s[3] != "zz"))
    specs += hybrid
    # resume: skip hybrid runs already present in hybrid.csv (other job kinds are cheap; rerun them
    # only if their CSV is missing)
    out = Path(a.out)
    done = set()
    if (out / "hybrid.csv").exists():
        import pandas as pd
        h = pd.read_csv(out / "hybrid.csv")
        done = {(r.dataset, r.n, r.encoding, float(r.gamma), r.seed) for r in h.itertuples()}
    skip_cheap = (out / "baselines.csv").exists() and (out / "metrics.csv").exists()
    specs = [s for s in specs if not (
        (s[0] == "hybrid" and (Path(s[1]).stem, s[2], s[3], float(s[4]), s[5]) in done)
        or (s[0] != "hybrid" and skip_cheap))]
    print(f"{len(specs)} jobs")
    t0 = time.perf_counter()
    with Pool(a.workers) as pool:
        for i, s in enumerate(pool.imap_unordered(job, specs), 1):
            if i % 10 == 0 or i == len(specs):
                print(f"[{time.perf_counter() - t0:7.0f}s] {i}/{len(specs)} done", flush=True)
