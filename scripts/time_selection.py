"""Wall-clock time of the whole KTA selection procedure (one PCA per sample, all encodings x gammas),
as it runs in practice. Single thread, same samples as run_experiments.py."""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from qhybrid.encodings import make_encoding
from qhybrid.selection import centred_kta, fidelity_kernel, statevectors

p = argparse.ArgumentParser()
p.add_argument("--features", nargs="+", required=True)
p.add_argument("--out", required=True)
p.add_argument("--train-per-class", type=int, nargs="+", required=True)
p.add_argument("--val-per-class", type=int, nargs="+", required=True)
p.add_argument("--qubits", type=int, nargs="+", default=[2, 4, 6, 8])
p.add_argument("--gammas", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2])
p.add_argument("--select-per-class", type=int, default=25)
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
a = p.parse_args()
torch.set_num_threads(1)

rows = []
for f, tpc, vpc in zip(a.features, a.train_per_class, a.val_per_class):
    d = np.load(f)
    rng = np.random.default_rng(0)
    tr = []
    for c in np.unique(d["y_train"]):
        idx = rng.permutation(np.where(d["y_train"] == c)[0])
        tr += list(idx[:tpc])
    tr = rng.permutation(tr)
    for s in a.seeds:
        rs = np.random.default_rng(1000 + s)
        sel = np.concatenate([rs.permutation(tr[d["y_train"][tr] == c])[:a.select_per_class]
                              for c in np.unique(d["y_train"])])
        Z, y = d["Z_train"][sel].astype(np.float64), d["y_train"][sel]
        for n in a.qubits:
            if n == 2 and len(d["classes"]) > 2:
                continue
            t0 = time.perf_counter()
            mu = Z.mean(0)
            evals, evecs = np.linalg.eigh(np.cov(Z - mu, rowvar=False))
            order = np.argsort(evals)[::-1]
            best = None
            for e in ["angle", "zz", "amplitude"]:
                enc = make_encoding(e, n)
                k = enc.k
                ke = min(k, len(Z) - 1)
                V, lam = evecs[:, order[:ke]].T, np.maximum(evals[order[:ke]], 1e-12)
                W = np.zeros((k, Z.shape[1]))
                W[:ke] = V / (np.sqrt(lam)[:, None] if enc.whiten else np.sqrt(lam.sum() / k))
                U = torch.as_tensor((Z - mu) @ W.T)
                for g in a.gammas:
                    X = enc.normalize(U, g).numpy()
                    kta = centred_kta(fidelity_kernel(statevectors(enc, X)), y)
                    if best is None or kta > best[0]:
                        best = (kta, e, g)
            rows.append(dict(dataset=Path(f).stem, n=n, sample_seed=s, m=len(y),
                             time_s=time.perf_counter() - t0, encoding=best[1], gamma=best[2]))
            print(rows[-1], flush=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
