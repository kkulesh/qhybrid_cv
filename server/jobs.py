"""Background jobs for the web app: encoding selection (KTA on a small sample) and model training.

One job runs at a time (both are CPU-bound). A job runs in a daemon thread and appends events to
its list; the API streams them to the browser as Server-Sent Events (server/app.py).
"""
import json
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from qhybrid.encodings import make_encoding
from qhybrid.models import HybridModel, pca_init
from qhybrid.selection import centred_kta, fidelity_kernel, hs_distance, offdiag_var, statevectors
from qhybrid.train import evaluate, n_params, train_model

FINAL = ("done", "error", "cancelled")


class Cancelled(Exception):
    pass


class Job:
    def __init__(self, kind, params):
        self.id = uuid.uuid4().hex[:12]
        self.kind, self.params = kind, params
        self.status = "running"
        self.events, self.result, self.error = [], None, None
        self.cancel_requested = False
        self.started = time.time()

    def emit(self, type_, **data):
        self.events.append({"type": type_, "t": round(time.time() - self.started, 2), **data})

    def check_cancel(self):
        if self.cancel_requested:
            raise Cancelled()

    def snapshot(self):
        return {"id": self.id, "kind": self.kind, "params": self.params, "status": self.status,
                "events": self.events, "result": self.result, "error": self.error}


JOBS: dict[str, Job] = {}
_lock = threading.Lock()


def running_job():
    return next((j for j in JOBS.values() if j.status == "running"), None)


def start(kind, params, target, log):
    """Start target(job) in a thread; returns the job, or None if another job is still running."""
    with _lock:
        if running_job():
            return None
        job = Job(kind, params)
        JOBS[job.id] = job

    def run():
        try:
            job.result = target(job)
            job.status = "done"
            job.emit("done", result=job.result)
        except Cancelled:
            job.status = "cancelled"
            job.emit("cancelled")
            log.info("[%s %s] скасовано", kind, job.id)
        except Exception as e:  # report any failure to the page instead of losing it in a thread
            log.exception("[%s %s] помилка", kind, job.id)
            job.status, job.error = "error", str(e)
            job.emit("error", message=str(e))

    threading.Thread(target=run, daemon=True, name=f"{kind}-{job.id}").start()
    return job


def stratified(y, per_class, rng):
    return np.concatenate([rng.permutation(np.where(y == c)[0])[:per_class] for c in np.unique(y)])


def load_features(path):
    d = np.load(path)
    return (d["Z_train"].astype(np.float32), d["y_train"], d["Z_test"].astype(np.float32), d["y_test"],
            [str(c) for c in d["classes"]])


def run_selection(job, features_path, log):
    """Centred KTA (plus kernel concentration and HS distance) of every (encoding, gamma) on a small
    stratified sample of the training features. No VQC is trained."""
    p = job.params
    Z, y, *_ = load_features(features_path)
    idx = stratified(y, p["per_class"], np.random.default_rng(p["seed"]))
    Zs, ys = Z[idx], y[idx]
    total = len(p["encodings"]) * len(p["gammas"])
    log.info("[вибір %s] %d кубітів, вибірка %d (%d на клас), %d конфігурацій",
             job.id, p["n"], len(ys), p["per_class"], total)
    rows, t_all = [], time.perf_counter()
    for name in p["encodings"]:
        enc = make_encoding(name, p["n"])
        pca = pca_init(Zs, enc.k, enc.whiten)  # the same PCA-initialised adapter the training starts from
        for gamma in p["gammas"]:
            job.check_cancel()
            t0 = time.perf_counter()
            X = enc.normalize(torch.as_tensor(Zs @ pca[0].T + pca[1], dtype=torch.float64), gamma).numpy()
            K = fidelity_kernel(statevectors(enc, X))
            row = {"encoding": name, "gamma": gamma, "kta": round(centred_kta(K, ys), 5),
                   "var": float(f"{offdiag_var(K):.4g}"), "hs": float(f"{hs_distance(K, ys):.4g}"),
                   "time_s": round(time.perf_counter() - t0, 3)}
            rows.append(row)
            job.emit("row", row=row, done=len(rows), total=total)
            log.info("[вибір %s] %s γ=%g: KTA=%.4f (%.2f с)", job.id, name, gamma, row["kta"], row["time_s"])
    best_per_enc = [max((r for r in rows if r["encoding"] == e), key=lambda r: r["kta"]) for e in p["encodings"]]
    best = max(best_per_enc, key=lambda r: r["kta"])
    log.info("[вибір %s] рекомендовано: %s, γ=%g (KTA=%.4f)", job.id, best["encoding"], best["gamma"], best["kta"])
    return {"rows": rows, "best": best, "best_per_encoding": sorted(best_per_enc, key=lambda r: -r["kta"]),
            "sample_size": int(len(ys)), "time_s": round(time.perf_counter() - t_all, 2)}


def run_training(job, features_path, models_dir, register, log):
    """Train one hybrid model, stream loss / validation F1 per epoch, evaluate on the test split,
    save it to models_dir and register it so the classification page can use it at once."""
    p = job.params
    Z, y, Zte, yte, classes = load_features(features_path)
    rng = np.random.default_rng(p["seed"])
    idx = stratified(y, p["per_class"], rng)
    idx = rng.permutation(idx)
    n_val = max(len(idx) // 10, len(classes))
    va, tr = idx[:n_val], idx[n_val:]
    torch.manual_seed(p["seed"])
    hybrid = p.get("kind", "hybrid") == "hybrid"
    if hybrid:
        enc = make_encoding(p["encoding"], p["n"])
        model = HybridModel(Z.shape[1], len(classes), enc, p["gamma"], p["layers"],
                            pca_init(Z[tr], enc.k, enc.whiten), p["seed"])
    else:
        model = torch.nn.Linear(Z.shape[1], len(classes))
    job.emit("start", n_train=int(len(tr)), n_val=int(len(va)), n_test=int(len(yte)), params=n_params(model))
    if hybrid:
        log.info("[тренування %s] набір %s: %s, %d кубітів, γ=%g, %d шари, %d епох, %d прикладів навчання",
                 job.id, p["dataset"], p["encoding"], p["n"], p["gamma"], p["layers"], p["epochs"], len(tr))
    else:
        log.info("[тренування %s] набір %s: класична лінійна модель, %d епох, %d прикладів навчання",
                 job.id, p["dataset"], p["epochs"], len(tr))
    t0 = time.perf_counter()

    def on_epoch(e, rec):
        job.emit("epoch", epoch=e + 1, epochs=p["epochs"], loss=round(rec["loss"], 5),
                 val_f1=round(rec["val_f1"], 4), elapsed_s=round(time.perf_counter() - t0, 1))
        log.info("[тренування %s] епоха %d/%d: loss=%.4f, F1 (валідація)=%.4f",
                 job.id, e + 1, p["epochs"], rec["loss"], rec["val_f1"])
        job.check_cancel()

    train_model(model, Z[tr], y[tr], epochs=p["epochs"], lr=p["lr"], batch_size=p["batch_size"],
                seed=p["seed"], val=(Z[va], y[va]), on_epoch=on_epoch)
    train_s = time.perf_counter() - t0
    job.emit("evaluating")
    v, r = evaluate(model, Z[va], y[va]), evaluate(model, Zte, yte)
    stamp = f"{datetime.now():%m%d-%H%M%S}"
    if hybrid:
        gamma_txt = f"{p['gamma']:g}".replace(".", "_")
        mid = f"custom-{p['encoding']}-{p['n']}q-g{gamma_txt}-{stamp}"
        arch = {"kind": "hybrid", "encoding": p["encoding"], "n": p["n"], "gamma": p["gamma"], "layers": p["layers"]}
    else:
        mid = f"custom-classical-{stamp}"
        arch = {"kind": "classical", "encoding": None, "n": None, "gamma": None, "layers": 0}
    meta = {"id": mid, **arch, "seed": p["seed"], "params": n_params(model), "val_f1": round(v["f1"], 4),
            "test_f1": round(r["f1"], 4), "test_acc": round(r["acc"], 4), "train_s": round(train_s, 1),
            "epochs": p["epochs"], "lr": p["lr"], "batch_size": p["batch_size"], "per_class": p["per_class"],
            "custom": True, "created": datetime.now().isoformat(timespec="seconds")}
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(models_dir) / f"{mid}.pt")
    reg_path = Path(models_dir) / "registry.json"
    reg = (json.loads(reg_path.read_text(encoding="utf-8")) if reg_path.is_file() else
           {"dataset": p["dataset"], "classes": classes, "feature_dim": int(Z.shape[1]), "models": []})
    reg["models"].append(meta)
    reg_path.write_text(json.dumps(reg, indent=1, ensure_ascii=False), encoding="utf-8")
    register(meta, model.eval())
    log.info("[тренування %s] F1 (тест)=%.4f, точність=%.4f, збережено як %s", job.id, r["f1"], r["acc"], mid)
    return {"model": meta}
