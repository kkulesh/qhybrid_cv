"""Summarise a results directory produced by run_experiments.py into tables and figures."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

p = argparse.ArgumentParser()
p.add_argument("--results", default="results/round1")
a = p.parse_args()
R = Path(a.results)
FIG = R / "figures"
FIG.mkdir(exist_ok=True)

base = pd.read_csv(R / "baselines.csv")
hyb = pd.read_csv(R / "hybrid.csv")
met = pd.read_csv(R / "metrics.csv")
ENC = ["angle", "zz", "amplitude"]
COL = {"angle": "#2a7ab9", "zz": "#d9822b", "amplitude": "#3f9e5a"}
lines = []
out = lines.append

# ---------- 1. Baselines and hybrids at the best validation gamma ----------
agg = hyb.groupby(["dataset", "n", "encoding", "gamma"]).agg(
    f1=("f1", "mean"), f1_std=("f1", "std"), acc=("acc", "mean"), val=("val_f1_last", "mean"),
    ep1=("val_f1_ep1", "mean"), ep2=("val_f1_ep2", "mean"), ms=("ms_per_img", "mean"),
    train_s=("train_s", "mean"), params=("params", "first"), vqc_params=("vqc_params", "first")).reset_index()
best = agg.loc[agg.groupby(["dataset", "n", "encoding"])["val"].idxmax()]
at1 = agg[agg.gamma == 1.0].set_index(["dataset", "n", "encoding"])

out("# Round 1 results\n")
for ds in sorted(hyb.dataset.unique()):
    out(f"## {ds}\n")
    out("| Model | qubits / dim | Trainable params | Accuracy | Macro-F1 | ms / image (head only) |")
    out("|---|---|---|---|---|---|")
    b = base[base.dataset == ds].groupby(["model", "n"]).agg(
        acc=("acc", "mean"), f1=("f1", "mean"), f1_std=("f1", "std"), ms=("ms_per_img", "mean"),
        params=("params", "first")).reset_index()
    for _, r in b.sort_values(["model", "n"]).iterrows():
        sd = "" if np.isnan(r.f1_std) else f" ± {r.f1_std:.3f}"
        out(f"| classical: {r.model} | {r.n} | {r.params} | {r.acc:.3f} | {r.f1:.3f}{sd} | {r.ms:.3f} |")
    for _, r in best[best.dataset == ds].sort_values(["n", "encoding"]).iterrows():
        g1 = at1.loc[(ds, r.n, r.encoding), "f1"]
        out(f"| hybrid: {r.encoding} (γ={r.gamma:g}; at γ=1: F1 {g1:.3f}) | {r.n} | {r.params} (VQC {r.vqc_params}) "
            f"| {r.acc:.3f} | {r.f1:.3f} ± {r.f1_std:.3f} | {r.ms:.3f} |")
    out("")

# ---------- 2. Method A: does the a-priori criterion predict the trained result? ----------
m = met.merge(agg, on=["dataset", "n", "encoding", "gamma"])
m["combo"] = 0.0
for col in ["kta", "hs", "probe"]:
    m["combo"] += m.groupby(["dataset", "n"])[col].rank(pct=True)
CRIT = ["kta", "hs", "probe", "combo"]

out("## Method A: correlation of a-priori metrics with trained test F1\n")
out("Spearman ρ over all (encoding, γ) configurations within each dataset × qubit count, then averaged.\n")
out("| Metric | mean ρ | min ρ | max ρ |")
out("|---|---|---|---|")
for c in CRIT + ["var"]:
    rs = [spearmanr(g[c], g["f1"]).statistic for _, g in m.groupby(["dataset", "n"])]
    out(f"| {c} | {np.mean(rs):.2f} | {np.min(rs):.2f} | {np.max(rs):.2f} |")
out("")

# encoding + gamma selection: pick argmax of the criterion over all (encoding, gamma) configs
out("## Method A: choosing encoding and γ without training\n")
out("Oracle = best test F1 over all trained configs. Regret = oracle F1 − F1 of the chosen config. "
    "Short training = pick config by validation F1 after 1 or 2 epochs (needs every config trained that long).\n")
out("| Selector | top-1 encoding hit rate | mean regret (F1) | max regret | cost (s, all configs) |")
out("|---|---|---|---|---|")
rows = []
for (ds, n), g in m.groupby(["dataset", "n"]):
    oracle = g.loc[g.f1.idxmax()]
    rec = {"dataset": ds, "n": n, "oracle_enc": oracle.encoding, "oracle_gamma": oracle.gamma, "oracle_f1": oracle.f1}
    epoch_s = g.train_s / 10  # 10 epochs per run
    for sel, score, cost in [(c, g[c], g.time_s.sum()) for c in CRIT] + [
            ("short training (1 epoch)", g.ep1, epoch_s.sum()), ("short training (2 epochs)", g.ep2, 2 * epoch_s.sum()),
            ("always angle, γ=1", None, 0.0), ("full training (val F1)", g.val, g.train_s.sum())]:
        ch = g[(g.encoding == "angle") & (g.gamma == 1.0)].iloc[0] if score is None else g.loc[score.idxmax()]
        rec[sel] = (ch.encoding == oracle.encoding, oracle.f1 - ch.f1, cost, ch.encoding, ch.gamma)
    rows.append(rec)
for sel in CRIT + ["short training (1 epoch)", "short training (2 epochs)", "always angle, γ=1", "full training (val F1)"]:
    hit = np.mean([r[sel][0] for r in rows]); reg = [r[sel][1] for r in rows]; cost = np.mean([r[sel][2] for r in rows])
    out(f"| {sel} | {hit:.0%} | {np.mean(reg):.3f} | {np.max(reg):.3f} | {cost:.0f} |")
out("\nPer case (dataset, qubits): oracle vs choice of `kta` and of 1-epoch training\n")
out("| dataset | n | oracle | kta picks | 1-epoch picks |")
out("|---|---|---|---|---|")
for r in rows:
    c, s = r["kta"], r["short training (1 epoch)"]
    out(f"| {r['dataset']} | {r['n']} | {r['oracle_enc']} γ={r['oracle_gamma']:g} (F1 {r['oracle_f1']:.3f}) "
        f"| {c[3]} γ={c[4]:g} (−{c[1]:.3f}) | {s[3]} γ={s[4]:g} (−{s[1]:.3f}) |")
out("")

# gamma choice within a fixed encoding
out("## Method A: choosing γ for a fixed encoding\n")
out("| Selector | mean regret (F1) | max regret |")
out("|---|---|---|")
for c in CRIT:
    reg = [g.f1.max() - g.loc[g[c].idxmax(), "f1"] for _, g in m.groupby(["dataset", "n", "encoding"])]
    out(f"| {c} | {np.mean(reg):.3f} | {np.max(reg):.3f} |")
reg = [g.f1.max() - g[g.gamma == 1.0].f1.iloc[0] for _, g in m.groupby(["dataset", "n", "encoding"])]
out(f"| fixed γ=1 | {np.mean(reg):.3f} | {np.max(reg):.3f} |")
(R / "summary.md").write_text("\n".join(lines))
print("\n".join(lines))

# ---------- figures ----------
for ds in sorted(m.dataset.unique()):
    ns = sorted(m.n.unique())
    fig, axes = plt.subplots(2, len(ns), figsize=(4.2 * len(ns), 6.5), sharex=True)
    for j, n in enumerate(ns):
        for e in ENC:
            g = m[(m.dataset == ds) & (m.n == n) & (m.encoding == e)].sort_values("gamma")
            axes[0, j].errorbar(g.gamma, g.f1, yerr=g.f1_std, marker="o", color=COL[e], label=e, capsize=3)
            axes[1, j].plot(g.gamma, g.kta, marker="s", color=COL[e], label=e)
        axes[0, j].set_title(f"{ds}, {n} qubits")
        axes[1, j].set_xscale("log"); axes[0, j].set_xscale("log")
        axes[1, j].set_xlabel("scale γ")
    axes[0, 0].set_ylabel("test macro-F1 (trained)"); axes[1, 0].set_ylabel("KTA (before training)")
    axes[0, 0].legend()
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / f"gamma_sweep_{ds}.png", dpi=130); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, c in zip(axes, ["kta", "hs", "probe"]):
    for e in ENC:
        g = m[m.encoding == e]
        ax.scatter(g[c], g.f1, s=18, color=COL[e], label=e, alpha=0.8)
    ax.set_xlabel(f"{c} (before training)"); ax.grid(alpha=0.3)
axes[0].set_ylabel("test macro-F1 (trained)"); axes[0].legend()
fig.tight_layout(); fig.savefig(FIG / "metric_vs_f1.png", dpi=130); plt.close(fig)
print("figures in", FIG)
