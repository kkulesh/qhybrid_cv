"""Analysis for the journal article: quality of encoding selectors on a unified experiment grid.

Reads runs/<dataset>/{hybrid,metrics,baselines}.csv and writes tables (CSV + Markdown) and figures.

Ground truth for a (dataset, n) case: mean test macro-F1 over training seeds for every
configuration c = (encoding, gamma). Each selector picks one configuration; we report whether
its encoding is the best encoding, the regret F1*(best c) - F1*(chosen c), and its cost in seconds.
Selectors with randomness (sample for KTA, training seed for short training) are repeated
for every seed r and averaged.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

p = argparse.ArgumentParser()
p.add_argument("--runs", default="../runs")
p.add_argument("--out", default="../article_results")
p.add_argument("--datasets", nargs="+", default=["cifar10", "fmnist", "voc", "caltech10", "cowsheep"])
a = p.parse_args()
out = Path(a.out)
(out / "figures").mkdir(parents=True, exist_ok=True)

NAMES = {"cifar10": "CIFAR-10", "fmnist": "Fashion-MNIST", "voc": "PASCAL VOC (20)",
         "caltech10": "Caltech-101 (10)", "cowsheep": "VOC корова/вівця"}
ENC_UA = {"angle": "кутове", "zz": "ZZ", "amplitude": "амплітудне"}
SHORT_FRAC = 0.1  # short training = 10 % of the full epoch budget (1 of 10, 3 of 30)

H, M, B = [], [], []
for d in a.datasets:
    r = Path(a.runs) / d
    if not (r / "hybrid.csv").exists():
        continue
    H.append(pd.read_csv(r / "hybrid.csv"))
    M.append(pd.read_csv(r / "metrics.csv"))
    B.append(pd.read_csv(r / "baselines.csv"))
H, M, B = pd.concat(H), pd.concat(M), pd.concat(B)
T = pd.read_csv(Path(a.runs) / "selection_timing.csv").set_index(["dataset", "n", "sample_seed"]).time_s
H["val"] = H.val_hist.str.split().apply(lambda v: [float(x) for x in v])
H["epochs"] = H.val.apply(len)
H["short_ep"] = (H.epochs * SHORT_FRAC).round().astype(int).clip(lower=1)
H["val_short"] = [v[k - 1] for v, k in zip(H.val, H.short_ep)]
H["val_full"] = H.val.apply(lambda v: v[-1])
H["t_short"] = H.train_s * H.short_ep / H.epochs
datasets = [d for d in a.datasets if d in set(H.dataset)]

rows, sel_rows, corr_rows = [], [], []
for (d, n), h in H.groupby(["dataset", "n"]):
    m = M[(M.dataset == d) & (M.n == n)]
    truth = h.groupby(["encoding", "gamma"]).f1.mean()
    sd = h.groupby(["encoding", "gamma"]).f1.std()
    best_c = truth.idxmax()
    best_enc_f1 = truth.groupby(level=0).max()
    best_enc = best_enc_f1.idxmax()
    gap = best_enc_f1.sort_values().iloc[-1] - best_enc_f1.sort_values().iloc[-2]
    kta_mean = m.groupby(["encoding", "gamma"]).kta.mean()
    rho_all = spearmanr(kta_mean.loc[truth.index], truth).correlation
    kta_enc = kta_mean.groupby(level=0).max()
    rho_enc = spearmanr(kta_enc.loc[best_enc_f1.index], best_enc_f1).correlation
    corr_rows.append(dict(dataset=d, n=n, rho_configs=rho_all, rho_encodings=rho_enc,
                          best_encoding=best_enc, gap_to_second=gap,
                          **{f"f1_{e}": best_enc_f1.get(e, np.nan) for e in ENC_UA}))

    seeds = sorted(h.seed.unique())

    def record(name, c, cost):
        sel_rows.append(dict(dataset=d, n=n, selector=name, encoding=c[0], gamma=c[1],
                             hit=c[0] == best_enc, regret=truth.max() - truth[c], cost_s=cost))

    full_cost_per_seed = h.groupby("seed").train_s.sum().mean()
    for r in seeds:
        hr = h[h.seed == r].set_index(["encoding", "gamma"])
        mr = m[m.sample_seed == r].set_index(["encoding", "gamma"])
        kta_cost = T[(d, n, r)]  # whole KTA procedure, measured without other load
        # 1. KTA over all (encoding, gamma)
        record("KTA", mr.kta.idxmax(), kta_cost)
        # 2. short training of every configuration
        record("Коротке навчання", hr.val_short.idxmax(), hr.t_short.sum())
        # 3. full search on validation
        record("Повний перебір", hr.val_full.idxmax(), hr.train_s.sum())
        # 4. two-stage: KTA keeps top-2 encodings, short training picks (encoding, gamma) among them
        enc_score = mr.kta.groupby(level=0).max().sort_values(ascending=False)
        keep = list(enc_score.index[:2])
        sub = hr[hr.index.get_level_values(0).isin(keep)]
        record("Двоетапний (KTA top-2 + коротке)", sub.val_short.idxmax(), kta_cost + sub.t_short.sum())
        # 5. KTA chooses the encoding, short training chooses gamma inside it
        sub1 = hr[hr.index.get_level_values(0) == enc_score.index[0]]
        record("KTA кодування + коротке γ", sub1.val_short.idxmax(), kta_cost + sub1.t_short.sum())
    # fixed default and random choice (expectation)
    if ("angle", 1.0) in truth.index:
        record("Завжди кутове, γ = 1", ("angle", 1.0), 0.0)
    for c in truth.index:
        sel_rows.append(dict(dataset=d, n=n, selector="Випадковий вибір", encoding=c[0], gamma=c[1],
                             hit=c[0] == best_enc, regret=truth.max() - truth[c], cost_s=0.0))
    for c in truth.index:
        rows.append(dict(dataset=d, n=n, encoding=c[0], gamma=c[1], f1=truth[c], f1_sd=sd[c],
                         kta=kta_mean.get(c, np.nan)))

S = pd.DataFrame(sel_rows)
C = pd.DataFrame(corr_rows)
G = pd.DataFrame(rows)
S.to_csv(out / "selectors_raw.csv", index=False)
C.to_csv(out / "correlation.csv", index=False)
G.to_csv(out / "grid_f1_kta.csv", index=False)

order = ["Випадковий вибір", "Завжди кутове, γ = 1", "KTA", "KTA кодування + коротке γ",
         "Двоетапний (KTA top-2 + коротке)", "Коротке навчання", "Повний перебір"]
case = S.groupby(["selector", "dataset", "n"]).agg(hit=("hit", "mean"), regret=("regret", "mean"),
                                                    cost=("cost_s", "mean")).reset_index()
summary = case.groupby("selector").agg(cases=("hit", "size"), hit=("hit", "mean"),
                                       regret_mean=("regret", "mean"), regret_max=("regret", "max"),
                                       cost_mean_s=("cost", "mean")).reindex(order).dropna(how="all")
full = summary.loc["Повний перебір", "cost_mean_s"]
summary["speedup_vs_full"] = full / summary.cost_mean_s.replace(0, np.nan)
summary.to_csv(out / "selectors_summary.csv")
per_ds = case.pivot_table(index="selector", columns="dataset", values="hit").reindex(order)
per_ds_reg = case.pivot_table(index="selector", columns="dataset", values="regret").reindex(order)
per_ds.to_csv(out / "selectors_hit_per_dataset.csv")
per_ds_reg.to_csv(out / "selectors_regret_per_dataset.csv")

# baselines with matching width / parameters
bl = B.groupby(["dataset", "model", "n"]).agg(f1=("f1", "mean"), f1_sd=("f1", "std"),
                                              params=("params", "first")).reset_index()
bl.to_csv(out / "baselines_summary.csv", index=False)
best = (G.sort_values("f1").groupby(["dataset", "n", "encoding"]).tail(1)
        .sort_values(["dataset", "n", "encoding"]))
params = H.groupby(["dataset", "n", "encoding"]).params.first()
best["params"] = [params[(r.dataset, r.n, r.encoding)] for r in best.itertuples()]
best.to_csv(out / "best_per_encoding.csv", index=False)

# paired test: amplitude hybrid (best gamma) vs linear bottleneck 2^n-1, over seeds and cases
pairs = []
for (d, n), g in best[best.encoding == "amplitude"].groupby(["dataset", "n"]):
    gam = g.gamma.iloc[0]
    hy = H[(H.dataset == d) & (H.n == n) & (H.encoding == "amplitude") & (H.gamma == gam)].set_index("seed").f1
    lb = B[(B.dataset == d) & (B.model == "linear_bottleneck") & (B.n == 2 ** n - 1)].set_index("seed").f1
    common = hy.index.intersection(lb.index)
    pairs += list(zip(hy[common], lb[common]))
if len(pairs) > 5:
    x, y = np.array(pairs).T
    st = wilcoxon(x, y)
    json.dump({"amplitude_vs_linear_2n-1": {"pairs": len(x), "mean_diff": float((x - y).mean()),
               "wins": int((x > y).sum()), "p": float(st.pvalue)}},
              open(out / "tests.json", "w"), indent=1, ensure_ascii=False)

# ---------------- figures ----------------
COL = {"angle": "#1f77b4", "zz": "#d62728", "amplitude": "#2ca02c"}
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})

# Fig: KTA vs F1, every configuration, one panel per dataset
fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.4))
axes = [list(axs.flat)]
for ax, d in zip(axes[0], datasets):
    g = G[G.dataset == d]
    for e in ENC_UA:
        ge = g[g.encoding == e]
        ax.scatter(ge.kta, ge.f1, s=12, alpha=.7, c=COL[e], label=ENC_UA[e])
    rho = spearmanr(g.kta, g.f1).correlation
    ax.set_title(f"{NAMES[d]}\nρ = {rho:.2f}", fontsize=9)
    ax.set_xlabel("KTA")
    ax.grid(alpha=.3)
for ax in (axs[0][0], axs[1][0]):
    ax.set_ylabel("macro-F1 після навчання")
h, l = axs[0][0].get_legend_handles_labels()
axs[1][2].axis("off")
axs[1][2].legend(h, l, loc="center", fontsize=9, title="Кодування", markerscale=2)
fig.tight_layout()
fig.savefig(out / "figures/kta_vs_f1.png", dpi=200)
plt.close(fig)

# Fig: gamma curves (KTA and F1) for the largest n of each dataset
gds = [d for d in ["voc", "caltech10", "cowsheep"] if d in datasets]
fig, axes = plt.subplots(2, len(gds), figsize=(2.6 * len(gds), 4.0), squeeze=False, sharex=True)
for j, d in enumerate(gds):
    g = G[G.dataset == d]
    nmax = g.n.max() if d != "cowsheep" else 2
    g = g[g.n == nmax]
    for e in ENC_UA:
        ge = g[g.encoding == e].sort_values("gamma")
        axes[0][j].plot(ge.gamma, ge.kta, "o-", ms=3, c=COL[e], label=ENC_UA[e])
        axes[1][j].errorbar(ge.gamma, ge.f1, yerr=ge.f1_sd, fmt="o-", ms=3, c=COL[e], capsize=2)
    axes[0][j].set_title(f"{NAMES[d]}, n = {nmax}", fontsize=9)
    for i in range(2):
        axes[i][j].set_xscale("log")
        axes[i][j].grid(alpha=.3)
    axes[1][j].set_xlabel("γ")
axes[0][0].set_ylabel("KTA (до навчання)")
axes[1][0].set_ylabel("macro-F1 (після навчання)")
axes[0][0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(out / "figures/gamma_curves.png", dpi=200)
plt.close(fig)

# Fig: cost vs regret per selector
fig, ax = plt.subplots(figsize=(5.2, 3.2))
for s, r in summary.iterrows():
    if r.cost_mean_s <= 0:
        continue
    ax.scatter(r.cost_mean_s, r.regret_mean, s=40)
    ax.annotate(s, (r.cost_mean_s, r.regret_mean), fontsize=7, xytext=(4, 3), textcoords="offset points")
ax.set_xscale("log")
ax.set_xlabel("Середні витрати на вибір, с (лог. шкала)")
ax.set_ylabel("Середня втрата F1 (регрет)")
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(out / "figures/cost_vs_regret.png", dpi=200)
plt.close(fig)

with open(out / "summary.md", "w", encoding="utf-8") as f:
    f.write("# Selectors\n\n" + summary.round(4).to_markdown() + "\n\n")
    f.write("## Hit rate per dataset\n\n" + per_ds.round(2).to_markdown() + "\n\n")
    f.write("## Regret per dataset\n\n" + per_ds_reg.round(3).to_markdown() + "\n\n")
    f.write("## Correlation and best encodings\n\n" + C.round(3).to_markdown(index=False) + "\n\n")
    f.write("## Best per encoding\n\n" + best.round(3).to_markdown(index=False) + "\n\n")
    f.write("## Baselines\n\n" + bl.round(3).to_markdown(index=False) + "\n")
print(summary.round(3).to_string())
print(C.round(3).to_string())
