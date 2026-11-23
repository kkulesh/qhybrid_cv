"""Figures for the written report: dataset balance, VOC confusion matrix, per-class F1,
F1 vs qubits of the deployable models, inference-time breakdown, adapter output histograms.

    python -m scripts.report_figures --out results/report_figures
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, f1_score

from qhybrid.encodings import make_encoding
from qhybrid.models import HybridModel

p = argparse.ArgumentParser()
p.add_argument("--features", default="features/voc.npz")
p.add_argument("--models", default="models/voc")
p.add_argument("--stats", default="results/voc_round1/dataset_stats.json")
p.add_argument("--out", default="results/report_figures")
a = p.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e4e3df"
COL = {"amplitude": "#2a78d6", "angle": "#eb6834", "zz": "#1baf7a", "classical": "#52514e"}
NAME = {"amplitude": "Амплітудне", "angle": "Кутове", "zz": "ZZ-відображення", "classical": "Класична (1280 ознак)"}
UK = dict(aeroplane="літак", bicycle="велосипед", bird="птах", boat="човен", bottle="пляшка", bus="автобус",
          car="автомобіль", cat="кіт", chair="стілець", cow="корова", diningtable="стіл", dog="собака",
          horse="кінь", motorbike="мотоцикл", person="людина", pottedplant="рослина", sheep="вівця",
          sofa="диван", train="потяг", tvmonitor="монітор")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})
BLUES = LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])

d = np.load(a.features, allow_pickle=True)
Zte, yte, classes = torch.tensor(d["Z_test"], dtype=torch.float32), d["y_test"], list(d["classes"])
reg = json.loads((Path(a.models) / "registry.json").read_text(encoding="utf-8"))
torch.set_num_threads(4)


def load(meta):
    if meta["kind"] == "classical":
        m = torch.nn.Linear(reg["feature_dim"], len(classes))
    else:
        m = HybridModel(reg["feature_dim"], len(classes), make_encoding(meta["encoding"], meta["n"]), meta["gamma"], meta["layers"])
    m.load_state_dict(torch.load(Path(a.models) / f"{meta['id']}.pt", map_location="cpu"))
    return m.eval()


models = {m["id"]: (m, load(m)) for m in reg["models"]}
pred = {}
with torch.no_grad():
    for mid, (meta, m) in models.items():
        pred[mid] = m(Zte).argmax(1).numpy()

# 1. dataset balance
st = json.loads(Path(a.stats).read_text(encoding="utf-8"))
order = sorted(classes, key=lambda c: -st["train"][c])
fig, ax = plt.subplots(figsize=(9, 3.6))
x = np.arange(len(order))
ax.bar(x - 0.2, [st["train"][c] for c in order], 0.38, color=COL["amplitude"], label="trainval (навчання)")
ax.bar(x + 0.2, [st["test"][c] for c in order], 0.38, color=COL["angle"], label="test (тест)")
ax.set_xticks(x, [UK[c] for c in order], rotation=55, ha="right"); ax.set_yscale("log")
ax.set_ylabel("Кількість вирізок (лог. шкала)"); ax.legend(frameon=False); ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(out / "voc_class_balance.png"); plt.close(fig)

# 2. confusion matrix of the best hybrid
cm = confusion_matrix(yte, pred["amplitude-8q"], normalize="true")
fig, ax = plt.subplots(figsize=(7.4, 6.4))
im = ax.imshow(cm, cmap=BLUES, vmin=0, vmax=1); ax.grid(False)
lab = [UK[c] for c in classes]
ax.set_xticks(range(20), lab, rotation=60, ha="right", fontsize=8); ax.set_yticks(range(20), lab, fontsize=8)
for i in range(20):
    for j in range(20):
        if cm[i, j] >= 0.05:
            ax.text(j, i, f"{cm[i, j]:.2f}"[1:] if cm[i, j] < 1 else "1", ha="center", va="center", fontsize=6,
                    color="#ffffff" if cm[i, j] > 0.55 else INK)
ax.set_xlabel("Передбачений клас"); ax.set_ylabel("Справжній клас")
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Частка прикладів класу")
fig.tight_layout(); fig.savefig(out / "voc_confusion_amplitude8.png"); plt.close(fig)

# 3. per-class F1
sel = ["classical", "amplitude-8q", "angle-8q", "zz-6q"]
f1c = {mid: f1_score(yte, pred[mid], average=None) for mid in sel}
ordc = np.argsort(-f1c["amplitude-8q"])
fig, ax = plt.subplots(figsize=(9.5, 3.8))
w = 0.2
for k, mid in enumerate(sel):
    enc = models[mid][0]["encoding"] or "classical"
    lbl = NAME[enc] + ("" if enc == "classical" else f", {models[mid][0]['n']} кубітів")
    ax.bar(np.arange(20) + (k - 1.5) * w, f1c[mid][ordc], w * 0.92, color=COL[enc], label=lbl)
ax.set_xticks(range(20), [lab[i] for i in ordc], rotation=55, ha="right"); ax.set_ylim(0, 1)
ax.set_ylabel("F1 класу"); ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16)); ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(out / "voc_per_class_f1.png"); plt.close(fig)

# 4. F1 vs qubits of the deployable models
fig, ax = plt.subplots(figsize=(6, 3.8))
for enc in ["amplitude", "angle", "zz"]:
    ms = sorted([m for m in reg["models"] if m["encoding"] == enc], key=lambda m: m["n"])
    ax.plot([m["n"] for m in ms], [m["test_f1"] for m in ms], "-o", color=COL[enc], lw=2, ms=7, label=NAME[enc])
    for m in ms:
        lo, hi = min(m["test_f1_all_seeds"]), max(m["test_f1_all_seeds"])
        ax.vlines(m["n"], lo, hi, color=COL[enc], lw=1, alpha=0.5)
c = [m for m in reg["models"] if m["kind"] == "classical"][0]["test_f1"]
ax.axhline(c, color=COL["classical"], ls="--", lw=1.5, label=NAME["classical"])
ax.set_xticks([4, 6, 8]); ax.set_xlabel("Кількість кубітів"); ax.set_ylabel("Macro-F1 на тесті VOC"); ax.set_ylim(0, 1)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(out / "voc_f1_vs_qubits.png"); plt.close(fig)

# 5. head time per image (batch of 1), median over 50 runs
times = {}
with torch.no_grad():
    for mid, (meta, m) in models.items():
        z = Zte[:1]; ts = []
        for _ in range(50):
            t0 = time.perf_counter(); m(z); ts.append((time.perf_counter() - t0) * 1000)
        times[mid] = float(np.median(ts))
ids = list(times)
fig, ax = plt.subplots(figsize=(7, 3.8))
cols = [COL[models[i][0]["encoding"] or "classical"] for i in ids]
ax.barh(range(len(ids)), [times[i] for i in ids], color=cols, height=0.7)
ax.set_yticks(range(len(ids)), ids); ax.invert_yaxis(); ax.set_xscale("log")
ax.set_xlabel("Час голови на одне зображення, мс (лог. шкала, CPU)"); ax.grid(axis="y", visible=False)
for i, v in enumerate(times[k] for k in ids):
    ax.text(v * 1.08, i, f"{v:.2f}", va="center", fontsize=8, color=INK2)
fig.tight_layout(); fig.savefig(out / "voc_head_time.png"); plt.close(fig)

# 6. adapter outputs: angle / ZZ saturate
fig, axs = plt.subplots(1, 3, figsize=(10, 3))
with torch.no_grad():
    for ax, mid in zip(axs, ["angle-8q", "zz-8q", "amplitude-8q"]):
        meta, m = models[mid]
        x = m.adapter(Zte[:1000]).numpy().ravel()
        ax.hist(x, bins=60, color=COL[meta["encoding"]])
        ax.set_title(f"{NAME[meta['encoding']]}, {meta['n']} кубітів, γ={meta['gamma']}", fontsize=10)
        ax.set_xlabel("значення на вході кодування"); ax.grid(axis="x", visible=False)
axs[0].set_ylabel("кількість")
fig.tight_layout(); fig.savefig(out / "adapter_output_hist.png"); plt.close(fig)

summary = {"per_class_f1": {mid: dict(zip(classes, map(float, f1c[mid]))) for mid in sel},
           "head_ms": times,
           "macro_f1": {mid: float(f1_score(yte, pred[mid], average="macro")) for mid in pred}}
(out / "report_numbers.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary["macro_f1"], indent=0)); print(json.dumps(times, indent=0))
