"""Draw the hybrid circuit (encoding + StronglyEntanglingLayers + readout) for each encoding."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch

from qhybrid.encodings import make_encoding

p = argparse.ArgumentParser()
p.add_argument("--qubits", type=int, default=4)
p.add_argument("--layers", type=int, default=2)
p.add_argument("--out", default="results/circuits")
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0)
weights = torch.tensor(rng.uniform(0, 2 * np.pi, (a.layers, a.qubits, 3)))
for name in ["angle", "zz", "amplitude"]:
    enc = make_encoding(name, a.qubits)
    x = enc.normalize(torch.tensor(rng.normal(size=enc.k)), 0.5)
    dev = qml.device("default.qubit", wires=a.qubits)

    @qml.qnode(dev)
    def circuit(x, w):
        enc.circuit(x)
        qml.StronglyEntanglingLayers(w, wires=range(a.qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(a.qubits)]

    for level, suffix in [("top", "compact"), ("device", "gates")]:
        fig, _ = qml.draw_mpl(circuit, level=level, decimals=None, style="pennylane")(x, weights)
        fig.suptitle(f"{name} encoding, {a.qubits} qubits, {a.layers} variational layers", fontsize=14)
        fig.savefig(out / f"{name}_{a.qubits}q_{suffix}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    (out / f"{name}_{a.qubits}q.txt").write_text(qml.draw(circuit, level="device", decimals=2)(x, weights))
    print(name, "ok")
