"""Hybrid model: encoding-aware adapter -> VQC -> classical head; and classical controls."""
import numpy as np
import pennylane as qml
import torch
from torch import nn

# Quantum devices the circuits can run on. Training always uses default.qubit (exact, backprop);
# braket.local.qubit is the free local simulator of the Amazon Braket SDK (amazon-braket-pennylane-plugin),
# offered for running trained models and composer circuits. No cloud (QPU) devices are used.
DEVICES = {
    "default.qubit": "Сервер: PennyLane, вбудований симулятор (default.qubit)",
    "braket.local.qubit": "Сервер: Amazon Braket, локальний симулятор (braket.local.qubit)",
}


def make_device(name, wires, seed=None):
    """A local simulator from DEVICES (ValueError for anything else, so no cloud device can be requested).
    The seed is used by default.qubit only: the Braket local simulator has no seed option."""
    if name not in DEVICES:
        raise ValueError(f"Невідомий пристрій {name}. Доступні: {', '.join(DEVICES)}.")
    kw = {"seed": seed} if seed is not None and name == "default.qubit" else {}
    try:
        return qml.device(name, wires=wires, **kw)
    except Exception as e:  # the plugin is not installed or failed to load
        raise RuntimeError(f"Пристрій {name} недоступний ({e}). Встановіть плагін: "
                           "pip install amazon-braket-pennylane-plugin") from e


def with_shots(qnode, shots):
    """Finite-shot execution of a QNode (PennyLane >= 0.42 sets shots on the QNode, not the device)."""
    return qnode if not shots else qml.set_shots(qnode, shots=shots)


class Adapter(nn.Module):
    """Linear projection d -> k, initialised from PCA, followed by the encoding's
    normalisation with scale gamma."""

    def __init__(self, d, k, encoding, gamma, pca=None):
        super().__init__()
        self.lin = nn.Linear(d, k)
        if pca is not None:
            W, b = pca
            self.lin.weight.data = torch.tensor(W, dtype=torch.float32)
            self.lin.bias.data = torch.tensor(b, dtype=torch.float32)
        self.encoding, self.gamma = encoding, gamma

    def forward(self, z):
        return self.encoding.normalize(self.lin(z), self.gamma)


def pca_init(Z, k, whiten=True):
    """PCA projection as (W, b) so that W z + b has zero mean. With whiten=True every component
    has unit variance; otherwise all components share one scale (total variance k), which keeps
    the leading components dominant."""
    mu = Z.mean(0)
    k_eff = min(k, Z.shape[1], Z.shape[0] - 1)
    # eigendecomposition of the d x d covariance is much cheaper than an SVD of the n x d data
    evals, evecs = np.linalg.eigh(np.cov(Z - mu, rowvar=False))
    order = np.argsort(evals)[::-1][:k_eff]
    Vt, std = evecs[:, order].T, np.sqrt(np.maximum(evals[order], 1e-12))
    W = np.zeros((k, Z.shape[1]), dtype=np.float32)
    scale = std[:, None] if whiten else np.sqrt((std ** 2).sum() / k)
    W[:k_eff] = Vt[:k_eff] / scale
    return W, -(W @ mu)


class VQC(nn.Module):
    """Encoding + StronglyEntanglingLayers. readout="z" measures <Z_i> (n outputs); readout="zz"
    adds all two-qubit correlators <Z_i Z_j> (n + n(n-1)/2 outputs). reupload=True repeats the
    encoding before every variational layer (data re-uploading, Perez-Salinas et al., 2020)."""

    def __init__(self, encoding, n_layers, seed=0, readout="z", reupload=False):
        super().__init__()
        self.encoding, self.n = encoding, encoding.n
        n = self.n
        obs = [qml.PauliZ(i) for i in range(n)]
        if readout == "zz":
            obs += [qml.PauliZ(i) @ qml.PauliZ(j) for i in range(n) for j in range(i + 1, n)]
        self.n_out = len(obs)

        def circuit(inputs, weights):
            if reupload:
                for layer in range(n_layers):
                    encoding.circuit(inputs)
                    qml.StronglyEntanglingLayers(weights[layer:layer + 1], wires=range(n))
            else:
                encoding.circuit(inputs)
                qml.StronglyEntanglingLayers(weights, wires=range(n))
            return [qml.expval(o) for o in obs]

        # training and the default inference: exact default.qubit with backpropagation
        self.qnode = qml.QNode(circuit, qml.device("default.qubit", wires=n), interface="torch", diff_method="backprop")
        self._circuit, self._qnodes = circuit, {}

        g = torch.Generator().manual_seed(seed)
        self.weights = nn.Parameter(0.1 * torch.randn(n_layers, n, 3, generator=g))

    def forward(self, x):
        return torch.stack(self.qnode(x, self.weights), dim=-1).float()

    @torch.no_grad()
    def run_on(self, x, device="default.qubit", shots=None):
        """Inference with the trained weights on another local device and/or with a finite number of
        shots (then every <Z> is estimated from samples). Samples are run one by one, because not every
        device supports parameter broadcasting."""
        if device == "default.qubit" and not shots:
            return self(x)
        key = (device, shots)
        if key not in self._qnodes:
            qn = qml.QNode(self._circuit, make_device(device, self.n), interface=None)
            self._qnodes[key] = with_shots(qn, shots)
        # float64: the Braket simulator cannot take numpy float32 values as circuit parameters
        qn, w = self._qnodes[key], self.weights.detach().double().numpy()
        rows = [np.array(qn(xi, w), dtype=np.float64) for xi in x.detach().double().numpy()]
        return torch.tensor(np.stack(rows), dtype=torch.float32)


class HybridModel(nn.Module):
    def __init__(self, d, n_classes, encoding, gamma, n_layers, pca=None, seed=0, readout="z", reupload=False):
        super().__init__()
        self.adapter = Adapter(d, encoding.k, encoding, gamma, pca)
        self.vqc = VQC(encoding, n_layers, seed, readout, reupload)
        # Expectation values after entangling layers often vary by only ~0.01-0.05 across inputs
        # (information spreads into multi-qubit correlators), which starves the linear head of
        # gradient. BatchNorm rescales each measured value before the head; it adds 2 * n_out parameters.
        self.norm = nn.BatchNorm1d(self.vqc.n_out)
        self.head = nn.Linear(self.vqc.n_out, n_classes)

    def forward(self, z):
        return self.head(self.norm(self.vqc(self.adapter(z))))

    @torch.no_grad()
    def predict_on(self, z, device="default.qubit", shots=None):
        """Logits with the circuit executed on the given local device (see VQC.run_on)."""
        return self.head(self.norm(self.vqc.run_on(self.adapter(z), device, shots)))


class BottleneckMLP(nn.Module):
    """Classical control for the VQC: the same n-dimensional bottleneck and bounded activations,
    with one n x n hidden layer in place of the circuit (n^2 + n parameters, close to the
    3 * n * layers parameters of a 2-layer VQC)."""

    def __init__(self, d, n, n_classes, pca=None):
        super().__init__()
        self.proj = nn.Linear(d, n)
        if pca is not None:
            self.proj.weight.data = torch.tensor(pca[0], dtype=torch.float32)
            self.proj.bias.data = torch.tensor(pca[1], dtype=torch.float32)
        self.mid = nn.Sequential(nn.Tanh(), nn.Linear(n, n), nn.Tanh())
        self.head = nn.Linear(n, n_classes)

    def forward(self, z):
        return self.head(self.mid(self.proj(z)))
