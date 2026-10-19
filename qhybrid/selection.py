"""A-priori encoding selection (method A): cheap quantum-kernel metrics on a small sample.

The fidelity kernel K(x, x') = |<psi(x)|psi(x')>|^2 is computed from statevectors, so no VQC
is trained. Metrics:
  M1 kta     centred kernel-target alignment (Cortes et al., 2012)
  M2 var     variance of off-diagonal kernel entries (concentration; Thanasilp et al., 2024)
  M3 hs      mean pairwise Hilbert-Schmidt distance between class-mean states (Lloyd et al., 2020)
  M4 probe   accuracy of logistic regression on <Z_i>, <Z_i Z_j> after encoding + one random layer
"""
import numpy as np
import pennylane as qml
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from .models import pca_init


def statevectors(encoding, X):
    dev = qml.device("default.qubit", wires=encoding.n)

    @qml.qnode(dev, interface="torch")
    def state(x):
        encoding.circuit(x)
        return qml.state()

    return state(torch.as_tensor(X, dtype=torch.float64)).detach().numpy()


def fidelity_kernel(psi):
    return np.abs(psi.conj() @ psi.T) ** 2


def centred_kta(K, y):
    Y = np.eye(y.max() + 1)[y]
    Y = Y @ Y.T
    H = np.eye(len(y)) - 1.0 / len(y)
    Kc, Yc = H @ K @ H, H @ Y @ H
    return float((Kc * Yc).sum() / (np.linalg.norm(Kc) * np.linalg.norm(Yc)))


def offdiag_var(K):
    return float(K[~np.eye(len(K), dtype=bool)].var())


def hs_distance(K, y):
    """Tr[(rho_a - rho_b)^2] for class-mean pure-state ensembles, from kernel block means."""
    classes = np.unique(y)
    m = {(a, b): K[np.ix_(y == a, y == b)].mean() for a in classes for b in classes}
    d = [m[a, a] + m[b, b] - 2 * m[a, b] for i, a in enumerate(classes) for b in classes[i + 1:]]
    return float(np.mean(d))


def random_probe(encoding, X, y, seed=0):
    n = encoding.n
    dev = qml.device("default.qubit", wires=n)
    w = np.random.default_rng(seed).uniform(0, 2 * np.pi, (1, n, 3))
    obs = [qml.PauliZ(i) for i in range(n)] + [qml.PauliZ(i) @ qml.PauliZ(j)
                                               for i in range(n) for j in range(i + 1, n)]

    @qml.qnode(dev, interface="torch")
    def feats(x):
        encoding.circuit(x)
        qml.StronglyEntanglingLayers(torch.as_tensor(w), wires=range(n))
        return [qml.expval(o) for o in obs]

    F = torch.stack(feats(torch.as_tensor(X, dtype=torch.float64)), -1).detach().numpy()
    clf = LogisticRegression(max_iter=2000)
    return float(cross_val_score(clf, F, y, cv=3).mean())


def encoding_metrics(encoding, Z, y, gamma, pca=None):
    """All metrics for one encoding at one scale gamma, using the PCA-initialised adapter."""
    W, b = pca if pca is not None else pca_init(Z, encoding.k, encoding.whiten)
    X = encoding.normalize(torch.as_tensor(Z @ W.T + b, dtype=torch.float64), gamma).numpy()
    K = fidelity_kernel(statevectors(encoding, X))
    return {"kta": centred_kta(K, y), "var": offdiag_var(K), "hs": hs_distance(K, y),
            "probe": random_probe(encoding, X, y), "k_mean": float(K[~np.eye(len(K), dtype=bool)].mean())}
