"""Quantum encodings of feature vectors and the encoding-aware adapter normalisation.

Each encoding declares how many input values it needs for n qubits (k) and how an adapter
output u in R^k is mapped into the range the encoding expects, scaled by gamma.
"""
import numpy as np
import pennylane as qml
import torch

PI = np.pi


class Encoding:
    name = "base"
    whiten = True  # whether the PCA-initialised adapter whitens its output

    def __init__(self, n_qubits):
        self.n = n_qubits

    @property
    def k(self):
        return self.n

    def normalize(self, u, gamma):
        raise NotImplementedError

    def circuit(self, x):
        raise NotImplementedError


class AngleEncoding(Encoding):
    """H then RY(x_i) on each qubit (as in Mari et al., 2020); x in [-gamma*pi/2, gamma*pi/2].

    Starting from |+> makes <Z_i> ~ -sin(x_i), an odd, monotonic function of the feature on this
    range. Plain RY on |0> gives <Z_i> ~ cos(x_i), which is even and throws away the sign of every
    centred feature (round-1 experiments collapsed to ~2 separable groups because of this)."""
    name = "angle"

    def normalize(self, u, gamma):
        return gamma * (PI / 2) * torch.tanh(u)

    def circuit(self, x):
        for i in range(self.n):
            qml.Hadamard(wires=i)
        qml.AngleEmbedding(x, wires=range(self.n), rotation="Y")


class ZZEncoding(Encoding):
    """Havlicek et al. ZZ feature map: H, RZ(2x_i), ZZ(2(pi-x_i)(pi-x_j)) on all pairs, `reps` times."""
    name = "zz"

    def __init__(self, n_qubits, reps=2):
        super().__init__(n_qubits)
        self.reps = reps

    def normalize(self, u, gamma):
        return gamma * PI * torch.sigmoid(u)

    def circuit(self, x):
        for _ in range(self.reps):
            for i in range(self.n):
                qml.Hadamard(wires=i)
                qml.RZ(2.0 * x[..., i], wires=i)
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    qml.CNOT(wires=[i, j])
                    qml.RZ(2.0 * (PI - x[..., i]) * (PI - x[..., j]), wires=j)
                    qml.CNOT(wires=[i, j])


class AmplitudeEncoding(Encoding):
    """Amplitude encoding of 2^n values. The last amplitude is a constant c/gamma that keeps
    information about the norm of u, which plain L2 normalisation would discard. The adapter
    output has total variance k, so ||u|| ~ sqrt(k), so c = sqrt(k)/gamma makes gamma comparable across qubit counts."""
    name = "amplitude"
    whiten = False  # 2^n-1 components: whitening would blow up the noisy trailing ones

    @property
    def k(self):
        return 2 ** self.n - 1

    def normalize(self, u, gamma):
        c = torch.full_like(u[..., :1], np.sqrt(self.k) / gamma)
        v = torch.cat([u, c], dim=-1)
        return v / v.norm(dim=-1, keepdim=True)

    def circuit(self, x):
        qml.AmplitudeEmbedding(x, wires=range(self.n), normalize=False)


ENCODINGS = {"angle": AngleEncoding, "zz": ZZEncoding, "amplitude": AmplitudeEncoding}


def make_encoding(name, n_qubits):
    return ENCODINGS[name](n_qubits)
