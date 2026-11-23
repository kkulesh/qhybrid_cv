"""Run a circuit from the web composer in PennyLane (default.qubit): exact probabilities, state and
sampled measurement counts. The composer simulates in the browser; this is the same circuit executed
by the library the hybrid models use, so the two results can be compared."""
import time

import numpy as np
import pennylane as qml

GATES = {
    "H": qml.Hadamard, "X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ, "S": qml.S, "T": qml.T, "SX": qml.SX,
    "SDG": lambda wires: qml.adjoint(qml.S(wires=wires)), "TDG": lambda wires: qml.adjoint(qml.T(wires=wires)),
    "RX": qml.RX, "RY": qml.RY, "RZ": qml.RZ, "P": qml.PhaseShift,
    "CX": qml.CNOT, "CZ": qml.CZ, "SWAP": qml.SWAP, "CCX": qml.Toffoli, "RZZ": qml.IsingZZ,
}
PARAMS = {"RX": 1, "RY": 1, "RZ": 1, "P": 1, "RZZ": 1}
ARITY = {"CX": 2, "CZ": 2, "SWAP": 2, "RZZ": 2, "CCX": 3}
MAX_QUBITS = 12


def _apply(ops):
    for op in ops:
        k = op["kind"]
        if k in ("MEASURE", "BARRIER"):
            continue
        g = GATES[k]
        if PARAMS.get(k):
            g(op["params"][0], wires=op["wires"])
        else:
            g(wires=op["wires"] if len(op["wires"]) > 1 else op["wires"][0])


def validate(n, ops):
    if not 1 <= n <= MAX_QUBITS:
        raise ValueError(f"Кількість кубітів має бути від 1 до {MAX_QUBITS}.")
    for op in ops:
        k, w = op["kind"], op["wires"]
        if k not in GATES and k not in ("MEASURE", "BARRIER"):
            raise ValueError(f"Невідомий вентиль {k}.")
        if any(not 0 <= q < n for q in w) or len(set(w)) != len(w) or len(w) != ARITY.get(k, 1):
            raise ValueError(f"Некоректні кубіти для вентиля {k}: {w}.")
        if PARAMS.get(k) and len(op["params"]) < 1:
            raise ValueError(f"Вентиль {k} потребує кут.")


def run(n, ops, shots, seed):
    """Composer convention: qubit q is bit q of the basis index (q[0] least significant), so PennyLane's
    wire order is reversed when reading probabilities and the state."""
    validate(n, ops)
    ops = sorted(ops, key=lambda o: o["col"])
    measured = sorted({o["wires"][0] for o in ops if o["kind"] == "MEASURE"})
    rev = list(range(n - 1, -1, -1))
    t0 = time.perf_counter()

    @qml.qnode(qml.device("default.qubit", wires=n))
    def exact():
        _apply(ops)
        return qml.probs(wires=rev), qml.state()

    probs, state = exact()
    out = {"probs": np.asarray(probs).tolist(), "measured": measured}
    # state is indexed with wire 0 as the most significant bit; reorder to the composer's indexing
    st = np.asarray(state).reshape([2] * n).transpose(rev).reshape(-1)
    out["state"] = [[float(z.real), float(z.imag)] for z in st]
    if measured:
        order = sorted(measured, reverse=True)  # bitstring c[high] ... c[low], as in the composer

        def sampled():
            _apply(ops)
            return qml.counts(wires=order)

        if hasattr(qml, "set_shots"):  # PennyLane >= 0.42; older versions take shots on the device
            sampled = qml.set_shots(qml.QNode(sampled, qml.device("default.qubit", wires=n, seed=seed)), shots=shots)
        else:
            sampled = qml.QNode(sampled, qml.device("default.qubit", wires=n, shots=shots, seed=seed))
        out["counts"] = {str(k): int(v) for k, v in sampled().items()}
    out["time_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return out
