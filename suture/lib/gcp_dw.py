"""Domain-wall K=3 encoding for GCP (P-T6b arm).

Codec (2 qubits per vertex, big-endian (b1, b2) = (2u, 2u+1)):
    00 -> color 0     10 -> color 1     11 -> color 2     01 -> INVALID

Transitions on coloring space are the same swaps the one-hot chain uses:
  c0<->c1 : |10><00| + h.c.   (rotation in span{00,10})
  c0<->c2 : |11><00| + h.c.   (rotation in span{00,11})  [2-bit flip]
Trained angles are ENCODING-INDEPENDENT (the objective lives on coloring
space), so domain-wall reuses the one-hot chain's COBYLA parameters —
a matched-chain comparison isolating the encoding.

Single-flip structure (contrast with T6.1 one-hot):
  00 --flip b1--> 10  : SILENT identity error (c0 -> c1, still valid)
  00 --flip b2--> 01  : detected; 1-flip neighbors of 01 = {00, 11} -> list {c0, c2}
  10 --flip b1--> 00  : SILENT (c1 -> c0)
  10 --flip b2--> 11  : SILENT (c1 -> c2)
  11 --flip b1--> 01  : detected; list {c0, c2}
  11 --flip b2--> 10  : SILENT (c2 -> c1)
=> 4/6 of single flips are silent identity errors; the detected 1/3 yield
2-candidate lists. One-hot: 0 silent, every flip detected (T6.1). This is
the registered risk on P-T6b, scored below.
"""
from __future__ import annotations
import sys, os
# repo-relative path setup: lib/ itself + the upstream rasengan clone
_LIB = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_LIB))
sys.path.insert(0, os.path.join(_REPO, "rasengan"))
sys.path.insert(0, _LIB)

import numpy as np

CODE = {0: (0, 0), 1: (1, 0), 2: (1, 1)}          # color -> (b1, b2)
DECODE = {(0, 0): 0, (1, 0): 1, (1, 1): 2, (0, 1): -1}  # -1 = invalid


def dw_transition_unitary(theta, kind):
    """4x4 unitary exp(-i theta H) on (b1,b2) basis order [00,01,10,11].
    kind '01': H=|10><00|+h.c.; kind '02': H=|11><00|+h.c."""
    U = np.eye(4, dtype=complex)
    j = 2 if kind == "01" else 3
    c, s = np.cos(theta), -1j * np.sin(theta)
    U[0, 0] = c; U[j, j] = c
    U[0, j] = s; U[j, 0] = s
    return U


def dw_pair_channel_matrix(theta, kind, lam, seed=17):
    """Exact 4x4 stochastic matrix T[b_in, b_out] for the noisy dw transition
    (2-qubit depolarizing after each 2q gate layer, scaled by lam), via
    density-matrix sim of the transpiled unitary. Mirrors pair_channel_matrix
    for one-hot (gcp_k3.py)."""
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from h1_pilot_lib import make_noise_model, CORE_BASIS

    U = dw_transition_unitary(theta, kind)
    nm = make_noise_model(lam, 0.0)
    sim = AerSimulator(method="density_matrix", noise_model=nm, seed_simulator=seed)
    T = np.zeros((4, 4))
    for b_in in range(4):
        qc = QuantumCircuit(2)
        # b_in bits (b1,b2) with b1 = qubit 0
        if (b_in >> 1) & 1: qc.x(0)
        if b_in & 1: qc.x(1)
        qc.unitary(U, [1, 0])  # little-endian: qubit1 is MSB of basis index
        qc.save_density_matrix()
        tqc = transpile(qc, sim, basis_gates=CORE_BASIS + ["save_density_matrix"],
                        optimization_level=0)
        rho = np.asarray(sim.run(tqc).result().data(0)["density_matrix"])
        p = np.clip(np.real(np.diag(rho)), 0, None)
        # basis index of rho is little-endian (q1 q0); convert to (b1,b2) code
        for idx in range(4):
            q0, q1 = idx & 1, (idx >> 1) & 1   # q0 = b1, q1 = b2
            T[b_in, (q0 << 1) | q1] += p[idx]
    T /= T.sum(axis=1, keepdims=True)
    return T


def dw_gate_cost(theta, kind):
    """Transpiled 2q-gate count and depth in CORE_BASIS, vs one-hot swap."""
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from h1_pilot_lib import CORE_BASIS
    qc = QuantumCircuit(2)
    qc.unitary(dw_transition_unitary(theta, kind), [1, 0])
    tqc = transpile(qc, AerSimulator(), basis_gates=CORE_BASIS, optimization_level=1)
    n2q = sum(1 for inst in tqc.data if inst.operation.num_qubits == 2)
    return dict(gates_2q=n2q, depth=tqc.depth(), size=tqc.size())
