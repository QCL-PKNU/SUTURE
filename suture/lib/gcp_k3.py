"""Fixed-K GCP (K=3) packs for the go/no-go experiment.

Encoding: one-hot, K colors per vertex, n = v*K decision qubits, NO slack —
assignment constraints (sum_c x_{u,c} = 1) are the hard feasibility system
(purification filter), edge conflicts are the objective (standard GCP QUBO form,
CCR convention). Rationale vs Rasengan's K=V hard-edge model: keeps n=24
tractable and |F| = K^v large enough (6561) that junk controls are no longer
trivially competitive. Deviation documented in the go/no-go report.

Transitions: null-space basis of the assignment system restricted to
{-1,0,1} = within-block color swaps e_{u,c} - e_{u,c'} — 2-qubit circuits after
Rasengan's driver_component synthesis (k=2 nonzeros -> plain cp decomposition,
no ancilla).

Noise/channel model (exact, not sampled from a global simulator): a segment
applies gates ONLY on its 2 active qubits, so under per-gate depolarizing +
independent readout error the segment channel factorizes:
    (2-qubit noisy channel on the active pair) (x) identity, then
    independent readout flips on all n bits.
We precompute the 4x4 transition matrix T[b_in, b_out] of the active pair by
density-matrix simulation of the transpiled 2-qubit circuit under the scaled
noise model, then run the whole segmented chain as a classical Markov sampler.
This is exact for the stated noise model and ~10^4x faster than whole-register
Aer, enabling n=24 with dense seed grids.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator

import sys, os
# repo-relative path setup: lib/ itself + the upstream rasengan clone
_LIB = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_LIB))
sys.path.insert(0, os.path.join(_REPO, "rasengan"))
sys.path.insert(0, _LIB)
from rasengan.solvers.qiskit.circuit.hdi_decompose import driver_component

from h1_pilot_lib import make_noise_model, CORE_BASIS, BASE_PRO


# ---------------------------------------------------------------------------
# Instance
# ---------------------------------------------------------------------------

@dataclass
class GcpK3Pack:
    name: str
    v: int
    K: int
    edges: list                     # [(u, w), ...]
    chain: list                     # [(vertex, c_from_pos, c_to_pos), ...] per segment
    params: np.ndarray              # trained transition times
    seed_coloring: np.ndarray       # planted proper coloring (len v)
    n: int = 0

    def __post_init__(self):
        self.n = self.v * self.K
        self.adj = [[] for _ in range(self.v)]
        for u, w in self.edges:
            self.adj[u].append(w); self.adj[w].append(u)

    # ---- encoding helpers ----
    def coloring_to_bits(self, col) -> np.ndarray:
        x = np.zeros(self.n, dtype=np.int8)
        for u, c in enumerate(col):
            x[u * self.K + int(c)] = 1
        return x

    def bits_to_blocks(self, bits: np.ndarray):
        return bits.reshape(self.v, self.K)

    def is_feasible_bits(self, bits: np.ndarray) -> bool:
        return bool((self.bits_to_blocks(bits).sum(axis=1) == 1).all())

    def decode_coloring(self, bits: np.ndarray):
        """Feasible bits -> coloring array; caller must check feasibility."""
        return self.bits_to_blocks(bits).argmax(axis=1)

    def conflicts(self, col) -> int:
        return sum(1 for u, w in self.edges if col[u] == col[w])

    def proper(self, col) -> bool:
        return self.conflicts(col) == 0


def num_proper_colorings(v_edges, v, K):
    """Chromatic polynomial by deletion-contraction is overkill; brute force v<=10."""
    from itertools import product
    cnt = 0
    for col in product(range(K), repeat=v):
        if all(col[u] != col[w] for u, w in v_edges):
            cnt += 1
    return cnt


# ---------------------------------------------------------------------------
# Segment channel: 4x4 transition matrix of the active pair
# ---------------------------------------------------------------------------

def transition_circuit(theta_val: float) -> QuantumCircuit:
    """2-qubit transition e^{-i H_tau t} for u = (+1, -1) via Rasengan synthesis."""
    qc = QuantumCircuit(2)
    theta = Parameter("t")
    # nonzero indices [0, 1]; hdi_bitstr for u=(1,-1) -> [1, 0]
    driver_component(qc, [0, 1], [], [1, 0], theta, "none")
    return qc.assign_parameters([theta_val])


def pair_channel_matrix(theta_val: float, lam_gate: float, seed: int = 7) -> np.ndarray:
    """T[b_in, b_out] for the active pair under scaled gate noise (no readout).
    b in {0:'00',1:'01',2:'10',3:'11'} with bit order (c_from, c_to)."""
    nm = make_noise_model(lam_gate, 0.0)
    sim = AerSimulator(method="density_matrix", noise_model=nm, seed_simulator=seed)
    base = transition_circuit(theta_val)
    T = np.zeros((4, 4))
    for b_in in range(4):
        qc = QuantumCircuit(2, 2)
        if b_in & 2: qc.x(0)   # bit order: q0 = c_from position
        if b_in & 1: qc.x(1)
        qc.compose(base, inplace=True)
        qc.save_density_matrix()
        tqc = transpile(qc, sim, basis_gates=CORE_BASIS + ["save_density_matrix"],
                        optimization_level=0)
        rho = sim.run(tqc).result().data(0)["density_matrix"]
        p = np.real(np.diag(np.asarray(rho)))
        # qiskit little-endian: state index bit0 = q0. Map to our (q0,q1) order.
        for idx in range(4):
            q0 = idx & 1; q1 = (idx >> 1) & 1
            T[b_in, (q0 << 1) | q1] += p[idx]
    T = np.clip(T, 0, None)
    T /= T.sum(axis=1, keepdims=True)
    return T


# ---------------------------------------------------------------------------
# Markov chain executor (exact channel sampling)
# ---------------------------------------------------------------------------

class GcpChannel:
    """Per-(segment) channels + readout, for a given noise scale lam."""

    def __init__(self, pack: GcpK3Pack, lam: float, seed: int = 7):
        self.pack = pack
        self.lam = lam
        self.p_ro = min(BASE_PRO * lam, 0.5)
        self._T = [pair_channel_matrix(pack.params[t], lam, seed=seed + t)
                   for t in range(len(pack.chain))]

    def run_segment(self, t: int, init_bits: np.ndarray, shots: int,
                    rng: np.random.Generator) -> dict[bytes, int]:
        """Sample `shots` outputs of segment t from init_bits. Returns counts
        keyed by bytes(bits) for speed."""
        u, pos_from, pos_to = self.pack.chain[t]
        i_from = u * self.pack.K + pos_from
        i_to = u * self.pack.K + pos_to
        b_in = (int(init_bits[i_from]) << 1) | int(init_bits[i_to])
        # active-pair outcomes
        outs = rng.choice(4, size=shots, p=self._T[t][b_in])
        # readout flips on all n bits
        n = self.pack.n
        flips = rng.random((shots, n)) < self.p_ro
        counts: dict[bytes, int] = {}
        base = np.tile(init_bits, (shots, 1))
        base[:, i_from] = (outs >> 1) & 1
        base[:, i_to] = outs & 1
        out = np.where(flips, 1 - base, base).astype(np.int8)
        for row in out:
            k = row.tobytes()
            counts[k] = counts.get(k, 0) + 1
        return counts
