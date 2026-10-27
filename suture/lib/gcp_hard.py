"""Hard-edge GCP (Rasengan-style): edges are IN the feasibility system.

Encoding: one-hot K=3 decision bits (3v) + one slack bit per (edge, color)
(3|E|): rows are  A_u: sum_c x_uc = 1  and  E_(u,w),c: x_uc + x_wc + s_ewc = 1.
Feasible strings <-> proper colorings (slack determined).

Transitions (u, cf, ct): Δx_u,cf=-1, Δx_u,ct=+1, and for every neighbor w:
Δs_(uw),cf=+1, Δs_(uw),ct=-1 — null-space vectors of the full system with
2+2·deg(u) nonzeros. H_τ annihilates moves that would create a conflict
(slack would leave {0,1}), so ideal dynamics stay on proper colorings.

Objective: random per-(vertex,color) integer weights W; cost(σ) = Σ_u W[u,σ(u)].
x* by enumeration (v ≤ 8).

Channel: per segment, exact output distribution from EVERY feasible-restriction
input of the active qubit set (density-matrix sim of the Rasengan-synthesized
circuit under scaled depolarizing noise), + independent readout flips on all
n bits at sample time. Register states are always feasible for purify/rec arms,
so feasible-input rows are the only ones needed.
"""
from __future__ import annotations
import sys, os
# repo-relative path setup: lib/ itself + the upstream rasengan clone
_LIB = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_LIB))
sys.path.insert(0, os.path.join(_REPO, "rasengan"))
sys.path.insert(0, _LIB)

import numpy as np
from itertools import product

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator

from rasengan.solvers.qiskit.circuit.hdi_decompose import driver_component
from h1_pilot_lib import make_noise_model, CORE_BASIS


class HardPack:
    def __init__(self, name, v, edges, rounds=2, wseed=5):
        self.name, self.v, self.K = name, v, 3
        self.edges = sorted((min(u, w), max(u, w)) for u, w in edges)
        self.E = len(self.edges)
        self.n = 3 * v + 3 * self.E
        self.adj = [[] for _ in range(v)]
        self.eidx = {}
        for i, (u, w) in enumerate(self.edges):
            self.adj[u].append(w); self.adj[w].append(u)
            self.eidx[(u, w)] = i
        rng = np.random.default_rng(wseed)
        self.W = rng.integers(1, 11, size=(v, 3))
        # proper colorings (v <= 8)
        self.proper = [np.array(c) for c in product(range(3), repeat=v)
                       if all(c[a] != c[b] for a, b in self.edges)]
        costs = np.array([self.cost(c) for c in self.proper])
        self.i_star = int(np.argmin(costs))
        self.x_star = tuple(int(x) for x in self.proper[self.i_star])
        self.e_star = float(costs.min())
        self.e_med = float(np.median(costs))
        # chain: rounds x vertices x 2 swaps
        self.chain = []
        for _ in range(rounds):
            for u in range(v):
                self.chain.append((u, 0, 1))
                self.chain.append((u, 0, 2))
        self.params = np.full(len(self.chain), np.pi / 4)
        # seed = first proper coloring (lexicographic)
        self.seed_coloring = self.proper[0]

    def cost(self, col):
        return float(sum(self.W[u, col[u]] for u in range(self.v)))

    # ---- encoding ----
    def coloring_to_bits(self, col):
        x = np.zeros(self.n, dtype=np.int8)
        for u, c in enumerate(col):
            x[3 * u + int(c)] = 1
        for i, (a, b) in enumerate(self.edges):
            for c in range(3):
                s = 1 - x[3 * a + c] - x[3 * b + c]
                x[3 * self.v + 3 * i + c] = s
        return x

    def is_feasible_bits(self, bits):
        dec = bits[:3 * self.v].reshape(self.v, 3)
        if not (dec.sum(axis=1) == 1).all():
            return False
        col = dec.argmax(axis=1)
        if any(col[a] == col[b] for a, b in self.edges):
            return False
        # slack consistency
        for i, (a, b) in enumerate(self.edges):
            for c in range(3):
                if bits[3 * self.v + 3 * i + c] != 1 - bits[3 * a + c] - bits[3 * b + c]:
                    return False
        return True

    def decode_coloring(self, bits):
        return bits[:3 * self.v].reshape(self.v, 3).argmax(axis=1)

    # ---- transition vector / active set ----
    def transition_vector(self, u, cf, ct):
        vec = np.zeros(self.n, dtype=int)
        vec[3 * u + cf] = -1
        vec[3 * u + ct] = +1
        for w in self.adj[u]:
            i = self.eidx[(min(u, w), max(u, w))]
            vec[3 * self.v + 3 * i + cf] = +1
            vec[3 * self.v + 3 * i + ct] = -1
        return vec

    def active_set(self, u, cf, ct):
        return np.flatnonzero(self.transition_vector(u, cf, ct))


class AnalyticHardChannel:
    """Noise-free: swap u cf<->ct iff target color not used by any neighbor."""
    def __init__(self, pack):
        self.pack = pack; self.p_ro = 0.0

    def step_register(self, t, register, shots, rng):
        u, cf, ct = self.pack.chain[t]
        th = self.pack.params[t]
        c2, s2 = np.cos(th) ** 2, np.sin(th) ** 2
        out = {}
        for col_t, w in register.items():
            col = list(col_t)
            tgt = None
            if col[u] == cf and all(col[x] != ct for x in self.pack.adj[u]):
                tgt = ct
            elif col[u] == ct and all(col[x] != cf for x in self.pack.adj[u]):
                tgt = cf
            if tgt is None:
                out[col_t] = out.get(col_t, 0) + w
            else:
                nsw = rng.binomial(w, s2)
                col2 = list(col); col2[u] = tgt
                out[col_t] = out.get(col_t, 0) + (w - nsw)
                k2 = tuple(col2)
                out[k2] = out.get(k2, 0) + nsw
        return {k: v for k, v in out.items() if v > 0}


def build_hard_channels(pack, lams, verbose=True):
    """{lam: {(t, r_bytes): (out_codes int64[], probs)}} over active-set restrictions
    of all feasible strings; + p_ro per lam."""
    import time as _t
    F_bits = [pack.coloring_to_bits(c) for c in pack.proper]
    channels = {lam: {} for lam in lams}
    for t, (u, cf, ct) in enumerate(pack.chain):
        t0 = _t.time()
        A = pack.active_set(u, cf, ct)
        m = len(A)
        vec = pack.transition_vector(u, cf, ct)[A]
        # synthesize circuit on m + 2 qubits
        theta = Parameter("t")
        qc0 = QuantumCircuit(m + 2)
        hdi = [0 if x == -1 else 1 for x in vec]
        driver_component(qc0, list(range(m)), [m, m + 1], hdi, theta, "constant")
        bound = qc0.assign_parameters([pack.params[t]])
        # distinct feasible restrictions
        restr = {}
        for fb in F_bits:
            r = bytes(fb[A].tolist())
            restr[r] = fb[A].copy()
        for lam in lams:
            nm = make_noise_model(lam, 0.0)
            sim = AerSimulator(method="density_matrix", noise_model=nm,
                               seed_simulator=101 + t)
            for r, rbits in restr.items():
                qc = QuantumCircuit(m + 2)
                for i, b in enumerate(rbits):
                    if b: qc.x(i)
                qc.compose(bound, inplace=True)
                qc.save_density_matrix(qubits=list(range(m)))
                tqc = transpile(qc, sim, basis_gates=CORE_BASIS + ["save_density_matrix"],
                                optimization_level=0)
                rho = np.asarray(sim.run(tqc).result().data(0)["density_matrix"])
                p = np.clip(np.real(np.diag(rho)), 0, None)
                p /= p.sum()
                nz = np.flatnonzero(p > 1e-9)
                # qiskit little-endian index -> our bit order (qubit i = A[i])
                codes = []
                for idx in nz:
                    bits = [(idx >> i) & 1 for i in range(m)]
                    code = 0
                    for i, b in enumerate(bits):
                        code |= b << i
                    codes.append(code)
                channels[lam][(t, r)] = (np.array(codes, dtype=np.int64), p[nz] / p[nz].sum())
        if verbose:
            print(f"  seg {t} |A|={m} restr={len(restr)} ({_t.time()-t0:.1f}s)", flush=True)
    return channels
