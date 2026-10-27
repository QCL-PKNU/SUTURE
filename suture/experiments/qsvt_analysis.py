"""E9 — Does qubitization/QSVT help the transition-Hamiltonian expansion?

Rasengan implements each transition exp(-i theta (|a><b|+|b><a|)) EXACTLY
(basis-change + multi-controlled phase; hdi_decompose.py) — there is no
Hamiltonian-simulation approximation to improve. The only place QSVT can
enter is at the LEVEL ABOVE: implement the full driver H_d = sum_tau H_tau
in one coherent block (qubitization gives optimal-query e^{-iH_d T}), or a
QSVT feasibility filter (coherent purification + amplitude amplification).

This experiment quantifies both against the segmented product, in the exact
coloring-space representation (soft-edge GCP, gcp_c5: v=5 cycle, K=3,
3^5 = 243 colorings; chain = 2 rounds x (0->1, 0->2) per vertex, uniform
theta = pi/4 — E7b showed trained angles are not load-bearing).

Part 1 (mixing quality, noise-free): segmented product  Prod_t e^{-i th H_t}
vs full-driver  e^{-i H_d T}  over T: support entropy over colorings,
P(proper), E[conflicts].
Part 2 (resource + noise model): 2q-gate counts from the E5/E8-measured
Heron numbers vs LCU-qubitization query costs; survival-fidelity estimate
exp(-p2 * G) with p2 = Heron-class median CZ error.

Output: results/qsvt_compare.csv, results/qsvt_resources.csv.
"""
import os
import numpy as np
import pandas as pd
from itertools import product
from scipy.linalg import expm

V, K = 5, 3
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
THETA = np.pi / 4
COLS = list(product(range(K), repeat=V))          # 243 colorings
IDX = {c: i for i, c in enumerate(COLS)}
N = len(COLS)

def conflicts(c):
    return sum(1 for a, b in EDGES if c[a] == c[b])

CONF = np.array([conflicts(c) for c in COLS])

def transition_H(u, cf, ct):
    """H_tau in coloring space: swaps color cf<->ct at vertex u."""
    H = np.zeros((N, N))
    for i, c in enumerate(COLS):
        if c[u] == cf:
            c2 = list(c); c2[u] = ct
            j = IDX[tuple(c2)]
            H[i, j] = H[j, i] = 1.0
    return H

CHAIN = [(u, 0, c) for _ in range(2) for u in range(V) for c in (1, 2)]
H_LIST = [transition_H(*tr) for tr in CHAIN]
H_D = sum(H_LIST)

def metrics(psi):
    p = np.abs(psi) ** 2
    p = p / p.sum()
    ent = -np.sum(p[p > 1e-15] * np.log2(p[p > 1e-15]))
    return dict(entropy=ent, p_proper=float(p[CONF == 0].sum()),
                e_conf=float((p * CONF).sum()),
                support=int((p > 1e-9).sum()))

rows = []
psi0 = np.zeros(N); psi0[IDX[(0,) * V]] = 1.0

# segmented product at quarter/half/full chain
for frac_label, n_seg in (("quarter", 5), ("half", 10), ("full", 20)):
    psi = psi0.copy()
    for t in range(n_seg):
        psi = expm(-1j * THETA * H_LIST[t]) @ psi
    rows.append(dict(family="segmented", T=n_seg, **metrics(psi)))

# full-driver evolution (what qubitization/QSVT implements optimally)
for T in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
    psi = expm(-1j * T * H_D) @ psi0
    rows.append(dict(family="qsvt_full_driver", T=T, **metrics(psi)))

df = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)

# ---- Part 2: resource model (Heron-measured numbers) ----
p2 = 3.0e-3           # Heron-class median CZ error scale
L = len(CHAIN)
g_seg_total = 8 * L   # E5/E8 median: 8 2q gates per transpiled segment
g_query = L * 24      # SELECT over L multiplexed 2-level rotations (5-bit control)
res_rows = []
for T in (1.0, 2.0):
    n_query = int(np.ceil(L * T)) + 8       # lambda*T + O(log 1/eps), lambda=L
    g_total = n_query * g_query
    res_rows.append(dict(method=f"qubitization e^(-iH_d T), T={T:g}",
                         gates_2q=g_total, ancillas=int(np.ceil(np.log2(L))) + 1,
                         survival=np.exp(-p2 * g_total)))
res_rows.append(dict(method="segmented chain (20 segments, coherent)",
                     gates_2q=g_seg_total, ancillas=2,
                     survival=np.exp(-p2 * g_seg_total)))
res_rows.append(dict(method="one segment (between measurements)",
                     gates_2q=8, ancillas=2, survival=np.exp(-p2 * 8)))
g_aa = 2 * g_seg_total + 40   # 1 fixed-point AA round: 2x chain + feasibility phase
res_rows.append(dict(method="QSVT coherent purification (+1 AA round)",
                     gates_2q=g_aa, ancillas=3, survival=np.exp(-p2 * g_aa)))
res = pd.DataFrame(res_rows)

df.to_csv("results/qsvt_compare.csv", index=False)
res.to_csv("results/qsvt_resources.csv", index=False)
print(df.round(3).to_string(index=False))
print()
print(res.round(4).to_string(index=False))