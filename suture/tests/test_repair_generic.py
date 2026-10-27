"""Gate tests for repair_generic (must pass before the campaign dispatches).

(a) one-hot K=3 soft-edge: on 1k planted single flips, generic repair's
    exact-recovery rate matches the T6 analytic arithmetic for coset-uniform
    repair WITHOUT edge information (assignment rows only => weight-2 cosets
    are 50/50, weight-0 uniform over K): predicted 0.444 (T6, NEAREST bound).
    With edge rows added as soft scoring it should approach decode_synd —
    here we test the pure linear-system contract (assignment rows only) and
    separately the K=V model where edge rows ARE constraint rows.
(b) K=V paper instance (edges in C via slack): planted 1-2 flips repaired
    back into F; single-flip exact recovery >= 0.9 (syndrome localizes the
    block AND edge rows prune wrong candidates, mirroring hard-edge SYND).
(c) junk control: uniform random bits -> fraction repaired into F documents
    the solver-contamination rate (must be reported, not hidden).

Each gate below is an independently seeded pytest test, so a failure names the
gate that regressed instead of aborting collection of the whole module.
"""
import os
import random
import sys
from itertools import product

import numpy as np
import pytest

from repair_generic import repair, syndrome

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))

# ---------- (a) soft-edge one-hot K=3, assignment rows only ----------
V, K = 8, 3
EDGES = [(i, (i + 1) % 8) for i in range(8)] + [(0, 2), (4, 6)]
C_assign = np.zeros((V, V * K + 1), dtype=np.int64)
for u in range(V):
    C_assign[u, u * K:(u + 1) * K] = 1
    C_assign[u, -1] = 1
PROPER = [c for c in product(range(K), repeat=V)
          if all(c[a] != c[b] for a, b in EDGES)]


def col_bits(col):
    x = np.zeros(V * K, dtype=np.int8)
    for u, c in enumerate(col):
        x[u * K + c] = 1
    return x


def test_softedge_onehot_single_flip_matches_t6_prediction():
    """T6 coset-uniform predicts 0.444 without edge information."""
    rng = np.random.default_rng(11)
    hits = 0
    N = 1000
    for _ in range(N):
        col = PROPER[rng.integers(len(PROPER))]
        x = col_bits(col)
        j = rng.integers(V * K)
        x[j] = 1 - x[j]
        out = repair(C_assign, x, rng)
        assert out is not None and len(syndrome(C_assign, out)) == 0
        hits += int(np.array_equal(out, col_bits(col)))
    rate = hits / N
    print(f"(a) soft one-hot single-flip exact recovery: {rate:.3f} "
          f"(T6 coset-uniform predicts 0.444)")
    assert abs(rate - 0.444) < 0.05, rate


# ---------- (b) K=V paper instance ----------
@pytest.fixture(scope="module")
def kv_instance():
    """Generated K=V instance with edge rows in C; seeded for a fixed instance."""
    np.random.seed(0xdb)
    random.seed(0x7f)
    from rasengan.problems.graph_coloring_problem import generate_gcp
    import prep_bench10 as pb
    probs, _ = generate_gcp(1, [(4, 4)], 1, 20)
    prb = probs[0][0]
    C = np.asarray(prb.lin_constr_mtx, dtype=np.int64)
    F = pb.enumerate_F_kv(prb, prb.num_graphs, prb.num_colors)
    print(f"(b) kv instance: n={C.shape[1] - 1} |F|={len(F)} rows={C.shape[0]}")
    return C, F, {fb.tobytes() for fb in F}


@pytest.mark.parametrize("nflip,min_rate", [(1, 0.90), (2, 0.55)])
def test_kv_planted_flips_recover_original(kv_instance, nflip, min_rate):
    """Exact recovery = repaired back to the ORIGINAL planted bitstring."""
    C, F, _ = kv_instance
    rng = np.random.default_rng(11)
    exact = unrepaired = 0
    N = 600
    for _ in range(N):
        orig = F[rng.integers(len(F))]
        x = orig.copy()
        for j in rng.choice(len(x), size=nflip, replace=False):
            x[j] = 1 - x[j]
        out = repair(C, x, rng)
        if out is None or len(syndrome(C, out)) > 0:
            unrepaired += 1
            continue
        exact += int(np.array_equal(out, orig))
    print(f"    {nflip}-flip: exact-recovery {exact / N:.3f} (gate >= {min_rate}), "
          f"unrepaired {unrepaired / N:.3f}")
    assert exact / N >= min_rate, (nflip, exact / N)


# ---------- (c) junk control ----------
def test_junk_control_contamination_stays_negligible(kv_instance):
    """Uniform random bits must not let the decoder fabricate feasible states.

    The measured rate is the solver-contamination figure reported alongside the
    campaign endpoints; the bound is a regression guard, not a tuned threshold.
    """
    C, _, Fset = kv_instance
    rng = np.random.default_rng(11)
    junk_in_F = junk_repaired = 0
    N = 1000
    for _ in range(N):
        x = rng.integers(0, 2, size=C.shape[1] - 1).astype(np.int8)
        out = repair(C, x, rng)
        if out is not None and len(syndrome(C, out)) == 0:
            junk_repaired += 1
            junk_in_F += int(out.tobytes() in Fset)
    print(f"(c) junk control: repaired-to-feasible rate {junk_repaired / N:.3f} "
          f"(solver contamination — report, never use register endpoints)")
    assert junk_repaired / N < 0.05, junk_repaired / N
