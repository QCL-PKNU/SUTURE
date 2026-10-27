"""Generic linear-system syndrome repair decoder (numpy only — ships to server).

Input: measured bits x, constraint matrix C|b (rows C[:, :-1], rhs C[:, -1]).
Syndrome: violated rows V = {r : C_r x != b_r}. Suspects: variables with
nonzero coefficient in any violated row. Repair: bounded backtracking over
suspect flips (weight-ordered, <= MAX_CORRUPT) to reach C x' = b at minimum
Hamming distance from x, intact variables pinned; uniform tie-break among
minimum-weight repairs.

On one-hot GCP with assignment rows only, this reduces exactly to the T6
list-coloring decoder (violated assignment row <-> corrupted block; flips
restoring row sum 1 <-> candidate colors), verified by test_repair_generic.
"""
from __future__ import annotations
import numpy as np

MAX_CORRUPT = 6


def syndrome(C, x):
    lhs = C[:, :-1] @ x
    return np.flatnonzero(lhs != C[:, -1])


MAX_WORK = 200_000     # combination cap per weight level (vectorized rows)


def repair(C, x, rng, max_corrupt=MAX_CORRUPT):
    """Return repaired copy of x with C x = b, or None if beyond budget.

    Weight-ordered vectorized enumeration over suspect-variable flip sets:
    the first weight w with solutions is the minimum Hamming distance, and
    we tie-break uniformly among all solutions at that weight. Work is capped at MAX_WORK combinations per weight —
    exceeding the cap returns None (a documented give-up, hit essentially
    only by junk-like inputs where most rows are violated)."""
    from itertools import combinations
    from math import comb

    x = np.asarray(x, dtype=np.int8)
    viol = syndrome(C, x)
    if len(viol) == 0:
        return x.copy()
    A = C[:, :-1]
    suspects = np.flatnonzero(np.any(A[viol] != 0, axis=0))
    if len(suspects) == 0:
        return None
    resid = (C[:, -1] - A @ x).astype(np.int64)     # need A @ delta == resid
    # flip j changes x_j by sign_j: column contribution = A[:, j] * sign_j
    sign = 1 - 2 * x[suspects].astype(np.int64)
    Acol = A[:, suspects] * sign[None, :]           # rows x suspects
    ns = len(suspects)

    for w in range(1, max_corrupt + 1):
        if comb(ns, w) > MAX_WORK:
            return None
        combos = list(combinations(range(ns), w))
        if not combos:
            continue
        idx = np.array(combos, dtype=np.int64)          # (m, w)
        # sum of chosen columns for every combination: (m, rows)
        delta = Acol.T[idx].sum(axis=1)
        ok = np.flatnonzero(np.all(delta == resid[None, :], axis=1))
        if len(ok) > 0:
            pick = idx[ok[rng.integers(len(ok))] if len(ok) > 1 else ok[0]]
            out = x.copy()
            for j in pick:
                out[suspects[j]] = 1 - out[suspects[j]]
            return out
    return None
