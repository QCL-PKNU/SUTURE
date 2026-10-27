"""P-T6b part (a): planted-flip identity retention, one-hot vs domain-wall.

Matched per-qubit flip rate p: each qubit of the encoded string flips i.i.d.
with prob p (one-hot: 3v qubits; domain-wall: 2v qubits). Decode with the
encoding's syndrome decoder (min-conflict list-coloring, intact blocks pinned).
Score per trial:
  exact    - full planted coloring recovered
  frac_ok  - fraction of vertices with correct color after decode
  detected - fraction of corrupted vertices whose corruption was DETECTED
             (block invalid / weight != 1) rather than silent
Graph: gcp_v8 (v=8). 20k trials per (encoding, p).
"""
import sys, os
import _paths  # noqa: F401  (repo-relative sys.path setup)
import numpy as np
import pandas as pd
from itertools import product

from gcp_dw import CODE, DECODE

EDGES = [(i, (i + 1) % 8) for i in range(8)] + [(0, 2), (4, 6)]
V = 8
ADJ = [[] for _ in range(V)]
for u, w in EDGES:
    ADJ[u].append(w); ADJ[w].append(u)
PROPER = [np.array(c) for c in product(range(3), repeat=V)
          if all(c[u] != c[w] for u, w in EDGES)]


def solve_listcolor(col, corrupt, lists, rng):
    """Min-conflict assignment of `lists` to corrupt vertices, intact pinned."""
    best = {"cost": np.inf, "assign": None, "count": 0}
    corrupt = list(corrupt)

    def bt(i, cost):
        if cost > best["cost"]: return
        if i == len(corrupt):
            if cost < best["cost"]:
                best.update(cost=cost, assign=[col[u] for u in corrupt], count=1)
            elif cost == best["cost"] and rng.random() < 1.0 / (best["count"] + 1):
                best["assign"] = [col[u] for u in corrupt]; best["count"] += 1
            return
        u = corrupt[i]
        for c in lists[i]:
            col[u] = c
            bt(i + 1, cost + sum(1 for w in ADJ[u] if col[w] == c))
            col[u] = -1
        col[u] = -1

    bt(0, 0)
    for u, c in zip(corrupt, best["assign"]):
        col[u] = c
    return col


def decode_onehot(bits, rng):
    blocks = bits.reshape(V, 3)
    wts = blocks.sum(axis=1)
    col = np.full(V, -1, dtype=int)
    detected = []
    for u in range(V):
        if wts[u] == 1:
            col[u] = int(blocks[u].argmax())
        else:
            detected.append(u)
    if not detected:
        return col, detected
    lists = []
    for u in detected:
        ones = np.flatnonzero(blocks[u])
        cand = list(ones) if len(ones) == 2 else [0, 1, 2]
        rng.shuffle(cand)
        lists.append(cand)
    return solve_listcolor(col, detected, lists, rng), detected


def decode_dw(bits, rng):
    col = np.full(V, -1, dtype=int)
    detected = []
    for u in range(V):
        b = (int(bits[2 * u]), int(bits[2 * u + 1]))
        c = DECODE[b]
        if c >= 0:
            col[u] = c
        else:
            detected.append(u)
    if not detected:
        return col, detected
    lists = []
    for u in detected:
        cand = [0, 2]   # 1-flip neighbors of 01
        rng.shuffle(cand)
        lists.append(cand)
    return solve_listcolor(col, detected, lists, rng), detected


def encode_onehot(col):
    x = np.zeros(3 * V, dtype=np.int8)
    for u, c in enumerate(col):
        x[3 * u + c] = 1
    return x


def encode_dw(col):
    x = np.zeros(2 * V, dtype=np.int8)
    for u, c in enumerate(col):
        b1, b2 = CODE[int(c)]
        x[2 * u] = b1; x[2 * u + 1] = b2
    return x


def main():
    rng = np.random.default_rng(7)
    N = 20000
    rows = []
    for p in (0.02, 0.05, 0.10, 0.20):
        for enc, encode, decode, nq in (("one-hot", encode_onehot, decode_onehot, 3 * V),
                                        ("domain-wall", encode_dw, decode_dw, 2 * V)):
            n_exact = 0; frac_ok = 0.0; det_num = 0; det_den = 0
            for _ in range(N):
                col0 = PROPER[rng.integers(len(PROPER))]
                bits = encode(col0).copy()
                flips = rng.random(nq) < p
                bits[flips] = 1 - bits[flips]
                # which vertices were corrupted (any flip in their block)
                if enc == "one-hot":
                    vcorr = {i // 3 for i in np.flatnonzero(flips)}
                else:
                    vcorr = {i // 2 for i in np.flatnonzero(flips)}
                col, detected = decode(bits, rng)
                n_exact += int(np.array_equal(col, col0))
                frac_ok += float((col == col0).mean())
                det_num += len(set(detected) & vcorr)
                det_den += len(vcorr)
            rows.append(dict(encoding=enc, p=p, qubits=nq,
                             exact=n_exact / N, frac_ok=frac_ok / N,
                             detect_rate=det_num / max(det_den, 1)))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("results/encoding_flips.csv", index=False)
    print("wrote results/encoding_flips.csv")


if __name__ == "__main__":
    main()
