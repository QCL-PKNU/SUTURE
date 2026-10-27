"""Score prediction P-T6 (report/06_coset_lemma.md) before any chain experiment.

Protocol: take proper colorings of gcp_v8; apply exactly ONE bit flip at a uniform
random decision position; decode with NEAREST and SYND; score exact recovery of the
planted coloring. Also score the 2-flip and 3-flip channels (beyond-lemma robustness)
and a junk control (uniform random bits through both decoders).

Lemma arithmetic for the single-flip case (K=3):
  P(flip hits the block's 1) = 1/3 -> weight-0, candidates = colors consistent w/
      neighbors; SYND recovers iff planted color drawn from list: 1/|L(u)|.
  P(flip hits a 0)          = 2/3 -> weight-2, candidates = {planted, flipped}.
      SYND: edge conflicts eliminate the wrong candidate when the flipped color
      collides with a neighbor; else 1/2.
  NEAREST: same weight-2 arithmetic WITHOUT conflict elimination (1/2 always),
      and uniform 1/K on weight-0. So the SYND-NEAREST gap concentrates in
      (a) conflict-breakable weight-2 events, (b) neighbor-restricted weight-0 lists.
"""
import sys, os
import _paths  # noqa: F401  (repo-relative sys.path setup)
import numpy as np
import dill
from itertools import product

from gcp_gonogo_lib import decode_nearest, decode_synd, decode_uniform

with open("packs/gcp_k3_packs.dill", "rb") as fh:
    packs = dill.load(fh)
pack = packs["gcp_v8"]

proper = [np.array(c) for c in product(range(3), repeat=pack.v)
          if all(c[u] != c[w] for u, w in pack.edges)]
print(f"gcp_v8: {len(proper)} proper colorings")

rng = np.random.default_rng(42)
N = 20000

def planted_trial(n_flips):
    rec = {"nearest": 0, "synd": 0}
    exact_pred = 0.0
    for i in range(N):
        col = proper[rng.integers(len(proper))]
        bits = pack.coloring_to_bits(col)
        pos = rng.choice(pack.n, size=n_flips, replace=False)
        bits2 = bits.copy()
        bits2[pos] = 1 - bits2[pos]
        for name, dec in (("nearest", decode_nearest), ("synd", decode_synd)):
            out = dec(pack, bits2, rng)
            if out is not None and np.array_equal(out, col):
                rec[name] += 1
    return {k: v / N for k, v in rec.items()}

print("\nplanted-corruption exact-recovery rate (planted coloring recovered exactly):")
for nf in (1, 2, 3):
    r = planted_trial(nf)
    ratio = r['synd'] / max(r['nearest'], 1e-9)
    print(f"  {nf} flip(s): NEAREST={r['nearest']:.3f}  SYND={r['synd']:.3f}  ratio={ratio:.2f}x")

# analytic single-flip prediction for SYND / NEAREST
tot_s = tot_n = 0.0
for col in proper:
    for u in range(pack.v):
        Lu = [c for c in range(3) if all(col[w] != c for w in pack.adj[u])]
        for posc in range(3):
            if posc == col[u]:
                # weight-0: SYND uniform over Lu; NEAREST uniform over K
                tot_s += (1.0 / len(Lu)) if col[u] in Lu else 0.0
                tot_n += 1.0 / 3
            else:
                # weight-2 {col[u], posc}: SYND eliminates posc if it conflicts
                conflict = any(col[w] == posc for w in pack.adj[u])
                tot_s += 1.0 if conflict else 0.5
                tot_n += 0.5
den = len(proper) * pack.v * 3
print(f"\nanalytic single-flip prediction: SYND={tot_s/den:.3f}  NEAREST={tot_n/den:.3f}")

# junk control: neither decoder should 'recover' anything but proper-rate ~ base
junk_proper = {"nearest": 0, "synd": 0, "uniform": 0}
M = 5000
for i in range(M):
    bits = rng.integers(0, 2, size=pack.n).astype(np.int8)
    for name, dec in (("nearest", decode_nearest), ("synd", decode_synd), ("uniform", decode_uniform)):
        out = dec(pack, bits, rng)
        if out is not None and pack.proper(out):
            junk_proper[name] += 1
print(f"\njunk-control proper rate (base rate {len(proper)/3**pack.v:.4f}): "
      + "  ".join(f"{k}={v/M:.4f}" for k, v in junk_proper.items()))