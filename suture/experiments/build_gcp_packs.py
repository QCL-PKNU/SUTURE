"""Build + train the fixed-K GCP packs for the go/no-go run.

Instances:
  gcp_c5    cycle C5, v=5, K=3, n=15          (calibration; 30 proper colorings)
  gcp_v8    C8 + chords (0,2),(4,6), v=8, K=3, n=24 (go/no-go; chi=3)

Chain: R rounds over vertices, transitions (c0<->c1) and (c0<->c2) per vertex
(within-block swaps = null-space vectors e_{u,c} - e_{u,c'}; 2-qubit circuits).
Seed: all-color-0 (feasible one-hot, maximally improper) so training must move.
Training: COBYLA on the analytic noise-free channel (cos^2/sin^2 mixing),
objective = expected conflicts of the final register at 512 shots.
"""
import sys, os, time
import _paths  # noqa: F401  (repo-relative sys.path setup)

import numpy as np
import dill
from scipy.optimize import minimize

from gcp_k3 import GcpK3Pack, num_proper_colorings
from gcp_gonogo_lib import run_chain, GcpRunResult


class AnalyticChannel:
    """Noise-free segment channel: pair mixing cos^2/sin^2, no readout error."""

    def __init__(self, pack):
        self.pack = pack
        self.p_ro = 0.0

    def run_segment(self, t, init_bits, shots, rng):
        u, pf, pt = self.pack.chain[t]
        i_from, i_to = u * self.pack.K + pf, u * self.pack.K + pt
        b_in = (int(init_bits[i_from]) << 1) | int(init_bits[i_to])
        th = self.pack.params[t]
        c2, s2 = np.cos(th) ** 2, np.sin(th) ** 2
        out = init_bits.copy()
        counts = {}
        if b_in in (1, 2):  # |01> or |10>: mixes
            n_swap = rng.binomial(shots, s2)
            stay = out.tobytes()
            sw = out.copy()
            sw[i_from], sw[i_to] = out[i_to], out[i_from]
            counts[stay] = shots - n_swap
            counts[sw.tobytes()] = n_swap
        else:               # |00>, |11>: invariant
            counts[out.tobytes()] = shots
        return {k: v for k, v in counts.items() if v > 0}


def build_pack(name, v, K, edges, rounds):
    chain = []
    for _ in range(rounds):
        for u in range(v):
            chain.append((u, 0, 1))
            chain.append((u, 0, 2))
    seed = np.zeros(v, dtype=np.int64)
    pack = GcpK3Pack(name=name, v=v, K=K, edges=edges, chain=chain,
                     params=np.full(len(chain), np.pi / 4), seed_coloring=seed)

    ach = AnalyticChannel(pack)

    def objective(params):
        pack.params = np.asarray(params)
        rng = np.random.default_rng(11)
        r = run_chain(pack, ach, 512, "purify", rng)
        reg = r.final_register
        tot = sum(reg.values())
        if tot == 0:
            return 1e9
        return sum(pack.conflicts(k) * w for k, w in reg.items()) / tot

    x0 = np.full(len(chain), np.pi / 4)
    t0 = time.time()
    res = minimize(objective, x0, method="COBYLA", options={"maxiter": 250, "rhobeg": 0.5})
    e_opt, e_init = objective(res.x), objective(x0)
    pack.params = res.x if e_opt <= e_init else x0
    n_proper = num_proper_colorings(edges, v, K)
    final = objective(pack.params)
    print(f"[{name}] v={v} n={v*K} |edges|={len(edges)} chain={len(chain)} "
          f"proper={n_proper}/{K**v} E[conflicts] init={e_init:.2f} trained={final:.2f} "
          f"({time.time()-t0:.0f}s)")
    return pack


def main():
    packs = {}
    packs["gcp_c5"] = build_pack("gcp_c5", 5, 3, [(0,1),(1,2),(2,3),(3,4),(4,0)], rounds=2)
    edges_v8 = [(i, (i+1) % 8) for i in range(8)] + [(0, 2), (4, 6)]
    packs["gcp_v8"] = build_pack("gcp_v8", 8, 3, edges_v8, rounds=2)
    os.makedirs("packs", exist_ok=True)
    with open("packs/gcp_k3_packs.dill", "wb") as fh:
        dill.dump(packs, fh)
    print("saved packs/gcp_k3_packs.dill")


if __name__ == "__main__":
    main()
