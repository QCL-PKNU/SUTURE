"""GCP go/no-go: syndrome repair vs dumb continuation vs purification variants.

Arms: purify, purify_restart, purify_last, rec_nearest, rec_synd, rec_uniform,
      junk_synd, feas_control.
Grid: 2 instances x lam {1,2,4,8,12,16,24,32,48,64} x S {64, 256} x 12 seeds.
Endpoints: measurement-level (proper_seen, best_conflicts_seen) and
           register-level (reg_proper_rate, reg_min/mean_conflicts), survival,
           restarts, f_min/f_last.
"""
import sys, os, time
import _paths  # noqa: F401  (repo-relative sys.path setup)

import numpy as np
import pandas as pd
import dill

from gcp_k3 import GcpChannel
from gcp_gonogo_lib import run_chain, endpoint_row, feasible_skyline

LAMS = [1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0]
SHOTS = [64, 256]
SEEDS = list(range(12))
ARMS = ["purify", "purify_restart", "purify_last",
        "rec_nearest", "rec_synd", "rec_uniform", "junk_synd"]


def main():
    with open("packs/gcp_k3_packs.dill", "rb") as fh:
        packs = dill.load(fh)
    rows = []
    t00 = time.time()
    for name, pack in packs.items():
        for lam in LAMS:
            ch = GcpChannel(pack, lam=lam)
            for S in SHOTS:
                for seed in SEEDS:
                    for arm in ARMS:
                        rng = np.random.default_rng(1_000_000 + hash((name, lam, S, seed, arm)) % 500_000)
                        r = run_chain(pack, ch, S, arm, rng)
                        rows.append(dict(instance=name, lam=lam, shots=S, seed=seed,
                                         arm=arm, n_segments=len(pack.chain),
                                         **endpoint_row(pack, r)))
                    rng = np.random.default_rng(2_000_000 + hash((name, lam, S, seed)) % 500_000)
                    rows.append(dict(instance=name, lam=lam, shots=S, seed=seed,
                                     arm="feas_control", n_segments=len(pack.chain),
                                     **feasible_skyline(pack, S * len(pack.chain), rng)))
            print(f"[{time.time()-t00:6.0f}s] {name} lam={lam:g} done", flush=True)
        pd.DataFrame(rows).to_csv("results/gonogo_gcp_partial.csv", index=False)
    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/gonogo_gcp.csv", index=False)
    print("wrote results/gonogo_gcp.csv", df.shape)


if __name__ == "__main__":
    main()
