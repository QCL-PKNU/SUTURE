"""E6 — benchmark broadening runner (numpy/pandas only, GPU-server CPU fan-out).

Grid: bench_bundle.pkl instances (DIMACS myciel3/4, queen5_5, K=4 planted,
degree sweep) x 9 lams x S=64 x 12 seeds x 6 arms. Output: bench_gcp.csv.
"""
import os, sys, time, pickle
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from remote_scale_campaign import Pack, Channel, run_chain

SEEDS = list(range(12))
ARMS = ["purify", "purify_restart", "rec_nearest", "rec_synd", "rec_uniform", "junk_synd"]


def one_cell(args):
    d, lam, S, seed = args
    pack = Pack(d)
    ch = Channel(pack, d["channels"][lam]["T"], d["channels"][lam]["p_ro"])
    rows = []
    for arm in ARMS:
        rng = np.random.default_rng(11_000_000 + hash((d["name"], lam, S, seed, arm)) % 900_000)
        r = run_chain(pack, ch, S, arm, rng)
        rows.append(dict(instance=d["name"], v=d["v"], K=d["K"], n=d["v"] * d["K"],
                         lam=lam, shots=S, seed=seed, arm=arm, **r))
    return rows


def main():
    with open("bench_bundle.pkl", "rb") as fh:
        b = pickle.load(fh)
    cells = [(d, lam, 64, seed) for d in b["instances"]
             for lam in b["lams"] for seed in SEEDS]
    print(f"{len(cells)} cells x {len(ARMS)} arms on {cpu_count()} cores", flush=True)
    t0 = time.time()
    rows = []
    with Pool(min(cpu_count(), 30)) as pool:
        for i, out in enumerate(pool.imap_unordered(one_cell, cells, chunksize=4)):
            rows.extend(out)
            if (i + 1) % 200 == 0:
                print(f"[{time.time()-t0:5.0f}s] {i+1}/{len(cells)}", flush=True)
    pd.DataFrame(rows).to_csv("bench_gcp.csv", index=False)
    print(f"wrote bench_gcp.csv ({len(rows)} rows) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
