"""Remote training-budget sweep (numpy/pandas only).

Grid: entries (3 sizes x 6 budgets) x 10 lams (incl. 0) x S=64 x 16 seeds
x arms {purify, rec_synd, rec_uniform}. Output: training_sweep.csv.

rec_uniform is kept as the dumb-continuation control; junk/skyline behavior
is already established (07/08) and budget-independent.
"""
import os, sys, time, pickle
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from remote_scale_campaign import Pack, Channel, run_chain

SEEDS = list(range(16))
ARMS = ["purify", "rec_synd", "rec_uniform"]


def one_cell(args):
    d, lam, S, seed = args
    pack = Pack(d)
    ch = Channel(pack, d["channels"][lam]["T"], d["channels"][lam]["p_ro"])
    rows = []
    for arm in ARMS:
        rng = np.random.default_rng(7_000_000 + hash((d["name"], d["budget"], lam, S, seed, arm)) % 900_000)
        r = run_chain(pack, ch, S, arm, rng)
        rows.append(dict(instance=d["name"], v=d["v"], budget=d["budget"],
                         e_trained=d["e_trained"], lam=lam, shots=S, seed=seed, arm=arm,
                         survived=r["survived"], f_min=r["f_min"],
                         best_conflicts_seen=r["best_conflicts_seen"],
                         proper_seen=r["proper_seen"],
                         reg_min_conflicts=r["reg_min_conflicts"]))
    return rows


def main():
    with open("training_bundle.pkl", "rb") as fh:
        b = pickle.load(fh)
    cells = [(d, lam, 64, seed) for d in b["entries"]
             for lam in b["lams"] for seed in SEEDS]
    print(f"{len(cells)} cells x {len(ARMS)} arms on {cpu_count()} cores", flush=True)
    t0 = time.time()
    rows = []
    with Pool(min(cpu_count(), 30)) as pool:
        for i, out in enumerate(pool.imap_unordered(one_cell, cells, chunksize=8)):
            rows.extend(out)
            if (i + 1) % 500 == 0:
                print(f"[{time.time()-t0:5.0f}s] {i+1}/{len(cells)}", flush=True)
    pd.DataFrame(rows).to_csv("training_sweep.csv", index=False)
    print(f"wrote training_sweep.csv ({len(rows)} rows) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
