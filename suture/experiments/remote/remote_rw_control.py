"""E7b — random-walk-register repair control (review O3's in-between arm).

Replaces the TRAINED quantum pair channel with a classical unbiased random
walk on the feasible graph: the active pair (c_from, c_to) swaps 10<->01 with
p = 1/2; 00/11 are fixed points (same support as the transition Hamiltonian,
no training bias, no gate noise). Corruption comes only from the same
lambda-scaled readout flips.

Arms (on the rw channel): purify_rw, rec_synd_rw. Compared offline against
rec_synd / purify on the true trained channel (scale_gcp.csv / training_sweep.csv).

NOTE (interpretation, from the O1 discussion): at product-register sizes the
trained channel IS a classical Markov chain, so this control isolates exactly
two ingredients — (a) the trained swap biases, (b) gate-noise corruption
inside the channel. If rec_synd_rw ~= rec_synd, generic legal dynamics +
corruption + repair suffice at these sizes (honest O1 framing); if it loses,
the trained chain carries weight even classically.

Grid: scale_bundle gcp_p{12,16,24}_s0 x lams x S=64 x 16 seeds.
Output: rw_control.csv.
"""
import os, sys, time, pickle
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from remote_scale_campaign import Pack, Channel, run_chain

SEEDS = list(range(16))
WANT = ("gcp_p12_s0", "gcp_p16_s0", "gcp_p24_s0")

T_RW = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.5, 0.0],
    [0.0, 0.5, 0.5, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])


def one_cell(args):
    d, lam, S, seed = args
    pack = Pack(d)
    n_seg = len(d["chain"])
    T = np.stack([T_RW] * n_seg)
    p_ro = d["channels"][lam]["p_ro"]  # same readout corruption as the real channel
    ch = Channel(pack, T, p_ro)
    rows = []
    for arm in ("purify", "rec_synd"):
        rng = np.random.default_rng(13_000_000 + hash((d["name"], lam, S, seed, arm)) % 900_000)
        r = run_chain(pack, ch, S, arm, rng)
        rows.append(dict(instance=d["name"], v=d["v"], lam=lam, shots=S, seed=seed,
                         arm=arm + "_rw", **r))
    return rows


def main():
    with open("scale_bundle.pkl", "rb") as fh:
        b = pickle.load(fh)
    insts = [d for d in b["instances"] if d["name"] in WANT]
    cells = [(d, lam, 64, seed) for d in insts for lam in b["lams"] for seed in SEEDS]
    print(f"{len(cells)} cells on {cpu_count()} cores", flush=True)
    t0 = time.time()
    rows = []
    with Pool(min(cpu_count(), 30)) as pool:
        for out in pool.imap_unordered(one_cell, cells, chunksize=8):
            rows.extend(out)
    pd.DataFrame(rows).to_csv("rw_control.csv", index=False)
    print(f"wrote rw_control.csv ({len(rows)} rows) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
