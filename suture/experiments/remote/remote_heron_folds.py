"""E8 — Heron-calibrated fold campaign (numpy/pandas, GPU-server CPU fan-out).

Channel: factorized fake-Heron pair channels (heron_channel_bundle.pkl) with
PER-QUBIT spectator readout flips (p_ro_vec) — the calibration-noise variant
of remote_scale_campaign.Channel (which assumed a uniform p_ro).

Part 1 (validation): fold-1 purify runs on gcp_c5/gcp_v8 -> per-segment f_t
  + block-locality of the infeasible mass, to compare against E5's
  whole-register ground truth offline.
Part 2 (win-rate): arms {purify, purify_restart, rec_nearest, rec_synd,
  rec_uniform, junk_synd} x folds {1,3,5,7} x 4 instances (15/24/36/48q)
  x S=64 x 16 seeds.

Output: heron_folds.csv, heron_fold1_ft.csv, heron_fold1_dd.csv.
"""
import os, sys, time, pickle
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from remote_scale_campaign import Pack, run_chain, decode_synd

SEEDS = list(range(16))
ARMS = ["purify", "purify_restart", "rec_nearest", "rec_synd", "rec_uniform", "junk_synd"]


class HeronChannel:
    """Pair channel + per-qubit readout flips (calibration-derived).
    p_vec has shape (n_seg, n): per-segment placement -> per-segment readout."""

    def __init__(self, pack, T, p_vec):
        self.pack, self._T, self.p_vec = pack, T, np.asarray(p_vec)

    def run_segment(self, t, init_bits, shots, rng):
        u, pf, pt = self.pack.chain[t]
        i_from, i_to = u * self.pack.K + pf, u * self.pack.K + pt
        b_in = (int(init_bits[i_from]) << 1) | int(init_bits[i_to])
        outs = rng.choice(4, size=shots, p=self._T[t][b_in])
        flips = rng.random((shots, self.pack.n)) < self.p_vec[t][None, :]
        # active pair's readout already inside T: mask their spectator flips
        flips[:, i_from] = False
        flips[:, i_to] = False
        base = np.tile(init_bits, (shots, 1))
        base[:, i_from] = (outs >> 1) & 1
        base[:, i_to] = outs & 1
        out = np.where(flips, 1 - base, base).astype(np.int8)
        counts = {}
        for row in out:
            k = row.tobytes()
            counts[k] = counts.get(k, 0) + 1
        return counts


def _packable(d):
    """heron_channel_bundle entries omit 'params' (angles are baked into T)."""
    d = dict(d)
    d.setdefault("params", None)
    return d


def fold1_validation(d):
    """purify-semantics f_t + infeasible block profile at fold 1 (matches E5 parts A/B)."""
    pack = Pack(_packable(d))
    ch = HeronChannel(pack, d["T"][1], d["p_ro_vec"])
    rng = np.random.default_rng(5)
    ft_rows, dd_rows = [], []
    register = {pack.coloring_to_bits(d["seed_coloring"]).tobytes(): 256}
    for t in range(len(pack.chain)):
        raw = {}
        for kb, w in register.items():
            for k2, v in ch.run_segment(t, np.frombuffer(kb, dtype=np.int8), int(w), rng).items():
                raw[k2] = raw.get(k2, 0) + v
        tot = sum(raw.values())
        feas = {}
        for kb, w in raw.items():
            bits = np.frombuffer(kb, dtype=np.int8)
            blocks = bits.reshape(pack.v, pack.K)
            weights = blocks.sum(axis=1)
            bad = int((weights != 1).sum())
            if bad == 0:
                feas[kb] = w
            else:
                dd_rows.append(dict(instance=pack.name, seg=t, w=w, n_bad_blocks=bad,
                                    w0_blocks=int((weights == 0).sum()),
                                    w2p_blocks=int((weights >= 2).sum())))
        ft_rows.append(dict(instance=pack.name, seg=t, f=sum(feas.values()) / max(tot, 1)))
        if not feas:
            break
        keys = list(feas)
        p = np.array([feas[k] for k in keys], dtype=float)
        draw = rng.multinomial(256, p / p.sum())
        register = {k: int(x) for k, x in zip(keys, draw) if x > 0}
    return pd.DataFrame(ft_rows), pd.DataFrame(dd_rows)


def one_cell(args):
    d, fold, S, seed = args
    pack = Pack(_packable(d))
    ch = HeronChannel(pack, d["T"][fold], d["p_ro_vec"])
    rows = []
    for arm in ARMS:
        rng = np.random.default_rng(17_000_000 + hash((d["name"], fold, S, seed, arm)) % 900_000)
        r = run_chain(pack, ch, S, arm, rng)
        rows.append(dict(instance=d["name"], v=d["v"], n=d["v"] * d["K"], fold=fold,
                         shots=S, seed=seed, arm=arm, **r))
    return rows


def main():
    with open("heron_channel_bundle.pkl", "rb") as fh:
        b = pickle.load(fh)
    # part 1
    fts, dds = [], []
    for d in b["instances"]:
        if d["name"] in ("gcp_c5", "gcp_v8"):
            ft, dd = fold1_validation(d)
            fts.append(ft); dds.append(dd)
            print(f"fold1 validation {d['name']} done", flush=True)
    pd.concat(fts).to_csv("heron_fold1_ft.csv", index=False)
    pd.concat(dds).to_csv("heron_fold1_dd.csv", index=False)
    # part 2
    cells = [(d, fold, 64, seed) for d in b["instances"]
             for fold in b["folds"] for seed in SEEDS]
    print(f"{len(cells)} cells x {len(ARMS)} arms on {cpu_count()} cores", flush=True)
    t0 = time.time()
    rows = []
    with Pool(min(cpu_count(), 30)) as pool:
        for i, out in enumerate(pool.imap_unordered(one_cell, cells, chunksize=4)):
            rows.extend(out)
            if (i + 1) % 100 == 0:
                print(f"[{time.time()-t0:5.0f}s] {i+1}/{len(cells)}", flush=True)
                pd.DataFrame(rows).to_csv("heron_folds.partial.csv", index=False)
    pd.DataFrame(rows).to_csv("heron_folds.csv", index=False)
    print(f"wrote heron_folds.csv ({len(rows)} rows) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
