"""Remote scale campaign: pure numpy/pandas chain sampler (no qiskit needed).

Self-contained copy of the go/no-go executor (gcp_gonogo_lib.py) driven by
precomputed pair-channel T-matrices from scale_bundle.pkl. Grid:
  instances (16) x lam (9) x S {64,256} x 12 noise seeds x 8 arms,
multiprocessing over all cores. Output: scale_gcp.csv.
"""
import os, sys, time, pickle
from itertools import product
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

MAX_CORRUPT = 6
MAX_REGISTER = 64
LAM_GRID = None  # filled from bundle
SHOTS = [64, 256]
SEEDS = list(range(12))
ARMS = ["purify", "purify_restart", "purify_last",
        "rec_nearest", "rec_synd", "rec_uniform", "junk_synd"]


class Pack:
    def __init__(self, d):
        self.name, self.v, self.K = d["name"], d["v"], d["K"]
        self.edges = d["edges"]; self.chain = d["chain"]
        self.params = d["params"]; self.seed_coloring = d["seed_coloring"]
        self.n = self.v * self.K
        self.adj = [[] for _ in range(self.v)]
        for u, w in self.edges:
            self.adj[u].append(w); self.adj[w].append(u)

    def coloring_to_bits(self, col):
        x = np.zeros(self.n, dtype=np.int8)
        for u, c in enumerate(col):
            x[u * self.K + int(c)] = 1
        return x

    def bits_to_blocks(self, bits):
        return bits.reshape(self.v, self.K)

    def is_feasible_bits(self, bits):
        return bool((self.bits_to_blocks(bits).sum(axis=1) == 1).all())

    def decode_coloring(self, bits):
        return self.bits_to_blocks(bits).argmax(axis=1)

    def conflicts(self, col):
        return sum(1 for u, w in self.edges if col[u] == col[w])


class Channel:
    def __init__(self, pack, T, p_ro):
        self.pack, self._T, self.p_ro = pack, T, p_ro

    def run_segment(self, t, init_bits, shots, rng):
        u, pf, pt = self.pack.chain[t]
        i_from, i_to = u * self.pack.K + pf, u * self.pack.K + pt
        b_in = (int(init_bits[i_from]) << 1) | int(init_bits[i_to])
        outs = rng.choice(4, size=shots, p=self._T[t][b_in])
        flips = rng.random((shots, self.pack.n)) < self.p_ro
        base = np.tile(init_bits, (shots, 1))
        base[:, i_from] = (outs >> 1) & 1
        base[:, i_to] = outs & 1
        out = np.where(flips, 1 - base, base).astype(np.int8)
        counts = {}
        for row in out:
            k = row.tobytes()
            counts[k] = counts.get(k, 0) + 1
        return counts


# ---- decoders (verbatim logic from gcp_gonogo_lib) ----

def decode_nearest(pack, bits, rng):
    blocks = pack.bits_to_blocks(bits)
    col = np.empty(pack.v, dtype=np.int64)
    for u in range(pack.v):
        ones = np.flatnonzero(blocks[u])
        if len(ones) == 1: col[u] = ones[0]
        elif len(ones) == 2: col[u] = ones[rng.integers(2)]
        else: col[u] = rng.integers(pack.K)
    return col


def decode_synd(pack, bits, rng):
    blocks = pack.bits_to_blocks(bits)
    weights = blocks.sum(axis=1)
    corrupt = np.flatnonzero(weights != 1)
    col = np.full(pack.v, -1, dtype=np.int64)
    for u in np.flatnonzero(weights == 1):
        col[u] = int(blocks[u].argmax())
    if len(corrupt) == 0: return col
    if len(corrupt) > MAX_CORRUPT: return None
    lists = []
    for u in corrupt:
        ones = np.flatnonzero(blocks[u])
        cand = list(ones) if len(ones) == 2 else list(range(pack.K))
        rng.shuffle(cand)
        lists.append(cand)
    corrupt_l = list(corrupt)
    best = {"cost": np.inf, "assign": None, "count": 0}

    def edge_cost(u, c):
        return sum(1 for w in pack.adj[u] if col[w] == c)

    def bt(i, cost):
        if cost > best["cost"]: return
        if i == len(corrupt_l):
            if cost < best["cost"]:
                best["cost"], best["assign"], best["count"] = cost, [col[u] for u in corrupt_l], 1
            elif cost == best["cost"] and rng.random() < 1.0 / (best["count"] + 1):
                best["assign"] = [col[u] for u in corrupt_l]; best["count"] += 1
            return
        u = corrupt_l[i]
        for c in lists[i]:
            col[u] = c
            bt(i + 1, cost + edge_cost(u, c))
            col[u] = -1

    bt(0, 0)
    for u, c in zip(corrupt_l, best["assign"]):
        col[u] = c
    return col


def decode_uniform(pack, bits, rng):
    return rng.integers(0, pack.K, size=pack.v)


DEC = {"rec_nearest": decode_nearest, "rec_uniform": decode_uniform,
       "rec_synd": decode_synd, "junk_synd": decode_synd}


def run_chain(pack, channel, shots, arm, rng):
    seed_col = tuple(int(c) for c in pack.seed_coloring)
    register = {seed_col: shots}
    junk = arm == "junk_synd"
    f_per, death, restarts = [], -1, 0
    best_seen, proper_seen = np.inf, False

    for t in range(len(pack.chain)):
        states = list(register.keys())
        wts = np.array([register[s] for s in states], dtype=float)
        alloc = rng.multinomial(shots, wts / wts.sum())
        raw = {}
        for col_t, a in zip(states, alloc):
            if a == 0: continue
            if junk:
                X = rng.integers(0, 2, size=(int(a), pack.n), dtype=np.int8)
                for row in X:
                    k = row.tobytes(); raw[k] = raw.get(k, 0) + 1
            else:
                init_bits = pack.coloring_to_bits(col_t)
                for k, cnum in channel.run_segment(t, init_bits, int(a), rng).items():
                    raw[k] = raw.get(k, 0) + cnum
        tot = sum(raw.values()); n_feas = 0
        new_reg = {}
        for k, cnum in raw.items():
            bits = np.frombuffer(k, dtype=np.int8)
            if pack.is_feasible_bits(bits):
                n_feas += cnum
                col = pack.decode_coloring(bits)
                nc = pack.conflicts(col)
                if nc < best_seen: best_seen = nc
                if nc == 0: proper_seen = True
                kk = tuple(int(x) for x in col)
                new_reg[kk] = new_reg.get(kk, 0) + cnum
            elif arm in DEC:
                col = DEC[arm](pack, bits, rng)
                if col is None: continue
                kk = tuple(int(x) for x in col)
                new_reg[kk] = new_reg.get(kk, 0) + cnum
        f_per.append(n_feas / tot if tot else 0.0)

        if not new_reg:
            if arm == "purify_restart":
                new_reg = {seed_col: shots}; restarts += 1
            elif arm == "purify_last":
                new_reg = dict(register); restarts += 1
            else:
                death = t; break
        if len(new_reg) > MAX_REGISTER:
            new_reg = dict(sorted(new_reg.items(), key=lambda kv: -kv[1])[:MAX_REGISTER])
        register = new_reg

    row = dict(survived=death < 0, death_segment=death, restarts=restarts,
               f_min=min(f_per) if f_per else np.nan,
               f_last=f_per[-1] if f_per else np.nan,
               best_conflicts_seen=best_seen, proper_seen=proper_seen)
    tot = sum(register.values())
    if tot > 0 and death < 0:
        confs = np.array([pack.conflicts(k) for k in register])
        wts = np.array([register[k] for k in register], dtype=float)
        row["reg_min_conflicts"] = float(confs.min())
        row["reg_mean_conflicts"] = float((confs * wts).sum() / wts.sum())
        row["reg_proper_rate"] = float(wts[confs == 0].sum() / wts.sum())
    else:
        row["reg_min_conflicts"] = np.nan
        row["reg_mean_conflicts"] = np.nan
        row["reg_proper_rate"] = 0.0
    return row


def one_cell(args):
    inst_d, lam, S, seed = args
    pack = Pack(inst_d)
    ch = Channel(pack, inst_d["channels"][lam]["T"], inst_d["channels"][lam]["p_ro"])
    rows = []
    for arm in ARMS:
        rng = np.random.default_rng(9_000_000 + hash((pack.name, lam, S, seed, arm)) % 900_000)
        row = run_chain(pack, ch, S, arm, rng)
        rows.append(dict(instance=pack.name, v=pack.v, n=pack.n, lam=lam, shots=S,
                         seed=seed, arm=arm, n_segments=len(pack.chain), **row))
    # feasible-uniform skyline
    rng = np.random.default_rng(9_500_000 + hash((pack.name, lam, S, seed)) % 900_000)
    budget = S * len(pack.chain)
    cols = rng.integers(0, pack.K, size=(budget, pack.v))
    confs = np.array([pack.conflicts(c) for c in cols])
    rows.append(dict(instance=pack.name, v=pack.v, n=pack.n, lam=lam, shots=S,
                     seed=seed, arm="feas_control", n_segments=len(pack.chain),
                     survived=True, death_segment=-1, restarts=0, f_min=1.0, f_last=1.0,
                     best_conflicts_seen=float(confs.min()),
                     proper_seen=bool((confs == 0).any()),
                     reg_min_conflicts=float(confs.min()),
                     reg_mean_conflicts=float(confs.mean()),
                     reg_proper_rate=float((confs == 0).mean())))
    return rows


def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parents[2] / 'packs/scale_bundle.pkl')
    parser.add_argument('--output', type=Path, default=Path('scale_gcp.csv'))
    parser.add_argument('--smoke', action='store_true', help='One cell, first instance, two segments')
    parser.add_argument('--workers', type=int, default=min(cpu_count(), 30))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    with args.bundle.open("rb") as fh:
        bundle = pickle.load(fh)
    lams = bundle["lams"]
    cells = [(inst, lam, S, seed)
             for inst in bundle["instances"]
             for lam in lams for S in SHOTS for seed in SEEDS]
    if args.smoke:
        instance, lam, shots, seed = cells[0]
        instance = dict(instance)
        instance['chain'] = instance['chain'][:2]
        cells = [(instance, lam, shots, seed)]
    print(f"{len(cells)} cells x {len(ARMS)+1} arms on {cpu_count()} cores", flush=True)
    t0 = time.time()
    rows = []
    with Pool(args.workers) as pool:
        for i, out in enumerate(pool.imap_unordered(one_cell, cells, chunksize=8)):
            rows.extend(out)
            if (i + 1) % 500 == 0:
                print(f"[{time.time()-t0:6.0f}s] {i+1}/{len(cells)} cells", flush=True)
    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {args.output} {df.shape} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
