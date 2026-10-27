"""Remote validation of the four remaining verdict items (numpy/pandas only).

E1 hard-edge : hard_bundle.pkl  -> hardedge_gcp.csv
E2 encoding  : dw_bundle.pkl    -> dw_chain.csv
E3 orderings : scale_bundle.pkl -> orderings.csv
E4 q-prior   : scale_bundle.pkl -> qprior.csv

Imports Pack/Channel/decoders/run_chain from remote_scale_campaign.py (shipped
alongside). All chains: multiprocessing over cells.
"""
import os, sys, time, pickle
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from remote_scale_campaign import Pack, Channel, decode_synd, run_chain

SEEDS = list(range(12))
DW_DECODE = {(0, 0): 0, (1, 0): 1, (1, 1): 2, (0, 1): -1}
DW_CODE = {0: (0, 0), 1: (1, 0), 2: (1, 1)}

# =====================================================================
# E1 — hard-edge
# =====================================================================

class HardInst:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)
        self.deg = [0] * self.v
        self.adj = [[] for _ in range(self.v)]
        self.eidx = {}
        for i, (a, b) in enumerate(self.edges):
            self.adj[a].append(b); self.adj[b].append(a)
            self.eidx[(a, b)] = i
        self.proper_set = set(self.proper)
        self.cost_cache = {c: float(sum(self.W[u, c[u]] for u in range(self.v)))
                           for c in self.proper}

    def cost(self, col_t):
        return self.cost_cache.get(col_t,
            float(sum(self.W[u, col_t[u]] for u in range(self.v))))

    def coloring_to_bits(self, col):
        x = np.zeros(self.n, dtype=np.int8)
        for u, c in enumerate(col):
            x[3 * u + int(c)] = 1
        for i, (a, b) in enumerate(self.edges):
            for c in range(3):
                x[3 * self.v + 3 * i + c] = 1 - x[3 * a + c] - x[3 * b + c]
        return x

    def is_feasible_bits(self, bits):
        dec = bits[:3 * self.v].reshape(self.v, 3)
        if not (dec.sum(axis=1) == 1).all():
            return False
        col = tuple(int(c) for c in dec.argmax(axis=1))
        if col not in self.proper_set:
            return False
        for i, (a, b) in enumerate(self.edges):
            for c in range(3):
                if bits[3 * self.v + 3 * i + c] != 1 - bits[3 * a + c] - bits[3 * b + c]:
                    return False
        return True


def hard_decode_synd(inst, bits, rng):
    """Slack-aware syndrome repair -> proper coloring or None (drop)."""
    dec = bits[:3 * inst.v].reshape(inst.v, 3)
    wts = dec.sum(axis=1)
    col = np.full(inst.v, -1, dtype=int)
    corrupted = set(np.flatnonzero(wts != 1))
    for u in range(inst.v):
        if u not in corrupted:
            col[u] = int(dec[u].argmax())
    # richer syndrome: violated edge rows raise per-vertex suspicion
    score = np.zeros(inst.v, dtype=int)
    for i, (a, b) in enumerate(inst.edges):
        for c in range(3):
            if bits[3 * a + c] + bits[3 * b + c] + bits[3 * inst.v + 3 * i + c] != 1:
                score[a] += 1; score[b] += 1
    for u in range(inst.v):
        if u not in corrupted and score[u] >= 3:
            corrupted.add(u); col[u] = -1
    if len(corrupted) > 6:
        return None
    corrupted = sorted(corrupted)
    lists = []
    for u in corrupted:
        ones = np.flatnonzero(dec[u])
        cand = list(ones) if len(ones) == 2 else [0, 1, 2]
        rng.shuffle(cand)
        lists.append(cand)
    best = {"assign": None}

    def bt(i):
        if best["assign"] is not None:
            return
        if i == len(corrupted):
            best["assign"] = [col[u] for u in corrupted]
            return
        u = corrupted[i]
        for c in lists[i]:
            if all(col[w] != c for w in inst.adj[u]):
                col[u] = c
                bt(i + 1)
                col[u] = -1
                if best["assign"] is not None:
                    return

    bt(0)
    if best["assign"] is None:
        return None
    for u, c in zip(corrupted, best["assign"]):
        col[u] = c
    ct = tuple(int(x) for x in col)
    return ct if ct in inst.proper_set else None


def hard_decode_nearest(inst, bits, rng):
    dec = bits[:3 * inst.v].reshape(inst.v, 3)
    col = np.empty(inst.v, dtype=int)
    for u in range(inst.v):
        ones = np.flatnonzero(dec[u])
        if len(ones) == 1: col[u] = ones[0]
        elif len(ones) == 2: col[u] = ones[rng.integers(2)]
        else: col[u] = rng.integers(3)
    ct = tuple(int(x) for x in col)
    return ct if ct in inst.proper_set else None


def hard_run_chain(inst, lam, shots, arm, rng):
    ch = inst.channels[lam]
    p_ro = min(inst.p_ro_base * lam, 0.5)
    seed_col = tuple(int(c) for c in inst.seed_coloring)
    register = {seed_col: shots}
    junk = arm == "junk_synd"
    f_per, death = [], -1
    best_cost, hit_opt = np.inf, False
    proper_list = list(inst.proper)

    for t in range(len(inst.chain)):
        A = np.array(inst.active_sets[t])
        states = list(register.keys())
        wts = np.array([register[s] for s in states], dtype=float)
        alloc = rng.multinomial(shots, wts / wts.sum())
        raw = {}
        for col_t, a in zip(states, alloc):
            if a == 0: continue
            if junk:
                X = rng.integers(0, 2, size=(int(a), inst.n), dtype=np.int8)
            else:
                fb = inst.coloring_to_bits(col_t)
                r = bytes(fb[A].tolist())
                codes, probs = ch[(t, r)]
                draws = rng.choice(len(codes), size=int(a), p=probs)
                X = np.tile(fb, (int(a), 1))
                for i_s, di in enumerate(draws):
                    code = codes[di]
                    X[i_s, A] = [(code >> i) & 1 for i in range(len(A))]
            flips = rng.random(X.shape) < p_ro
            X = np.where(flips, 1 - X, X).astype(np.int8)
            for row in X:
                k = row.tobytes()
                raw[k] = raw.get(k, 0) + 1
        tot = sum(raw.values()); n_feas = 0
        new_reg = {}
        for k, cnum in raw.items():
            bits = np.frombuffer(k, dtype=np.int8)
            if inst.is_feasible_bits(bits):
                n_feas += cnum
                ct = tuple(int(c) for c in bits[:3 * inst.v].reshape(inst.v, 3).argmax(axis=1))
                cst = inst.cost(ct)
                if cst < best_cost: best_cost = cst
                if ct == inst.x_star: hit_opt = True
                new_reg[ct] = new_reg.get(ct, 0) + cnum
            elif arm in ("rec_synd", "junk_synd", "rec_nearest", "rec_uniform"):
                if arm == "rec_uniform":
                    ct = proper_list[rng.integers(len(proper_list))]
                elif arm == "rec_nearest":
                    ct = hard_decode_nearest(inst, bits, rng)
                else:
                    ct = hard_decode_synd(inst, bits, rng)
                if ct is None:
                    continue
                new_reg[ct] = new_reg.get(ct, 0) + cnum
        f_per.append(n_feas / tot if tot else 0.0)
        if not new_reg:
            if arm == "purify_restart":
                new_reg = {seed_col: shots}
            else:
                death = t; break
        if len(new_reg) > 64:
            new_reg = dict(sorted(new_reg.items(), key=lambda kv: -kv[1])[:64])
        register = new_reg

    row = dict(survived=death < 0, death_segment=death,
               f_min=min(f_per) if f_per else np.nan,
               best_cost_seen=best_cost, hit_opt_seen=hit_opt)
    tot = sum(register.values())
    if tot > 0 and death < 0:
        row["reg_p_opt"] = register.get(inst.x_star, 0) / tot
    else:
        row["reg_p_opt"] = 0.0
    return row


def e1_cell(args):
    d, lam, S, seed = args
    inst = HardInst(d)
    rows = []
    for arm in ["purify", "purify_restart", "rec_nearest", "rec_synd", "rec_uniform", "junk_synd"]:
        rng = np.random.default_rng(3_000_000 + hash((d["name"], lam, S, seed, arm)) % 800_000)
        rows.append(dict(instance=d["name"], v=d["v"], n=d["n"], lam=lam, shots=S,
                         seed=seed, arm=arm, e_star=d["e_star"], e_med=d["e_med"],
                         **hard_run_chain(inst, lam, S, arm, rng)))
    return rows


# =====================================================================
# E2 — encoding co-design (domain-wall vs one-hot, matched chains/seeds)
# =====================================================================

def dw_run_chain(d, lam, shots, arm, rng):
    """arm in {dw_purify, dw_rec_synd}; register = colorings."""
    v = d["v"]; chain = d["chain"]
    T = d["channels_dw"][lam]["T"]; p_ro = d["channels_dw"][lam]["p_ro"]
    adj = [[] for _ in range(v)]
    for a, b in d["edges"]:
        adj[a].append(b); adj[b].append(a)

    def conflicts(col):
        return sum(1 for a, b in d["edges"] if col[a] == col[b])

    seed_col = tuple(int(c) for c in d["seed_coloring"])
    register = {seed_col: shots}
    f_per, death = [], -1
    best_seen, proper_seen = np.inf, False

    for t in range(len(chain)):
        u = chain[t][0]
        states = list(register.keys())
        wts = np.array([register[s] for s in states], dtype=float)
        alloc = rng.multinomial(shots, wts / wts.sum())
        raw = {}
        for col_t, a in zip(states, alloc):
            if a == 0: continue
            bits = np.zeros((int(a), 2 * v), dtype=np.int8)
            for uu, c in enumerate(col_t):
                b1, b2 = DW_CODE[int(c)]
                bits[:, 2 * uu] = b1; bits[:, 2 * uu + 1] = b2
            b_in = (int(bits[0, 2 * u]) << 1) | int(bits[0, 2 * u + 1])
            outs = rng.choice(4, size=int(a), p=T[t][b_in])
            bits[:, 2 * u] = (outs >> 1) & 1
            bits[:, 2 * u + 1] = outs & 1
            flips = rng.random(bits.shape) < p_ro
            bits = np.where(flips, 1 - bits, bits).astype(np.int8)
            for row_ in bits:
                k = row_.tobytes()
                raw[k] = raw.get(k, 0) + 1
        tot = sum(raw.values()); n_feas = 0
        new_reg = {}
        for k, cnum in raw.items():
            bb = np.frombuffer(k, dtype=np.int8)
            blocks = [(int(bb[2 * uu]), int(bb[2 * uu + 1])) for uu in range(v)]
            cols = [DW_DECODE[x] for x in blocks]
            bad = [uu for uu, c in enumerate(cols) if c < 0]
            if not bad:
                n_feas += cnum
                nc = conflicts(cols)
                if nc < best_seen: best_seen = nc
                if nc == 0: proper_seen = True
                kk = tuple(cols)
                new_reg[kk] = new_reg.get(kk, 0) + cnum
            elif arm == "dw_rec_synd":
                col = list(cols)
                ok = True
                for uu in bad:
                    cand = [0, 2]
                    rng.shuffle(cand)
                    cand.sort(key=lambda c: sum(1 for w in adj[uu] if col[w] == c))
                    col[uu] = cand[0]
                kk = tuple(col)
                new_reg[kk] = new_reg.get(kk, 0) + cnum
        f_per.append(n_feas / tot if tot else 0.0)
        if not new_reg:
            death = t; break
        if len(new_reg) > 64:
            new_reg = dict(sorted(new_reg.items(), key=lambda kv: -kv[1])[:64])
        register = new_reg

    return dict(survived=death < 0, death_segment=death,
                f_min=min(f_per) if f_per else np.nan,
                best_conflicts_seen=best_seen, proper_seen=proper_seen)


def e2_cell(args):
    d, lam, S, seed = args
    rows = []
    for arm in ["dw_purify", "dw_rec_synd"]:
        rng = np.random.default_rng(4_000_000 + hash((d["name"], lam, S, seed, arm)) % 800_000)
        rows.append(dict(instance=d["name"], encoding="domain-wall", v=d["v"],
                         n=2 * d["v"], lam=lam, shots=S, seed=seed,
                         arm=arm.replace("dw_", ""), **dw_run_chain(d, lam, S, arm, rng)))
    # matched one-hot runs (same seeds) via the scale sampler
    oh = dict(name=d["name"], v=d["v"], K=3, edges=d["edges"], chain=d["chain"],
              params=d["params"], seed_coloring=d["seed_coloring"])
    pack = Pack(oh)
    ch = Channel(pack, d["channels_onehot"][lam]["T"], d["channels_onehot"][lam]["p_ro"])
    for arm in ["purify", "rec_synd"]:
        rng = np.random.default_rng(4_000_000 + hash((d["name"], lam, S, seed, "oh_" + arm)) % 800_000)
        r = run_chain(pack, ch, S, arm, rng)
        rows.append(dict(instance=d["name"], encoding="one-hot", v=d["v"], n=3 * d["v"],
                         lam=lam, shots=S, seed=seed, arm=arm,
                         survived=r["survived"], death_segment=r["death_segment"],
                         f_min=r["f_min"], best_conflicts_seen=r["best_conflicts_seen"],
                         proper_seen=r["proper_seen"]))
    return rows


# =====================================================================
# E3 — chain orderings   /   E4 — Q-prior tie-breaks
# =====================================================================

def ordering_perm(inst_d, kind, pseed=0):
    chain = inst_d["chain"]; v = inst_d["v"]
    deg = np.zeros(v, dtype=int)
    for a, b in inst_d["edges"]:
        deg[a] += 1; deg[b] += 1
    T = len(chain); half = T // 2
    if kind == "default":
        return np.arange(T)
    if kind in ("degree_desc", "degree_asc"):
        order = np.argsort(-deg if kind == "degree_desc" else deg, kind="stable")
        idx = []
        for r in range(2):  # rounds
            base = r * half
            for u in order:
                idx.append(base + 2 * u); idx.append(base + 2 * u + 1)
        return np.array(idx)
    if kind == "random":
        rng = np.random.default_rng(555 + pseed)
        return rng.permutation(T)
    raise ValueError(kind)


def e3_cell(args):
    inst_d, kind, pseed, lam, S, seed = args
    perm = ordering_perm(inst_d, kind, pseed)
    d2 = dict(inst_d)
    d2["chain"] = [inst_d["chain"][i] for i in perm]
    d2["params"] = np.asarray(inst_d["params"])[perm]
    pack = Pack(d2)
    T = inst_d["channels"][lam]["T"][perm]
    ch = Channel(pack, T, inst_d["channels"][lam]["p_ro"])
    rng = np.random.default_rng(5_000_000 + hash((inst_d["name"], kind, pseed, lam, S, seed)) % 800_000)
    r = run_chain(pack, ch, S, "rec_synd", rng)
    return [dict(instance=inst_d["name"], v=inst_d["v"], ordering=kind, pseed=pseed,
                 lam=lam, shots=S, seed=seed, arm="rec_synd",
                 survived=r["survived"], f_min=r["f_min"],
                 best_conflicts_seen=r["best_conflicts_seen"], proper_seen=r["proper_seen"])]


def run_chain_qprior(pack, channel, shots, rng):
    """rec_synd with in-coset candidate lists ordered by accumulated per-vertex
    color frequencies from feasible shots (seed-gauge: no alignment needed)."""
    seed_col = tuple(int(c) for c in pack.seed_coloring)
    register = {seed_col: shots}
    prior = np.ones((pack.v, pack.K))
    f_per, death = [], -1
    best_seen, proper_seen = np.inf, False

    def decode_synd_prior(bits):
        blocks = pack.bits_to_blocks(bits)
        weights = blocks.sum(axis=1)
        corrupt = list(np.flatnonzero(weights != 1))
        col = np.full(pack.v, -1, dtype=np.int64)
        for u in np.flatnonzero(weights == 1):
            col[u] = int(blocks[u].argmax())
        if not corrupt:
            return col
        if len(corrupt) > 6:
            return None
        lists = []
        for u in corrupt:
            ones = np.flatnonzero(blocks[u])
            cand = list(ones) if len(ones) == 2 else [0, 1, 2]
            cand.sort(key=lambda c: -prior[u, c])   # prior-ordered, deterministic
            lists.append(cand)
        best = {"cost": np.inf, "assign": None}

        def bt(i, cost):
            if cost > best["cost"]: return
            if i == len(corrupt):
                if cost < best["cost"]:
                    best["cost"], best["assign"] = cost, [col[u] for u in corrupt]
                return
            u = corrupt[i]
            for c in lists[i]:
                col[u] = c
                bt(i + 1, cost + sum(1 for w in pack.adj[u] if col[w] == c))
                col[u] = -1

        bt(0, 0)
        for u, c in zip(corrupt, best["assign"]):
            col[u] = c
        return col

    for t in range(len(pack.chain)):
        states = list(register.keys())
        wts = np.array([register[s] for s in states], dtype=float)
        alloc = rng.multinomial(shots, wts / wts.sum())
        raw = {}
        for col_t, a in zip(states, alloc):
            if a == 0: continue
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
                for u in range(pack.v):
                    prior[u, col[u]] += cnum
                nc = pack.conflicts(col)
                if nc < best_seen: best_seen = nc
                if nc == 0: proper_seen = True
                kk = tuple(int(x) for x in col)
                new_reg[kk] = new_reg.get(kk, 0) + cnum
            else:
                col = decode_synd_prior(bits)
                if col is None: continue
                kk = tuple(int(x) for x in col)
                new_reg[kk] = new_reg.get(kk, 0) + cnum
        f_per.append(n_feas / tot if tot else 0.0)
        if not new_reg:
            death = t; break
        if len(new_reg) > 64:
            new_reg = dict(sorted(new_reg.items(), key=lambda kv: -kv[1])[:64])
        register = new_reg

    return dict(survived=death < 0, f_min=min(f_per) if f_per else np.nan,
                best_conflicts_seen=best_seen, proper_seen=proper_seen)


def e4_cell(args):
    inst_d, lam, S, seed = args
    pack = Pack(inst_d)
    ch = Channel(pack, inst_d["channels"][lam]["T"], inst_d["channels"][lam]["p_ro"])
    rows = []
    rng = np.random.default_rng(6_000_000 + hash((inst_d["name"], lam, S, seed, "u")) % 800_000)
    r = run_chain(pack, ch, S, "rec_synd", rng)
    rows.append(dict(instance=inst_d["name"], v=inst_d["v"], lam=lam, shots=S, seed=seed,
                     arm="rec_synd", survived=r["survived"], f_min=r["f_min"],
                     best_conflicts_seen=r["best_conflicts_seen"], proper_seen=r["proper_seen"]))
    rng = np.random.default_rng(6_000_000 + hash((inst_d["name"], lam, S, seed, "q")) % 800_000)
    r = run_chain_qprior(pack, ch, S, rng)
    rows.append(dict(instance=inst_d["name"], v=inst_d["v"], lam=lam, shots=S, seed=seed,
                     arm="rec_synd_qprior", **r))
    return rows


def run_pool(cells, fn, label):
    t0 = time.time()
    rows = []
    with Pool(min(cpu_count(), 30)) as pool:
        for i, out in enumerate(pool.imap_unordered(fn, cells, chunksize=4)):
            rows.extend(out)
            if (i + 1) % 200 == 0:
                print(f"  [{label}] {i+1}/{len(cells)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[{label}] done: {len(rows)} rows in {time.time()-t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def main():
    # E1
    with open("hard_bundle.pkl", "rb") as fh:
        hb = pickle.load(fh)
    cells = [(d, lam, S, seed) for d in hb["instances"] for lam in hb["lams"]
             for S in (64, 256) for seed in SEEDS]
    run_pool(cells, e1_cell, "E1-hardedge").to_csv("hardedge_gcp.csv", index=False)

    # E2
    with open("dw_bundle.pkl", "rb") as fh:
        db = pickle.load(fh)
    cells = [(d, lam, S, seed) for d in db["instances"] for lam in db["lams"]
             for S in (64,) for seed in SEEDS]
    run_pool(cells, e2_cell, "E2-encoding").to_csv("dw_chain.csv", index=False)

    # E3 + E4
    with open("scale_bundle.pkl", "rb") as fh:
        sb = pickle.load(fh)
    by_name = {d["name"]: d for d in sb["instances"]}
    lams34 = [2.0, 3.0, 4.0, 6.0, 8.0]
    orders = [("default", 0), ("degree_desc", 0), ("degree_asc", 0),
              ("random", 0), ("random", 1), ("random", 2)]
    cells = [(by_name[nm], kind, ps, lam, 64, seed)
             for nm in ("gcp_p16_s0", "gcp_p24_s0")
             for (kind, ps) in orders for lam in lams34 for seed in SEEDS]
    run_pool(cells, e3_cell, "E3-orderings").to_csv("orderings.csv", index=False)

    qinsts = [d for d in sb["instances"]
              if d["name"].startswith(("gcp_p12", "gcp_p16", "gcp_p24"))]
    cells = [(d, lam, 64, seed) for d in qinsts for lam in lams34 for seed in SEEDS]
    run_pool(cells, e4_cell, "E4-qprior").to_csv("qprior.csv", index=False)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
