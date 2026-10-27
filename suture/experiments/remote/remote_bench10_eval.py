"""remote_bench10 — GCP-only Fig 9/10-style sweep (numpy/pandas only).

Usage: python3 remote_bench10.py {kv|soft}

Arms:
  purify      upstream run_and_pick semantics: keep feasible (C x = b) shots,
              multinomial reallocation; chain DIES if no survivors (no fallback,
              faithful to rasengan_segmented.run_and_pick).
  repair      purify's keep-rule replaced by: infeasible shots go through the
              generic linear-system syndrome decoder; repaired shots reinjected.
  rec_uniform reinjection control: infeasible shots -> uniform draw from F.
  junk        A0 control: register replaced by uniform random bits each
              segment, then the repair decoder (documents solver contamination).

Endpoints per run, computed on the FINAL register (paper semantics:
DataAnalyzer.summary over collapse states) AND measurement-level:
  best_solution_probs (%), in_constraints_probs (%), ARG      [paper metrics]
  arg_seen, hit_opt_seen, proper_seen (soft family), f_min, survived
Paper metrics on the final register are reported for fidelity with Fig 9/10;
per reports 07-12 the measurement-level endpoints are the evidential ones —
both are in the CSV, the report says which is which.

Grid: 7 lams x cases x 12 seeds x S=1024 (paper shot count).
Partial CSV every 200 cells (E5 lesson).
"""
import os, sys, time, pickle

# thread pinning before numpy import (32-core lesson from the channel build)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from itertools import combinations
from math import comb
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

SEEDS = list(range(12))
SHOTS = 1024
MAX_REGISTER = 256
MAX_CORRUPT = 6
MAX_WORK = 200_000
MAX_CANDIDATES = 64


# ---------- generic repair (verbatim logic from scripts/repair_generic.py) ----------

def syndrome(C, x):
    return np.flatnonzero(C[:, :-1] @ x != C[:, -1])


_COMBO_CACHE = {}
_REPAIR_CACHE = {}   # y.tobytes() -> None | (comp_col_lists, comp_sol_lists); reset per cell


def _repair_components(C, x):
    """Exact min-weight repair search, decomposed over connected components
    of the suspect columns (columns adjacent iff they co-occur in any row).
    delta(row) depends only on the component owning that row's suspect
    columns, so the global min-weight solution set is the PRODUCT of the
    per-component min-weight solution sets, and total w* = sum of component
    minima. Returns (w0_global, w_star, ns_global, comp_cols, comp_sols)
    or None-marker (None fields) when some component is unrepairable within
    MAX_CORRUPT. Enumeration within a component mirrors the original
    weight-ordered combo search restricted to the component's columns and
    checked on ALL rows touching them (satisfied rows must stay at delta 0).
    """
    A = C[:, :-1]
    viol = np.flatnonzero(A @ x != C[:, -1])
    suspects = np.flatnonzero(np.any(A[viol] != 0, axis=0))
    resid = (C[:, -1] - A @ x).astype(np.int64)
    w0 = int(np.abs(resid).max())
    ns = len(suspects)
    if ns == 0:
        return w0, None, ns, None, None
    # union-find over suspect columns via shared rows
    pos = {int(j): i for i, j in enumerate(suspects)}
    parent = list(range(ns))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    sus_mask = np.zeros(A.shape[1], dtype=bool); sus_mask[suspects] = True
    for r in range(A.shape[0]):
        cols = np.flatnonzero((A[r] != 0) & sus_mask)
        for j in cols[1:]:
            ra, rb = find(pos[int(cols[0])]), find(pos[int(j)])
            if ra != rb:
                parent[ra] = rb
    groups = {}
    for i in range(ns):
        groups.setdefault(find(i), []).append(i)
    sign = 1 - 2 * x[suspects].astype(np.int64)
    # global bail weight: the original returns None at the first w with
    # comb(ns, w) > MAX_WORK, so any solution of TOTAL weight >= w_bail is
    # unreachable there. A component minimum >= w_bail forces total >= w_bail
    # -> None; hence capping per-component enumeration at w_bail - 1 is
    # exactly outcome-equivalent (big-ns packs: ns=64 -> w_bail=4, so the
    # comb(40..64, 4+) enumerations that dominated v4's runtime never run).
    w_bail = MAX_CORRUPT + 1
    for w in range(1, MAX_CORRUPT + 2):
        if w <= ns and comb(ns, w) > MAX_WORK:
            w_bail = w
            break
    comp_cols, comp_sols, w_star = [], [], 0
    for g in groups.values():
        cols = suspects[g]
        rows = np.flatnonzero(np.any(A[:, cols] != 0, axis=1))
        r_c = resid[rows]
        if not r_c.any():
            continue   # zero-residual component: optimal selection is empty
        Acol = A[np.ix_(rows, cols)] * sign[g][None, :]
        k = len(g)
        w0_c = int(np.abs(r_c).max())
        sols = None
        for w in range(max(1, w0_c), min(MAX_CORRUPT, k, w_bail - 1) + 1):
            if comb(k, w) > MAX_WORK:
                return int(np.abs(resid).max()), None, ns, None, None
            idx = _combo_idx(k, w)
            delta = Acol.T[idx].sum(axis=1)
            ok = np.flatnonzero(np.all(delta == r_c[None, :], axis=1))
            if len(ok) > 0:
                sols = [cols[idx[o]] for o in ok]
                w_star += w
                break
        if sols is None:
            return w0, None, ns, None, None   # component unrepairable
        comp_cols.append(cols); comp_sols.append(sols)
    return w0, w_star, ns, comp_cols, comp_sols


def _combo_idx(ns, w):
    """Cached index array of C(ns, w) combinations (identical order to
    itertools.combinations — preserves the original tie-break enumeration)."""
    key = (ns, w)
    hit = _COMBO_CACHE.get(key)
    if hit is None:
        hit = np.array(list(combinations(range(ns), w)), dtype=np.int64)
        _COMBO_CACHE[key] = hit
    return hit


def repair(C, x, rng, max_corrupt=MAX_CORRUPT):
    """Component-decomposed min-weight repair. OUTCOME-EQUIVALENT to the
    original single-enumeration search (A/B set-verified): the success/None
    set and the uniform distribution over min-weight repairs are identical;
    only the rng draw SEQUENCE differs (one draw per component instead of
    one global draw) — distributionally the same tie-break.

    Exact-equivalence guards replicated from the original:
      - global weight loop bail: for w in [max(1,w0), w*], if
        comb(ns_global, w) > MAX_WORK the original would have returned None
        before reaching w* — replicated on comb values without enumerating;
      - w* > max_corrupt -> None; unrepairable component -> None.
    Per-y results memoized in _REPAIR_CACHE (deterministic given y); the
    rng tie-break draw happens per occurrence, outside the cache.
    """
    x = np.asarray(x, dtype=np.int8)
    A_ = C[:, :-1]
    resid_full = (C[:, -1] - A_ @ x).astype(np.int64)
    viol = np.flatnonzero(resid_full)
    if len(viol) == 0:
        return x.copy()
    # signature key: the component analysis depends on x only via
    # (violated rows, their residuals, x restricted to the suspect columns) —
    # spectator bits are irrelevant, so this key is correctness-neutral.
    # Measured at 64q (kpp_p16b4, lam=8 synth): ~162k unique signatures per
    # arm-run out of ~178k strings — signatures barely compress there and the
    # cache gives NO speedup (590s vs 582s); it helps only at smaller n where
    # suspect sets recur. NOTE: families_eval_sweep.csv was produced by the
    # x.tobytes()-keyed variant; both keys verified outcome-equivalent.
    suspects_k = np.flatnonzero(np.any(A_[viol] != 0, axis=0))
    key = (viol.tobytes(), resid_full[viol].tobytes(), x[suspects_k].tobytes())
    hit = _REPAIR_CACHE.get(key, "MISS")
    if hit == "MISS":
        w0, w_star, ns, comp_cols, comp_sols = _repair_components(C, x)
        if ns == 0 or w_star is None or w_star > max_corrupt:
            hit = None
        else:
            bail = False
            for w in range(max(1, w0), w_star + 1):
                if w > ns:
                    break
                if comb(ns, w) > MAX_WORK:
                    bail = True
                    break
            hit = None if bail else (comp_sols,)
        if len(_REPAIR_CACHE) < 500_000:
            _REPAIR_CACHE[key] = hit
    if hit is None:
        return None
    out = x.copy()
    for sols in hit[0]:
        pick = sols[rng.integers(len(sols))] if len(sols) > 1 else sols[0]
        for j in pick:
            out[j] = 1 - out[j]
    return out


def decode_synd_soft(v, K, adj, bits, rng):
    """T6 list-coloring repair for soft one-hot GCP: syndrome = non-weight-1
    assignment blocks; candidate lists pruned by graph edges against intact
    neighbors; min-conflict assignment, uniform tie-break. (The 'our pipeline'
    decoder for the proper-coloring family — the generic linear-system repair
    sees only assignment rows here and degenerates to coset-uniform.)"""
    blocks = np.asarray(bits).reshape(v, K)
    weights = blocks.sum(axis=1)
    col = np.full(v, -1, dtype=np.int64)
    for u in range(v):
        if weights[u] == 1:
            col[u] = int(np.argmax(blocks[u]))
    corrupt = [u for u in range(v) if col[u] < 0]
    if len(corrupt) > 8:
        return None
    for u in corrupt:
        ones = np.flatnonzero(blocks[u])
        cands = list(ones) if len(ones) >= 2 else list(range(K))
        # prune colors conflicting with intact neighbors
        ok = [cc for cc in cands if all(col[w] != cc for w in adj[u] if col[w] >= 0)]
        pool = ok if ok else cands
        col[u] = pool[rng.integers(len(pool))]
    out = np.zeros(v * K, dtype=np.int8)
    for u in range(v):
        out[u * K + col[u]] = 1
    return out


# ---------- channel sampling ----------

class KvChannel:
    """Restriction-indexed exact distributions + uniform readout flips."""
    def __init__(self, d, ch, lam):
        self.d, self.ch = d, ch
        self.p_ro = min(1.0e-2 * lam, 0.5)
        self.active = [np.asarray(a, dtype=np.int64) for a in d["active_sets"]]

    def run_segment(self, t, init_bits, shots, rng):
        A = self.active[t]
        r = bytes(init_bits[A].tolist())
        entry = self.ch.get((t, r))
        out = np.tile(init_bits, (shots, 1))
        if entry is not None:
            codes, probs = entry
            draw = codes[rng.choice(len(codes), size=shots, p=probs)]
            for i in range(len(A)):
                out[:, A[i]] = (draw >> i) & 1
        # else: init restriction unseen (post-readout drift) -> identity channel
        if self.p_ro > 0:
            flips = rng.random(out.shape) < self.p_ro
            out = np.where(flips, 1 - out, out)
        return out.astype(np.int8)


class SoftChannel:
    """4x4 pair matrices + uniform readout flips (remote_scale_campaign semantics)."""
    def __init__(self, d, T, lam):
        self.d, self.T = d, T
        self.p_ro = min(1.0e-2 * lam, 0.5)
        self.chain = d["chain"]

    def run_segment(self, t, init_bits, shots, rng):
        u, cf, ct = self.chain[t]
        i_from, i_to = u * 3 + cf, u * 3 + ct
        b_in = (int(init_bits[i_from]) << 1) | int(init_bits[i_to])
        outs = rng.choice(4, size=shots, p=self.T[t][b_in])
        out = np.tile(init_bits, (shots, 1))
        out[:, i_from] = (outs >> 1) & 1
        out[:, i_to] = outs & 1
        if self.p_ro > 0:
            flips = rng.random(out.shape) < self.p_ro
            out = np.where(flips, 1 - out, out)
        return out.astype(np.int8)


# ---------- chain run ----------

def run_chain(C, seed_bits, n_seg, channel, arm, rng, F_bits, obj, best_cost,
              synd_ctx=None):
    """synd_ctx = (v, K, adj) enables the 'repair_synd' arm (soft family)."""
    n = len(seed_bits)
    Fset = {fb.tobytes() for fb in F_bits} if F_bits is not None else None
    register = {seed_bits.tobytes(): SHOTS}
    f_min, died_at = 1.0, -1
    seen_costs = []          # feasible measurement outcomes (cost per shot batch)
    for t in range(n_seg):
        raw = {}
        if arm in ("junk", "junk_synd"):
            for _ in range(4):
                kb = rng.integers(0, 2, size=n).astype(np.int8).tobytes()
                raw[kb] = raw.get(kb, 0) + SHOTS // 4
        else:
            for kb, w in register.items():
                outs = channel.run_segment(t, np.frombuffer(kb, dtype=np.int8), int(w), rng)
                for row in outs:
                    k2 = row.tobytes()
                    raw[k2] = raw.get(k2, 0) + 1
        tot = sum(raw.values())
        keep, w_feas = {}, 0
        for kb, w in raw.items():
            x = np.frombuffer(kb, dtype=np.int8)
            if len(syndrome(C, x)) == 0:
                keep[kb] = keep.get(kb, 0) + w
                w_feas += w
                seen_costs.extend([obj(x)] * w)
            elif arm in ("repair", "junk"):
                fixed = repair(C, x, rng)
                if fixed is not None:
                    kb2 = fixed.tobytes()
                    keep[kb2] = keep.get(kb2, 0) + w
            elif arm in ("repair_synd", "junk_synd"):
                fixed = decode_synd_soft(*synd_ctx, x, rng)
                if fixed is not None:
                    kb2 = fixed.tobytes()
                    keep[kb2] = keep.get(kb2, 0) + w
            elif arm == "rec_uniform" and F_bits is not None:
                pick = F_bits[rng.integers(len(F_bits))]
                kb2 = pick.tobytes()
                keep[kb2] = keep.get(kb2, 0) + w
        f_min = min(f_min, w_feas / max(tot, 1))
        if not keep:
            died_at = t
            break
        keys = list(keep)
        p = np.array([keep[k] for k in keys], dtype=float)
        if len(keys) > MAX_REGISTER:
            top = np.argsort(-p)[:MAX_REGISTER]
            keys = [keys[i] for i in top]; p = p[top]
        draw = rng.multinomial(SHOTS, p / p.sum())
        register = {k: int(x) for k, x in zip(keys, draw) if x > 0}
    # endpoints
    survived = died_at < 0
    # paper metrics on final register
    bsp = icp = 0.0; mean_cost = 0.0
    if survived and register:
        totr = sum(register.values())
        for kb, w in register.items():
            x = np.frombuffer(kb, dtype=np.int8)
            pr = w / totr
            cost = obj(x)
            mean_cost += pr * cost
            if len(syndrome(C, x)) == 0:
                icp += pr
                if cost == best_cost:
                    bsp += pr
        arg = abs((mean_cost - best_cost) / (best_cost + 1e-8))
    else:
        arg = np.nan
    # measurement-level
    seen = np.array(seen_costs) if seen_costs else np.array([])
    arg_seen = abs((seen.mean() - best_cost) / (best_cost + 1e-8)) if len(seen) else np.nan
    hit_opt_seen = bool(len(seen) and (seen == best_cost).any())
    return dict(survived=survived, death_segment=died_at, f_min=f_min,
                best_solution_probs=bsp * 100, in_constraints_probs=icp * 100,
                ARG=arg, arg_seen=arg_seen, hit_opt_seen=hit_opt_seen,
                best_cost_seen=float(seen.min()) if len(seen) else np.nan,
                n_feasible_shots=int(len(seen)))


# ---------- cells ----------

def make_obj(spec):
    kind = spec[0]
    if kind == "linear":
        c = np.asarray(spec[1], dtype=float)
        return lambda bits: float(c @ np.asarray(bits, dtype=float))
    if kind == "kpp":
        _, pairs, V, K = spec
        def f(bits):
            x = np.asarray(bits[:V * K]).reshape(V, K)
            blk = np.argmax(x, axis=1)  # valid on feasible strings only
            return float(sum(w for (a, b), w in pairs if blk[a] == blk[b]))
        return f
    raise ValueError(kind)


def kv_cell(args):
    """Eval cell: native objective from obj_spec (F-lookup abandoned at scale;
    run_chain evaluates obj only on feasible strings). rec_uniform only where
    the pack kept an enumerable F (oracle arm needs F)."""
    d, ch_by_lam, lam, seed = args
    _REPAIR_CACHE.clear()   # keys are bitstrings; equal-n packs would collide
    C = d["constr"]
    obj = make_obj(d["obj_spec"])
    arms = ["purify", "repair", "junk"] + (["rec_uniform"] if "F_bits" in d else [])
    rows = []
    for arm in arms:
        rng = np.random.default_rng(31_000_000 + hash((d["name"], lam, seed, arm)) % 900_000)
        ch = KvChannel(d, ch_by_lam[lam], lam)
        r = run_chain(C, d["seed_bits"], len(d["chain_vecs"]), ch, arm, rng,
                      d.get("F_bits"), obj, d["best_cost"])
        rows.append(dict(family=d.get("family", "kv"), instance=d["name"],
                         scale=d.get("scale", d["name"].rsplit("_", 1)[0]),
                         n=d["n"], lam=lam, seed=seed, arm=arm, shots=SHOTS, **r))
    return rows


def soft_cell(args):
    d, T_by_lam, lam, seed = args
    v = d["v"]
    C = np.zeros((v, v * 3 + 1), dtype=np.int64)
    for u in range(v):
        C[u, u * 3:(u + 1) * 3] = 1
        C[u, -1] = 1
    adj = [[] for _ in range(v)]
    for a, b in d["edges"]:
        adj[a].append(b); adj[b].append(a)
    def obj(bits):
        col = np.argmax(np.asarray(bits).reshape(v, 3), axis=1)
        return float(sum(1 for u in range(v) for w in adj[u] if w > u and col[u] == col[w]))
    seed_bits = np.zeros(v * 3, dtype=np.int8)
    for u, c in enumerate(d["seed_coloring"]):
        seed_bits[u * 3 + int(c)] = 1
    rows = []
    # repair_synd = T6 edge-aware decoder (our pipeline); repair = generic
    # linear-system decoder (coset-uniform here — weak-decoder control);
    # junk_synd = A0 control for the synd decoder. No rec_uniform (no F enum at v=16).
    for arm in ("purify", "repair", "repair_synd", "junk", "junk_synd"):
        rng = np.random.default_rng(37_000_000 + hash((d["name"], lam, seed, arm)) % 900_000)
        ch = SoftChannel(d, T_by_lam[lam], lam)
        r = run_chain(C, seed_bits, len(d["chain"]), ch, arm, rng, None, obj, 0.0,
                      synd_ctx=(v, 3, adj))
        r["proper_seen"] = r.pop("hit_opt_seen")     # best_cost=0 == proper coloring
        rows.append(dict(family="soft", instance=d["name"], scale=f"v{v}",
                         n=v * 3, lam=lam, seed=seed, arm=arm, shots=SHOTS, **r))
    return rows


def main():
    which = sys.argv[1]
    with open("families_eval_bundle.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    with open("families_eval_channels.pkl", "rb") as fh:
        channels = pickle.load(fh)
    lams = channels["lams"]
    if which == "kv":
        cells = [(d, channels["kv"][d["name"]], lam, seed)
                 for d in bundle["paper_kv"] for lam in lams for seed in SEEDS]
        fn, out_csv = kv_cell, "families_eval_sweep.csv"
    else:
        cells = [(d, channels["soft"][d["name"]], lam, seed)
                 for d in bundle["soft_k3"] for lam in lams for seed in SEEDS]
        fn, out_csv = soft_cell, "bench10_soft.csv"
    # resume: if a prior partial CSV is staged, skip its completed cells
    rows = []
    done = set()
    if os.path.exists("resume.partial.csv"):
        prev = pd.read_csv("resume.partial.csv")
        rows = prev.to_dict("records")
        done = {(r_["instance"], r_["lam"], r_["seed"]) for r_ in rows}
        print(f"resumed {len(done)} cells ({len(rows)} rows) from partial", flush=True)
    cells = [cell for cell in cells
             if (cell[0]["name"], cell[2], cell[3]) not in done]

    print(f"{len(cells)} cells on {cpu_count()} cores", flush=True)
    t0 = time.time()
    pool_n = int(os.environ.get("POOL", min(cpu_count(), 30)))
    with Pool(pool_n) as pool:
        for i, out in enumerate(pool.imap_unordered(fn, cells, chunksize=1)):
            rows.extend(out)
            if (i + 1) % 20 == 0:
                print(f"[{time.time()-t0:6.0f}s] {i+1}/{len(cells)}", flush=True)
                pd.DataFrame(rows).to_csv(out_csv.replace(".csv", ".partial.csv"), index=False)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(rows)} rows) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
