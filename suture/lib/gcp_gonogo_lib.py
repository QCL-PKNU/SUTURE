"""Decoders, chain protocols, and endpoints for the GCP go/no-go experiment.

Decoder classes (per T6):
  nearest  - per-block coset-uniform projection (T6.1's NEAREST: weight-1 keep,
             weight-0/3 uniform over K, weight-2 uniform over the two 1-positions).
             Ignores the graph. This is the weak-decoder control.
  synd     - syndrome-guided list-coloring repair (T6.3): intact blocks keep their
             color; corrupted blocks get candidate lists (weight-2 -> its two
             positions, else all K); residual min-conflict list-coloring solved
             exactly by backtracking over corrupted blocks (fallback: resample
             from register when > MAX_CORRUPT blocks corrupted).
  uniform  - infeasible sample -> uniform random coloring (continuation with
             zero per-sample information).
Arms:
  purify, purify_restart, purify_last, rec_nearest, rec_synd, rec_uniform,
  junk_synd (A0: uniform-random raw bits + the SYND decoder), feas_control.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from itertools import product

MAX_CORRUPT = 6


# ---------------------------------------------------------------------------
# Decoders (operate on raw bit arrays; return coloring or None for fallback)
# ---------------------------------------------------------------------------

def decode_nearest(pack, bits: np.ndarray, rng) -> np.ndarray:
    blocks = pack.bits_to_blocks(bits)
    col = np.empty(pack.v, dtype=np.int64)
    for u in range(pack.v):
        w = blocks[u]
        ones = np.flatnonzero(w)
        if len(ones) == 1:
            col[u] = ones[0]
        elif len(ones) == 2:
            col[u] = ones[rng.integers(2)]
        else:  # weight 0 or 3
            col[u] = rng.integers(pack.K)
    return col


def decode_synd(pack, bits: np.ndarray, rng) -> np.ndarray | None:
    blocks = pack.bits_to_blocks(bits)
    weights = blocks.sum(axis=1)
    intact = np.flatnonzero(weights == 1)
    corrupt = np.flatnonzero(weights != 1)
    col = np.full(pack.v, -1, dtype=np.int64)
    for u in intact:
        col[u] = int(blocks[u].argmax())
    if len(corrupt) == 0:
        return col
    if len(corrupt) > MAX_CORRUPT:
        return None  # fallback: resample from register
    # candidate lists per corrupted block (T6.1)
    lists = []
    for u in corrupt:
        ones = np.flatnonzero(blocks[u])
        cand = list(ones) if len(ones) == 2 else list(range(pack.K))
        rng.shuffle(cand)
        lists.append(cand)
    # exact min-conflict assignment over corrupted blocks (branch & bound)
    corrupt_l = list(corrupt)
    best = {"cost": np.inf, "assign": None, "count": 0}

    def edge_cost(u, c):
        cost = 0
        for w in pack.adj[u]:
            if col[w] == c:
                cost += 1
        return cost

    def bt(i, cost):
        if cost > best["cost"]:
            return
        if i == len(corrupt_l):
            if cost < best["cost"]:
                best["cost"], best["assign"], best["count"] = cost, [col[u] for u in corrupt_l], 1
            elif cost == best["cost"] and rng.random() < 1.0 / (best["count"] + 1):
                best["assign"] = [col[u] for u in corrupt_l]
                best["count"] += 1
            return
        u = corrupt_l[i]
        for c in lists[i]:
            col[u] = c
            bt(i + 1, cost + edge_cost(u, c))
            col[u] = -1

    # note: edge_cost inside bt double-counts nothing because col[w] for later
    # corrupted blocks is -1 until assigned; pairs among corrupted blocks are
    # counted when the second endpoint is placed.
    bt(0, 0)
    for u, c in zip(corrupt_l, best["assign"]):
        col[u] = c
    return col


def decode_uniform(pack, bits: np.ndarray, rng) -> np.ndarray:
    return rng.integers(0, pack.K, size=pack.v)


DECODERS = {"nearest": decode_nearest, "synd": decode_synd, "uniform": decode_uniform}


# ---------------------------------------------------------------------------
# Chain runner
# ---------------------------------------------------------------------------

@dataclass
class GcpRunResult:
    f_per_segment: list = field(default_factory=list)
    death_segment: int = -1
    restarts: int = 0
    final_register: dict = field(default_factory=dict)   # coloring-tuple -> weight
    best_conflicts_seen: float = np.inf                   # among feasible MEASUREMENTS
    proper_seen: bool = False
    total_shots: int = 0


MAX_REGISTER = 64  # cap register support (protocol approximation, same for all arms)


def _register_push(reg, col, w=1):
    k = tuple(int(c) for c in col)
    reg[k] = reg.get(k, 0) + w


def run_chain(pack, channel, shots: int, arm: str, rng) -> GcpRunResult:
    res = GcpRunResult()
    seed_col = tuple(int(c) for c in pack.seed_coloring)
    register = {seed_col: shots}
    junk = arm == "junk_synd"

    for t in range(len(pack.chain)):
        # allocate shots over register states
        states = list(register.keys())
        wts = np.array([register[s] for s in states], dtype=float)
        alloc = rng.multinomial(shots, wts / wts.sum())
        raw_counts: dict[bytes, int] = {}
        for col_t, a in zip(states, alloc):
            if a == 0:
                continue
            init_bits = pack.coloring_to_bits(col_t)
            if junk:
                X = rng.integers(0, 2, size=(int(a), pack.n), dtype=np.int8)
                for row in X:
                    k = row.tobytes()
                    raw_counts[k] = raw_counts.get(k, 0) + 1
            else:
                for k, c in channel.run_segment(t, init_bits, int(a), rng).items():
                    raw_counts[k] = raw_counts.get(k, 0) + c
        res.total_shots += int(alloc.sum())

        # feasibility + bookkeeping
        tot = sum(raw_counts.values())
        new_reg: dict[tuple, int] = {}
        n_feas = 0
        for k, c in raw_counts.items():
            bits = np.frombuffer(k, dtype=np.int8)
            if pack.is_feasible_bits(bits):
                n_feas += c
                col = pack.decode_coloring(bits)
                nc = pack.conflicts(col)
                if nc < res.best_conflicts_seen:
                    res.best_conflicts_seen = nc
                if nc == 0:
                    res.proper_seen = True
                if arm.startswith("purify") or arm.startswith("rec_") or junk:
                    _register_push(new_reg, col, c)
            else:
                if arm == "rec_nearest" or arm == "rec_uniform" or arm == "rec_synd" or junk:
                    mode = {"rec_nearest": "nearest", "rec_uniform": "uniform",
                            "rec_synd": "synd", "junk_synd": "synd"}[arm]
                    col = DECODERS[mode](pack, bits, rng)
                    if col is None:  # synd fallback: resample from current register
                        continue
                    _register_push(new_reg, col, c)
        res.f_per_segment.append(n_feas / tot if tot else 0.0)

        # register update / death handling
        if not new_reg:
            if arm == "purify_restart":
                new_reg = {seed_col: shots}
                res.restarts += 1
            elif arm == "purify_last":
                new_reg = dict(register)
                res.restarts += 1
            else:
                res.death_segment = t
                break
        # cap support
        if len(new_reg) > MAX_REGISTER:
            top = sorted(new_reg.items(), key=lambda kv: -kv[1])[:MAX_REGISTER]
            new_reg = dict(top)
        register = new_reg

    res.final_register = register
    return res


def endpoint_row(pack, res: GcpRunResult) -> dict:
    reg = res.final_register
    tot = sum(reg.values())
    row = dict(survived=res.death_segment < 0, death_segment=res.death_segment,
               restarts=res.restarts, total_shots=res.total_shots,
               f_min=min(res.f_per_segment) if res.f_per_segment else np.nan,
               f_last=res.f_per_segment[-1] if res.f_per_segment else np.nan,
               best_conflicts_seen=res.best_conflicts_seen,
               proper_seen=res.proper_seen)
    if tot > 0:
        confs = np.array([pack.conflicts(k) for k in reg])
        wts = np.array([reg[k] for k in reg], dtype=float)
        row["reg_min_conflicts"] = float(confs.min())
        row["reg_mean_conflicts"] = float((confs * wts).sum() / wts.sum())
        row["reg_proper_rate"] = float(wts[confs == 0].sum() / wts.sum())
    else:
        row["reg_min_conflicts"] = np.nan
        row["reg_mean_conflicts"] = np.nan
        row["reg_proper_rate"] = 0.0
    return row


def feasible_skyline(pack, budget: int, rng) -> dict:
    cols = rng.integers(0, pack.K, size=(budget, pack.v))
    confs = np.array([pack.conflicts(c) for c in cols])
    return dict(survived=True, death_segment=-1, restarts=0, total_shots=budget,
                f_min=1.0, f_last=1.0,
                best_conflicts_seen=float(confs.min()),
                proper_seen=bool((confs == 0).any()),
                reg_min_conflicts=float(confs.min()),
                reg_mean_conflicts=float(confs.mean()),
                reg_proper_rate=float((confs == 0).mean()))
