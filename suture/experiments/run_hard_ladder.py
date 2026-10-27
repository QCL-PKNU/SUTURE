"""run_hard_ladder — objective-blind proper-coloring ladder (report 22).

Every arm here reads only (y, C, b). In the hard-edge encoding the edge rows
x_uc + x_wc + s_ec = 1 are part of C, so the list-coloring decoder's
neighbour test is a syndrome operation on constraint rows; the objective is
the random vertex-colour weight matrix W, which no decoder touches. This is
the arm the soft-edge ladder cannot supply (there edges are the objective).

Arms:
  purify          discard infeasible (incumbent)
  purify_restart  on empty register, refill with S seed copies, resume at t
  purify_last     on empty register, refill with previous register
  synd            slack-aware syndrome repair (hard_decode_synd)
  nearest         coset-uniform ablation: same syndrome, no constraint-row
                  pruning -- uniform over each corrupted block's coset
  junk_synd       A0: uniform random bits through the synd decoder

Endpoints include feasible raw outputs and successful repairs (proper_seen, best cost).
Writes hard_ladder.csv in the run directory; archived CSVs are unchanged.
"""
import os, sys, time, pickle, itertools
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: F401  (repo-relative sys.path setup)
import numpy as np
import pandas as pd
from multiprocessing import Pool

SHOTS = 64
SEEDS = 12
MAX_CORRUPT = 6
MAX_REGISTER = 64


class LadderChannel:
    """Exact per-segment distributions over the active set + readout flips.

    Conventions taken verbatim from remote_validate.py's hard-edge runner:
    channels[lam][(t, r_bytes)] -> (codes, probs) with codes little-endian
    over the active set A, and p_ro = min(1e-2 * lam, 0.5)."""

    def __init__(self, rec, lam):
        self.rec = rec
        self.ch = rec["channels"][lam]
        self.A = [np.asarray(a, dtype=np.int64) for a in rec["active_sets"]]
        self.p_ro = min(1.0e-2 * lam, 0.5)
        self.n = rec["n"]

    def run_segment(self, t, init_bits, shots, rng):
        A = self.A[t]
        r = bytes(np.asarray(init_bits)[A].tolist())
        X = np.tile(np.asarray(init_bits, dtype=np.int8), (shots, 1))
        entry = self.ch.get((t, r))
        if entry is not None:
            codes, probs = entry
            draws = rng.choice(len(codes), size=shots, p=probs)
            for i_s, di in enumerate(draws):
                code = int(codes[di])
                X[i_s, A] = [(code >> i) & 1 for i in range(len(A))]
        flips = rng.random(X.shape) < self.p_ro
        X = np.where(flips, 1 - X, X).astype(np.int8)
        counts = {}
        for row in X:
            b = row.tobytes()
            counts[b] = counts.get(b, 0) + 1
        return counts


def blocks_of(bits, v):
    return np.asarray(bits[:3 * v]).reshape(v, 3)


def edge_syndrome_score(bits, rec):
    """Violated edge rows raise per-vertex suspicion -- a read of C, not f."""
    v = rec["v"]
    score = np.zeros(v, dtype=np.int64)
    for i, (a, b) in enumerate(rec["edges"]):
        for c in range(3):
            if bits[3 * a + c] + bits[3 * b + c] + bits[3 * v + 3 * i + c] != 1:
                score[a] += 1
                score[b] += 1
    return score


def decode(bits, rec, rng, prune):
    """prune=True: list-coloring repair using edge ROWS of C (blind to f).
       prune=False: coset-uniform ablation -- same syndrome, no pruning."""
    v = rec["v"]
    dec = blocks_of(bits, v)
    wts = dec.sum(axis=1)
    col = np.full(v, -1, dtype=np.int64)
    corrupt = set(np.flatnonzero(wts != 1).tolist())
    for u in range(v):
        if u not in corrupt:
            col[u] = int(dec[u].argmax())
    score = edge_syndrome_score(bits, rec)
    for u in range(v):
        if u not in corrupt and score[u] >= 3:
            corrupt.add(u)
            col[u] = -1
    if len(corrupt) > MAX_CORRUPT:
        return None
    adj = rec["adj"]
    cor = sorted(corrupt)
    lists = []
    for u in cor:
        ones = np.flatnonzero(dec[u])
        cands = list(ones) if len(ones) >= 2 else [0, 1, 2]
        rng.shuffle(cands)
        lists.append(cands)

    if not prune:
        # Coset-uniform ablation: identical syndrome and identical candidate
        # lists, drawn uniformly with no use of the edge rows.
        for i, u in enumerate(cor):
            col[u] = lists[i][rng.integers(len(lists[i]))]
        return col

    # T6 list-coloring repair, implementing Definition 5's R(y) exactly:
    # the MINIMUM-Hamming feasible repairs, sampled UNIFORMLY.
    #
    # Two things are needed and both were once missing here.
    # (a) Uniform tie-break: returning the first valid completion from a
    #     backtracking search over shuffled lists is randomized but not
    #     uniform on multi-block corruptions (measured deviations to 0.17).
    # (b) Minimum distance: distance must be measured on the FULL string,
    #     slack bits included. Two edge-consistent completions can differ
    #     sharply in global Hamming cost because each colour choice forces
    #     different slack co-flips (measured pairs at distance 1 vs 5, 2 vs
    #     6, 3 vs 7); 18% of corruptions admit such a spread, and a uniform
    #     draw over all completions would land off the minimum set about
    #     half the time in those cases, putting the decoder outside
    #     Proposition 3.
    # So: enumerate every conflict-free completion, keep only those at
    # minimum global Hamming distance to y, and draw uniformly among them.
    # Bounded by prod|Lambda(u)| <= K^c_max = 729.
    sols = []

    def bt(i):
        if i == len(cor):
            sols.append(col.copy())
            return
        u = cor[i]
        for c in lists[i]:
            if all(col[w] != c for w in adj[u] if col[w] >= 0):
                col[u] = c
                bt(i + 1)
                col[u] = -1

    bt(0)
    if not sols:
        return None
    dists = [int((canon_bits(s, rec) != bits).sum()) for s in sols]
    dmin = min(dists)
    best = [s for s, d in zip(sols, dists) if d == dmin]
    return best[rng.integers(len(best))]


def conflicts(col, edges):
    return sum(1 for a, b in edges if col[a] == col[b])


def wcost(col, W):
    """Objective value. Computed by the HARNESS for scoring only -- no arm
    reads W; every decoder sees (y, C, b) alone."""
    return float(sum(W[u][int(c)] for u, c in enumerate(col)))


def canon_bits(col, rec):
    """Colouring -> full hard-edge bitstring (assignment + slack), the same
    encoding HardPack.coloring_to_bits produces: slack s_ec = 1 - x_uc - x_wc."""
    v, edges = rec["v"], rec["edges"]
    b = np.zeros(rec["n"], dtype=np.int8)
    for u in range(v):
        b[3 * u + int(col[u])] = 1
    for i, (a, w) in enumerate(edges):
        for c in range(3):
            b[3 * v + 3 * i + c] = 1 - b[3 * a + c] - b[3 * w + c]
    return b


def feasible_hard(bits, rec):
    """Full hard-edge feasibility, identical to HardPack.is_feasible_bits:
    one-hot assignment rows, proper colouring (the edge rows admit no 0/1
    slack when both endpoints share a colour), and slack consistency. Reads
    only C and b -- the objective is the weight matrix W, never touched."""
    v, edges = rec["v"], rec["edges"]
    dec = blocks_of(bits, v)
    if not (dec.sum(axis=1) == 1).all():
        return False
    col = dec.argmax(axis=1)
    if any(col[a] == col[b] for a, b in edges):
        return False
    for i, (a, w) in enumerate(edges):
        for c in range(3):
            if bits[3 * v + 3 * i + c] != 1 - bits[3 * a + c] - bits[3 * w + c]:
                return False
    return True


def run_chain(rec, ch, arm, rng):
    v, n = rec["v"], rec["n"]
    edges = rec["edges"]
    T = len(rec["chain"])
    seed_bits = rec["seed_bits"]
    register = {seed_bits.tobytes(): SHOTS}
    proper_seen = False
    decoder_proper_seen = False
    best = np.inf
    best_cost = np.inf
    death = -1
    substitutions = 0

    for t in range(T):
        states = list(register.keys())
        w = np.array([register[s] for s in states], dtype=float)
        alloc = rng.multinomial(SHOTS, w / w.sum())
        raw = {}
        for s, a in zip(states, alloc):
            if a == 0:
                continue
            init = np.frombuffer(s, dtype=np.int8).copy()
            if arm == "junk_synd":
                X = rng.integers(0, 2, size=(int(a), n), dtype=np.int8)
                for row in X:
                    k = row.tobytes()
                    raw[k] = raw.get(k, 0) + 1
            else:
                for k, c in ch.run_segment(t, init, int(a), rng).items():
                    raw[k] = raw.get(k, 0) + c

        new_reg = {}
        for k, c in raw.items():
            bits = np.frombuffer(k, dtype=np.int8)
            # Feasibility is the FULL hard-edge system: one-hot assignment
            # rows AND edge rows with slack. Purification's filter is exactly
            # this predicate; nothing here reads the objective W.
            if feasible_hard(bits, rec):
                cc = blocks_of(bits, v).argmax(axis=1)
                decoder_proper_seen = True
                best = 0.0
                proper_seen = True
                best_cost = min(best_cost, wcost(cc, rec["W"]))
                new_reg[k] = new_reg.get(k, 0) + c
                continue
            if arm in ("synd", "junk_synd", "nearest"):
                out = decode(bits, rec, rng, prune=(arm != "nearest"))
                if out is None:
                    continue
                nc = conflicts(out, edges)
                # In the hard-edge system F = proper colourings, so a repair
                # that leaves any conflict is not a feasible output: the
                # decoder returns bot and the shot is dropped (Eq. 9).
                if nc != 0:
                    continue
                # Successful repairs count toward the hybrid output endpoint.
                best = 0.0
                proper_seen = True
                best_cost = min(best_cost, wcost(out, rec["W"]))
                decoder_proper_seen = True
                kk = canon_bits(out, rec).tobytes()
                new_reg[kk] = new_reg.get(kk, 0) + c

        if not new_reg:
            if arm == "purify_restart":
                new_reg = {seed_bits.tobytes(): SHOTS}
                substitutions += 1
            elif arm == "purify_last":
                new_reg = dict(register)
                substitutions += 1
            else:
                death = t
                break
        if len(new_reg) > MAX_REGISTER:
            new_reg = dict(sorted(new_reg.items(), key=lambda kv: -kv[1])[:MAX_REGISTER])
        register = new_reg

    return dict(survived=(death < 0), death_segment=death,
                substitutions=substitutions,
                proper_seen=bool(proper_seen),
                decoder_proper_seen=bool(decoder_proper_seen),
                endpoint_schema="hybrid-v1",
                best_conflicts=float(best) if np.isfinite(best) else np.nan,
                best_cost=float(best_cost) if np.isfinite(best_cost) else np.nan)


ARMS = ["purify", "purify_restart", "purify_last", "synd", "nearest", "junk_synd"]


def cell(args):
    rec, lam, arm, seed = args
    ch = LadderChannel(rec, lam)
    rng = np.random.default_rng(hash((rec["name"], lam, arm, seed)) % (2**32))
    out = run_chain(rec, ch, arm, rng)
    out.update(instance=rec["name"], n=rec["n"], v=rec["v"],
               lam=lam, arm=arm, seed=seed, shots=SHOTS)
    return out


def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parents[1] / 'packs/hard_ladder_bundle.pkl')
    parser.add_argument('--output', type=Path, default=Path('hard_ladder.csv'))
    parser.add_argument('--smoke', action='store_true', help='First instance, first noise level, one seed, two segments')
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 4)
    parser.add_argument('--instance', help='Restrict to one named instance')
    parser.add_argument('--lam', type=float, help='Restrict to one bundled noise level')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    with args.bundle.open("rb") as fh:
        bundle = pickle.load(fh)
    tasks = []
    instances = [r for r in bundle['instances'] if args.instance is None or r['name'] == args.instance]
    if not instances:
        parser.error('unknown instance')
    lams = [args.lam] if args.lam is not None else bundle['lams']
    if any(lam not in bundle['lams'] for lam in lams):
        parser.error('noise level is not in the bundle')
    if args.smoke:
        instances = [dict(instances[0])]
        instances[0]['chain'] = instances[0]['chain'][:2]
        lams = lams[:1]
    for rec in instances:
        rec["adj"] = [[w for (a, b) in rec["edges"]
                       for w in ((b,) if a == u else (a,) if b == u else ())]
                      for u in range(rec["v"])]
        rec["seed_bits"] = canon_bits(np.asarray(rec["seed_coloring"]), rec)
        assert feasible_hard(rec["seed_bits"], rec), f"{rec['name']}: seed infeasible"
        for lam in lams:
            for arm in ARMS:
                for s in range(1 if args.smoke else SEEDS):
                    tasks.append((rec, lam, arm, s))
    t0 = time.time()
    with Pool(args.workers) as pool:
        rows = pool.map(cell, tasks, chunksize=8)
    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(df)} rows) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
