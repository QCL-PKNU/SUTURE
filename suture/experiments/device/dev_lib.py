"""dev_lib — shared library for the Option A device campaign (reports/20).

Instances (from bench10_bundle.pkl soft_k3): gcp_v8 (v=8, n=24) and
gcp_p12_s0 (v=12, n=36), trained chains (32 / 48 segments).

Device protocol (E8 per-segment placement, reports/12):
  - each segment is an independent circuit: X-init layer -> XY(theta_t) on
    the segment's active pair -> measure all n.
  - per-segment initial_layout puts the active pair on adjacent physical
    qubits along a fixed path of n coupled qubits (no routing).
  - XY body: RXX(2th) * RYY(2th) (commuting; on the one-hot pair subspace
    {01,10} this swaps with P = sin^2(2th)... verified against the analytic
    pack channel in preflight gate G-L1 before any device use — the gate,
    not this docstring, is the authority on the angle convention).

Arms (A2): purify (discard infeasible), synd (T6 list-coloring repair —
decode_synd_soft vendored VERBATIM from experiments/remote/remote_bench10.py).
No restart arm (user directive; budget).

Budget discipline: BudgetGuard hard-stops at USAGE_CAP_S of cumulative
reported QPU usage; every A2 round checkpoints to disk before submitting
the next job, so an abort or crash loses at most one job.
"""
import json
import os
import pickle
import time

import numpy as np

K = 3
MAX_REGISTER = 64
USAGE_CAP_S = 480.0          # hard stop at 8 of the 10 available minutes
CHECK_FILE = "device/a2_state.pkl"


# ---------------- packs ----------------

def load_soft_pack(name, bundle_path="bench10_bundle.pkl"):
    with open(bundle_path, "rb") as fh:
        bundle = pickle.load(fh)
    packs = bundle["soft_k3"] if "soft_k3" in bundle else bundle["instances"]
    d = next(x for x in packs if x["name"] == name)
    v = int(d["v"])
    edges = [(int(a), int(b)) for a, b in d["edges"]]
    adj = {u: [] for u in range(v)}
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
    return dict(name=name, v=v, K=K, n=v * K, edges=edges, adj=adj,
                chain=[(int(u), int(c0), int(c1)) for u, c0, c1 in d["chain"]],
                params=np.asarray(d["params"], dtype=float),
                seed_coloring=np.asarray(d["seed_coloring"], dtype=np.int64))


def coloring_to_bits(col, v):
    out = np.zeros(v * K, dtype=np.int8)
    for u in range(v):
        out[u * K + int(col[u])] = 1
    return out


def enumerate_proper(pack, cap=200000):
    """All proper colorings (v<=12 => 3^12 = 531k assignments, fine)."""
    v = pack["v"]
    out = []
    for code in range(3 ** v):
        col, x = [], code
        for _ in range(v):
            col.append(x % 3); x //= 3
        if all(col[a] != col[b] for a, b in pack["edges"]):
            out.append(col)
            if len(out) >= cap:
                break
    return np.asarray(out, dtype=np.int64)




def sample_proper(pack, n_samples, rng, max_tries=2000):
    """n_samples distinct proper colorings via randomized backtracking.

    Replaces enumerate_proper for v > 12, where brute force over 3^v is
    infeasible (enumerate_proper's own docstring scopes it to v <= 12).
    Randomized vertex order + randomized color order per try; sparse
    K=3-colorable instances (all our packs) solve in microseconds.
    """
    v, edges = pack["v"], pack["edges"]
    nbr = {i: [] for i in range(v)}
    for a, b in edges:
        nbr[a].append(b); nbr[b].append(a)
    seen, out = set(), []
    for _ in range(max_tries):
        order = rng.permutation(v)
        col = [-1] * v
        ok = True
        for u in order:
            opts = [c for c in range(3) if all(col[w] != c for w in nbr[u])]
            if not opts:
                ok = False
                break
            col[u] = int(opts[rng.integers(len(opts))])
        if ok:
            key = tuple(col)
            if key not in seen:
                seen.add(key)
                out.append(col)
                if len(out) >= n_samples:
                    break
    if len(out) < n_samples:
        raise RuntimeError(f"sample_proper: only {len(out)}/{n_samples} found")
    return np.asarray(out, dtype=np.int64)

# ---------------- feasibility / decoding (classical) ----------------

def block_weights(bits, v):
    return np.asarray(bits).reshape(v, K).sum(axis=1)


def is_onehot(bits, v):
    return bool(np.all(block_weights(bits, v) == 1))


def bits_to_coloring(bits, v):
    return np.argmax(np.asarray(bits).reshape(v, K), axis=1)


def conflicts(bits_or_col, pack, is_bits=True):
    col = bits_to_coloring(bits_or_col, pack["v"]) if is_bits else np.asarray(bits_or_col)
    return int(sum(col[a] == col[b] for a, b in pack["edges"]))


def decode_synd_soft(v, K_, adj, bits, rng):
    """T6 list-coloring repair — vendored VERBATIM from
    suture/experiments/remote/remote_bench10.py::decode_synd_soft."""
    blocks = np.asarray(bits).reshape(v, K_)
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
        cands = list(ones) if len(ones) >= 2 else list(range(K_))
        ok = [cc for cc in cands if all(col[w] != cc for w in adj[u] if col[w] >= 0)]
        pool = ok if ok else cands
        col[u] = pool[rng.integers(len(pool))]
    out = np.zeros(v * K_, dtype=np.int8)
    for u in range(v):
        out[u * K_ + col[u]] = 1
    return out


# ---------------- circuits ----------------

def segment_circuit_baked(pack, t, init_bits):
    """X-init segment circuit (one circuit per distinct init)."""
    from qiskit import QuantumCircuit
    n = pack["n"]
    u, c0, c1 = pack["chain"][t]
    th = float(pack["params"][t])
    qa, qb = u * K + c0, u * K + c1
    qc = QuantumCircuit(n)
    for i in np.flatnonzero(np.asarray(init_bits)):
        qc.x(int(i))
    qc.barrier()
    qc.rxx(2 * th, qa, qb)
    qc.ryy(2 * th, qa, qb)
    qc.measure_all()
    return qc


def segment_layout(pack, t, path):
    """initial_layout (logical qubit i -> physical path slot) with the active
    pair on ADJACENT slots. Block u occupies path slots 3u..3u+2; (c0,c1) go
    in the first two slots of block u."""
    u, c0, c1 = pack["chain"][t]
    order = {u_: [0, 1, 2] for u_ in range(pack["v"])}
    rest = [c for c in range(K) if c not in (c0, c1)]
    order[u] = [c0, c1] + rest
    layout = [0] * pack["n"]
    for u_ in range(pack["v"]):
        for slot, c in enumerate(order[u_]):
            layout[u_ * K + c] = path[u_ * K + slot]
    return layout


def choose_path(backend, n):
    """Deterministic simple path of n qubits on the coupling map."""
    cm = backend.coupling_map
    edges = {tuple(e) for e in cm} | {tuple(e[::-1]) for e in cm}
    nq = backend.num_qubits
    nbr = {q: sorted({b for a, b in edges if a == q}) for q in range(nq)}
    best = None
    for start in sorted(range(nq), key=lambda q: -len(nbr[q]))[:20]:
        seen = {start}
        path = []

        def dfs(q):
            nonlocal best
            if best is not None:
                return
            path.append(q)
            if len(path) >= n:
                best = list(path)
                path.pop(); return
            for w in nbr[q]:
                if w not in seen:
                    seen.add(w); dfs(w); seen.discard(w)
            path.pop()

        dfs(start)
        if best:
            return best
    raise RuntimeError(f"no simple path of length {n} found")


def rows_from_bitarray(bitarray, n):
    """SamplerV2 BitArray -> (shots, n) np.int8 in LOGICAL qubit order."""
    arr = np.unpackbits(bitarray.array, axis=-1, bitorder="big")
    return arr[:, -n:][:, ::-1].astype(np.int8)   # verified in preflight G-P1


# ---------------- register / arms ----------------

class Register:
    def __init__(self, bits, rng):
        self.w = {np.asarray(bits, dtype=np.int8).tobytes(): 1.0}
        self.rng = rng
        self.n = len(bits)

    def sample_rows(self, shots):
        keys = list(self.w.keys())
        p = np.array([self.w[k] for k in keys]); p /= p.sum()
        idx = self.rng.choice(len(keys), size=shots, p=p)
        return np.stack([np.frombuffer(keys[i], dtype=np.int8) for i in idx])

    def update(self, kept_rows):
        if len(kept_rows) == 0:
            return False
        w = {}
        for r in kept_rows:
            kk = np.asarray(r, dtype=np.int8).tobytes()
            w[kk] = w.get(kk, 0.0) + 1.0
        if len(w) > MAX_REGISTER:
            w = dict(sorted(w.items(), key=lambda kv: -kv[1])[:MAX_REGISTER])
        self.w = w
        return True


def arm_postprocess(arm, rows, pack, rng):
    """rows -> (kept_rows, stats, decode_seconds). Measurement-level stats
    computed on MEASURED rows only (repaired strings never enter proper_seen
    — the standing endpoint discipline)."""
    v = pack["v"]
    kept, n_feas, n_rep = [], 0, 0
    best_conf, proper_seen = None, False
    t0 = time.perf_counter()
    for r in rows:
        if is_onehot(r, v):
            n_feas += 1
            kept.append(r)
            c = conflicts(r, pack)
            best_conf = c if best_conf is None else min(best_conf, c)
            proper_seen |= (c == 0)
        elif arm == "synd":
            fixed = decode_synd_soft(v, K, pack["adj"], r, rng)
            if fixed is not None:
                n_rep += 1
                kept.append(fixed)
    dt = time.perf_counter() - t0
    return kept, dict(f_t=n_feas / max(len(rows), 1), n_repaired=n_rep,
                      best_conflicts_measured=best_conf,
                      proper_seen_measured=bool(proper_seen)), dt


# ---------------- budget / latency ----------------

def job_usage_seconds(job):
    """Defensive usage extraction across runtime versions."""
    try:
        u = job.usage()
        if isinstance(u, (int, float)):
            return float(u)
        if hasattr(u, "seconds"):
            return float(u.seconds)
    except Exception:
        pass
    try:
        m = job.metrics()
        us = m.get("usage", {})
        val = us.get("seconds", us.get("quantum_seconds"))
        return float(val) if val is not None else None
    except Exception:
        return None


class BudgetGuard:
    def __init__(self, cap_s=USAGE_CAP_S, used_s=0.0):
        self.cap_s = cap_s
        self.used_s = used_s

    def add_job(self, job):
        u = job_usage_seconds(job)
        if u is not None:
            self.used_s += u
        return u

    def check(self):
        if self.used_s > self.cap_s:
            raise RuntimeError(
                f"BUDGET STOP: cumulative QPU usage {self.used_s:.0f}s > cap {self.cap_s:.0f}s")


def job_timestamps(job):
    try:
        m = job.metrics()
        ts = m.get("timestamps", {}) or {}
        return dict(created=ts.get("created"), running=ts.get("running"),
                    finished=ts.get("finished"))
    except Exception:
        return dict(created=None, running=None, finished=None)


def append_jsonl(path, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
