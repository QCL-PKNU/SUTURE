"""E5 — fake-Heron hardware-proxy campaign (runs on the GPU server).

Inputs: fake_heron_bundle.pkl (QPY-transpiled segment bodies + metadata).
Noise: NoiseModel.from_backend(FakeTorino()) — real Heron r1 calibration data
(gate errors, T1/T2 thermal relaxation, per-qubit readout error). No crosstalk
(not modeled by fake backends; recorded as a threat, answered only on device).

Parts:
  A decode-destination: for each proper coloring (<=60) and each segment,
    run the transpiled body, classify every output: feasible? corrupted-block
    count? per-block Hamming weight profile? -> block-locality verdict.
    (deviation from depolarizing model: readout error dominates and is
    q-dependent under calibration data.)
  B f_t decay: seed register, run chain forward (purify semantics), record
    per-segment feasible fraction.
  C chain arms at S=64: purify / rec_synd / rec_uniform / junk_synd, seeds 0-7.
    Per-(segment, init) output distributions are estimated on demand from
    ORACLE_SHOTS=512 GPU shots and cached (registers concentrate on few states,
    so the cache stays small).

GPU: AerSimulator(device='GPU', method='density_matrix' for <=15q parts,
statevector trajectories for 24q) with batched_shots_gpu. Falls back to CPU
automatically if GPU init fails.

Output: fake_heron_dd.csv (part A), fake_heron_ft.csv (B), fake_heron_arms.csv (C).
"""
import io, os, sys, time, pickle
from itertools import product

import numpy as np
import pandas as pd
from qiskit import qpy, QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeTorino

ORACLE_SHOTS = 256   # 512 -> 256 after the first run timed out mid part-C at 24q
SHOTS_ARM = 64
SEEDS = list(range(8))
MAX_CORRUPT = 6
MAX_REGISTER = 64

BACKEND = FakeTorino()
NOISE = NoiseModel.from_backend(BACKEND)


def make_sim():
    for dev in ("GPU", "CPU"):
        try:
            sim = AerSimulator(device=dev, method="statevector",
                               noise_model=NOISE, batched_shots_gpu=(dev == "GPU"))
            sim.set_options(max_parallel_experiments=0)
            # smoke
            qc = QuantumCircuit(2); qc.h(0); qc.cx(0, 1); qc.measure_all()
            from qiskit import transpile as _tp
            sim.run(_tp(qc, sim), shots=16).result()
            print(f"simulator device: {dev}", flush=True)
            return sim
        except Exception as e:
            print(f"{dev} init failed: {type(e).__name__}: {e}", flush=True)
    raise RuntimeError("no simulator")


SIM = make_sim()


class Inst:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)
        self.adj = [[] for _ in range(self.v)]
        for a, b in self.edges:
            self.adj[a].append(b); self.adj[b].append(a)
        self.circs = []
        for seg in self.segments:
            self.circs.append(qpy.load(io.BytesIO(seg["qpy"]))[0])
        self.layout = self.segments[0]["init_layout"] if self.segments else None

    # --- codec (one-hot K colors, logical bit i = layout[i] physical) ---
    def coloring_to_bits(self, col):
        x = np.zeros(self.n, dtype=np.int8)
        for u, c in enumerate(col):
            x[u * self.K + int(c)] = 1
        return x

    def bits_blocks(self, bits):
        return bits.reshape(self.v, self.K)

    def is_feasible(self, bits):
        return bool((self.bits_blocks(bits).sum(axis=1) == 1).all())

    def decode(self, bits):
        return self.bits_blocks(bits).argmax(axis=1)

    def conflicts(self, col):
        return sum(1 for a, b in self.edges if col[a] == col[b])


def counts_to_bits(inst, key):
    """qiskit counts key (little-endian, physical measure order after layout)
    -> logical bit array. measure_all measures virtual qubit i into clbit i,
    and transpile's final_layout bookkeeping keeps clbit order = virtual order,
    so key reversed gives logical bits directly."""
    s = key.replace(" ", "")[::-1]
    return np.frombuffer(s.encode(), dtype=np.uint8)[:inst.n] - ord("0")


def run_body(inst, t, init_bits, shots, seed):
    """Prepend basis-native X on the physical qubits carrying 1-bits; run."""
    body = inst.circs[t]
    qc = body.copy_empty_like()
    lay = inst.segments[t]["init_layout"]
    for i, b in enumerate(init_bits):
        if b:
            qc.x(qc.qubits[lay[i]])
    qc.compose(body, inplace=True)
    res = SIM.run(qc, shots=shots, seed_simulator=seed).result()
    out = {}
    for k, v in res.get_counts().items():
        kb = counts_to_bits(inst, k)
        out[kb.tobytes()] = out.get(kb.tobytes(), 0) + v
    return out


# ---------------------------------------------------------------------------
# Part A: decode destination / block locality
# ---------------------------------------------------------------------------

def proper_colorings(inst, limit=60):
    out = []
    for col in product(range(inst.K), repeat=inst.v):
        if all(col[a] != col[b] for a, b in inst.edges):
            out.append(np.array(col))
            if len(out) >= limit:
                break
    return out


def part_a(inst):
    rows = []
    cols = proper_colorings(inst)
    segs = range(0, len(inst.chain), max(1, len(inst.chain) // 8))
    for t in segs:
        for ci, col in enumerate(cols[:12]):
            init = inst.coloring_to_bits(col)
            cnt = run_body(inst, t, init, ORACLE_SHOTS, seed=1000 + 7 * t + ci)
            for kb, w in cnt.items():
                bits = np.frombuffer(kb, dtype=np.int8)
                blocks = inst.bits_blocks(bits)
                weights = blocks.sum(axis=1)
                bad = np.flatnonzero(weights != 1)
                ham = int(np.abs(bits - init).sum())
                rows.append(dict(instance=inst.name, seg=t, col_id=ci, w=w,
                                 feasible=bool(len(bad) == 0),
                                 n_bad_blocks=int(len(bad)),
                                 hamming=ham,
                                 w0_blocks=int((weights == 0).sum()),
                                 w2p_blocks=int((weights >= 2).sum())))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part B: f_t decay (purify semantics from the trained seed)
# ---------------------------------------------------------------------------

def part_b(inst, seed=5):
    rows = []
    rng = np.random.default_rng(seed)
    register = {inst.coloring_to_bits(inst.seed_coloring).tobytes(): 256}
    for t in range(len(inst.chain)):
        raw = {}
        for kb, w in register.items():
            cnt = run_body(inst, t, np.frombuffer(kb, dtype=np.int8), int(w),
                           seed=2000 + 13 * t)
            for k2, v in cnt.items():
                raw[k2] = raw.get(k2, 0) + v
        tot = sum(raw.values())
        feas = {k: v for k, v in raw.items()
                if inst.is_feasible(np.frombuffer(k, dtype=np.int8))}
        f = sum(feas.values()) / max(tot, 1)
        rows.append(dict(instance=inst.name, seg=t, f=f, n_survivors=len(feas)))
        if not feas:
            break
        # multinomial reallocation to 256
        keys = list(feas)
        p = np.array([feas[k] for k in keys], dtype=float)
        draw = rng.multinomial(256, p / p.sum())
        register = {k: int(x) for k, x in zip(keys, draw) if x > 0}
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part C: chain arms with cached empirical distributions
# ---------------------------------------------------------------------------

class DistCache:
    def __init__(self, inst):
        self.inst = inst
        self._c = {}

    def dist(self, t, init_key):
        k = (t, init_key)
        if k not in self._c:
            cnt = run_body(self.inst, t, np.frombuffer(init_key, dtype=np.int8),
                           ORACLE_SHOTS, seed=3000 + 31 * t + (hash(init_key) % 97))
            keys = list(cnt)
            p = np.array([cnt[x] for x in keys], dtype=float)
            self._c[k] = (keys, p / p.sum())
        return self._c[k]

    def sample(self, t, init_key, shots, rng):
        keys, p = self.dist(t, init_key)
        draw = rng.multinomial(shots, p)
        return {k: int(x) for k, x in zip(keys, draw) if x > 0}


def decode_synd(inst, bits, rng):
    blocks = inst.bits_blocks(bits.copy())
    weights = blocks.sum(axis=1)
    bad = list(np.flatnonzero(weights != 1))
    if len(bad) > MAX_CORRUPT:
        return None
    col = np.where(weights == 1, blocks.argmax(axis=1), -1)
    lists = []
    for u in bad:
        if weights[u] >= 2:
            cand = list(np.flatnonzero(blocks[u]))
        else:
            cand = list(range(inst.K))
        # conflict elimination against intact neighbors
        cand2 = [c for c in cand
                 if not any(col[w] == c for w in inst.adj[u] if col[w] >= 0)]
        lists.append(cand2 if cand2 else cand)
    best, best_cost = None, None
    order = np.argsort([len(l) for l in lists]) if bad else []

    def bt(i, cur, cost):
        nonlocal best, best_cost
        if best_cost is not None and cost > best_cost:
            return
        if i == len(bad):
            if best_cost is None or cost < best_cost:
                best, best_cost = cur.copy(), cost
            return
        u = bad[order[i]]
        opts = lists[order[i]][:]
        rng.shuffle(opts)
        for c in opts:
            cur[u] = c
            dc = sum(1 for w in inst.adj[u] if cur[w] == c and cur[w] >= 0)
            bt(i + 1, cur, cost + dc)
            cur[u] = -1

    bt(0, col.copy(), 0)
    if best is None:
        return None
    out = col.copy()
    out[out < 0] = 0
    for u in bad:
        out[u] = best[u]
    return out


def run_arm(inst, cache, arm, rng):
    n = inst.n
    register = {inst.coloring_to_bits(inst.seed_coloring).tobytes(): SHOTS_ARM}
    best_seen, proper_seen, f_min, died = np.inf, False, 1.0, -1
    for t in range(len(inst.chain)):
        raw = {}
        for kb, w in register.items():
            if arm == "junk_synd":
                bits = rng.integers(0, 2, size=(int(w), n)).astype(np.int8)
                for row in bits:
                    raw[row.tobytes()] = raw.get(row.tobytes(), 0) + 1
            else:
                for k2, v in cache.sample(t, kb, int(w), rng).items():
                    raw[k2] = raw.get(k2, 0) + v
        tot = sum(raw.values())
        feas, nxt = 0, {}
        for kb, w in raw.items():
            bits = np.frombuffer(kb, dtype=np.int8)
            if inst.is_feasible(bits):
                feas += w
                col = inst.decode(bits)
                cf = inst.conflicts(col)
                best_seen = min(best_seen, cf)
                proper_seen |= (cf == 0)
                nxt[kb] = nxt.get(kb, 0) + w
            elif arm in ("rec_synd", "junk_synd"):
                col = decode_synd(inst, bits, rng)
                if col is not None:
                    nxt[inst.coloring_to_bits(col).tobytes()] = \
                        nxt.get(inst.coloring_to_bits(col).tobytes(), 0) + w
            elif arm == "rec_uniform":
                col = rng.integers(0, inst.K, size=inst.v)
                nxt[inst.coloring_to_bits(col).tobytes()] = \
                    nxt.get(inst.coloring_to_bits(col).tobytes(), 0) + w
        f_min = min(f_min, feas / max(tot, 1))
        if not nxt:
            died = t
            break
        if len(nxt) > MAX_REGISTER:
            top = sorted(nxt.items(), key=lambda kv: -kv[1])[:MAX_REGISTER]
            nxt = dict(top)
        keys = list(nxt)
        p = np.array([nxt[k] for k in keys], dtype=float)
        draw = rng.multinomial(SHOTS_ARM, p / p.sum())
        register = {k: int(x) for k, x in zip(keys, draw) if x > 0}
    return dict(arm=arm, survived=died < 0, death_segment=died, f_min=f_min,
                best_conflicts_seen=(float(best_seen) if np.isfinite(best_seen) else np.inf),
                proper_seen=bool(proper_seen))


def _arm_worker(args):
    """One (arm, gpu) task in a spawned process: run all seeds, return rows.
    Each arm keeps its own DistCache (caches are arm-local anyway: purify and
    rec_* visit different registers)."""
    import os
    name, arm, gpu = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    global SIM
    SIM = make_sim()
    with open("fake_heron_bundle.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    inst = Inst([d for d in bundle["instances"] if d["name"] == name][0])
    cache = DistCache(inst)
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(9_000_000 + hash((inst.name, arm, seed)) % 800_000)
        r = run_arm(inst, cache, arm, rng)
        rows.append(dict(instance=inst.name, seed=seed, **r))
        pd.DataFrame(rows).to_csv(f"fake_heron_arms_{name}_{arm}.partial.csv", index=False)
    return rows


def main():
    import sys as _sys
    only = _sys.argv[1] if len(_sys.argv) > 1 else None
    with open("fake_heron_bundle.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    t00 = time.time()
    insts = [d for d in bundle["instances"] if only is None or d["name"] == only]
    arm_rows = []
    for d in insts:
        inst = Inst(d)
        tag = inst.name
        print(f"[{time.time()-t00:6.0f}s] {tag} part A", flush=True)
        part_a(inst).to_csv(f"fake_heron_dd_{tag}.csv", index=False)
        print(f"[{time.time()-t00:6.0f}s] {tag} part B", flush=True)
        part_b(inst).to_csv(f"fake_heron_ft_{tag}.csv", index=False)
        print(f"[{time.time()-t00:6.0f}s] {tag} part C (arms across 3 GPUs)", flush=True)
        from multiprocessing import get_context
        ctx = get_context("spawn")
        tasks = [(tag, arm, i % 3) for i, arm in
                 enumerate(("purify", "rec_synd", "rec_uniform", "junk_synd"))]
        with ctx.Pool(3) as pool:
            for rows in pool.imap_unordered(_arm_worker, tasks):
                arm_rows.extend(rows)
                pd.DataFrame(arm_rows).to_csv(f"fake_heron_arms_{tag}.csv", index=False)
                print(f"[{time.time()-t00:6.0f}s]   arm batch done ({rows[0]['arm']})",
                      flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
