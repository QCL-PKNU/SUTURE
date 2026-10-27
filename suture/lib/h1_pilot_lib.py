"""Shared library for the H1-R pilots (Rasengan purification vs bitstring recovery).

Mechanism-faithful reimplementation of the segmented Rasengan executor, built on
the upstream components (LcboModel constraint compilation, null-space
`driver_bitstr`, greedy Hamiltonian simplification, `driver_component` circuit
synthesis), with three deliberate substitutions, each documented:

1. Chain construction + pruning is done with the exact classical equivalent of
   the upstream explorer (reachable-set expansion over F: R' = R U (R+u) U (R-u)
   restricted to binary strings), instead of 51200-shot quantum simulations.
2. Shot reallocation between segments uses a multinomial draw instead of
   floor(S*w), so rare survivor states are not silently dropped.
3. Training uses scipy COBYLA on the noise-free segmented executor.

Conventions (verified against the cloned implementation):
- counts keys are qubit-order strings (q0 leftmost): circuits measure qubit i
  -> classical bit n-1-i.
- feasibility: for every row [c | b] of lin_constr_mtx, dot(bits, c) == b.
- transition circuits are synthesized on num_qubits + 2 qubits
  (mcx_mode="constant") with exactly one time Parameter each; they are
  transpiled ONCE (parametrized) to the core basis and bound per execution.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

CORE_BASIS = ["measure", "cx", "id", "rz", "sx", "x"]

# IBM Heron-like base rates (paper Sec. 5.5: 1q 0.035%, 2q 0.875%; readout ~1%)
BASE_P1, BASE_P2, BASE_PRO = 3.5e-4, 8.75e-3, 1.0e-2


# ---------------------------------------------------------------------------
# Noise models
# ---------------------------------------------------------------------------

def make_noise_model(lam_gate: float, lam_readout: float) -> NoiseModel | None:
    """Depolarizing gate noise scaled by lam_gate; readout flips scaled by lam_readout."""
    if lam_gate <= 0 and lam_readout <= 0:
        return None
    nm = NoiseModel(basis_gates=CORE_BASIS)
    if lam_gate > 0:
        p1 = min(BASE_P1 * lam_gate, 0.5)
        p2 = min(BASE_P2 * lam_gate, 0.75)
        nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["id", "rz", "sx", "x"])
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    if lam_readout > 0:
        pr = min(BASE_PRO * lam_readout, 0.5)
        nm.add_all_qubit_readout_error(ReadoutError([[1 - pr, pr], [pr, 1 - pr]]))
    return nm


class NoisyRunner:
    """Transpile-to-core-basis + noisy Aer execution with a per-body transpile cache."""

    def __init__(self, noise_model: NoiseModel | None, seed: int = 0):
        self.sim = AerSimulator(noise_model=noise_model, seed_simulator=seed)
        self._cache: dict[int, QuantumCircuit] = {}

    def run_counts(self, qc: QuantumCircuit, shots: int, cache_key=None,
                   pretranspiled: bool = False) -> dict[str, int]:
        if shots <= 0:
            return {}
        if pretranspiled:
            # circuit is already in CORE_BASIS (X init + pre-transpiled body):
            # skip the transpiler entirely — this is the hot path.
            tqc = qc
        elif cache_key is not None and cache_key in self._cache:
            tqc = self._cache[cache_key]
        else:
            # optimization_level=0: do not let the transpiler cancel gates and
            # silently change the effective noise.
            tqc = transpile(qc, self.sim, basis_gates=CORE_BASIS, optimization_level=0)
            if cache_key is not None:
                self._cache[cache_key] = tqc
        res = self.sim.run(tqc, shots=shots).result()
        return dict(res.get_counts())

    def run_batch(self, circuits: list[QuantumCircuit], shots_list: list[int]) -> list[dict]:
        """Run many pre-transpiled circuits; group by shot count into batched jobs."""
        out = [dict() for _ in circuits]
        by_shots: dict[int, list[int]] = {}
        for i, s in enumerate(shots_list):
            if s > 0:
                by_shots.setdefault(int(s), []).append(i)
        for s, idxs in by_shots.items():
            res = self.sim.run([circuits[i] for i in idxs], shots=s).result()
            for j, i in enumerate(idxs):
                out[i] = dict(res.get_counts(j)) if len(idxs) > 1 else dict(res.get_counts())
        return out


# ---------------------------------------------------------------------------
# Instance package: feasibility, energies, decoding
# ---------------------------------------------------------------------------

@dataclass
class InstancePack:
    name: str
    feasible_seed: np.ndarray       # initial feasible state (int, len n)
    chain_templates: list           # transpiled parametrized 1-transition circuits (with measure)
    chain_u: list                   # the transition vector index used at each chain position
    params: np.ndarray              # trained transition times (len == len(chain_templates))
    F: np.ndarray                   # (|F|, n) int8 feasible strings
    F_strings: list[str]
    F_energies: np.ndarray
    x_star: str
    e_star: float
    e_worst: float
    n: int
    constr: np.ndarray              # lin_constr_mtx (int)

    def is_feasible(self, key: str) -> bool:
        return key in self._fset

    def __post_init__(self):
        self._fset = set(self.F_strings)
        self._energy = dict(zip(self.F_strings, self.F_energies))
        self._fidx = {s: i for i, s in enumerate(self.F_strings)}

    def energy(self, key: str) -> float:
        return self._energy[key]

    def arg(self, key: str) -> float:
        return abs((self.energy(key) - self.e_star) / (self.e_star if self.e_star else 1.0))


def enumerate_feasible(constr: np.ndarray, n: int) -> np.ndarray:
    """All x in {0,1}^n with constr[:, :n] @ x == constr[:, n]. Vectorized, n <= 22."""
    xs = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)
    lhs = xs @ constr[:, :-1].T.astype(np.int64)
    ok = np.all(lhs == constr[:, -1][None, :].astype(np.int64), axis=1)
    return xs[ok]


# ---------------------------------------------------------------------------
# Chain construction (classical equivalent of the upstream explorer/pruning)
# ---------------------------------------------------------------------------

def build_chain(basis: np.ndarray, seed: np.ndarray, feasible_set: set[tuple],
                max_rounds: int | None = None) -> list[int]:
    """Greedy pruned transition chain: iterate the basis in rounds, keeping only
    transitions that expand the reachable set. Mirrors upstream pruning
    (rasengan_segmented.py lines 118-140) exactly, but on exact sets.
    """
    m = len(basis)
    max_rounds = max_rounds or m
    R = {tuple(seed.tolist())}
    chain = []
    for _ in range(max_rounds):
        grew_this_round = False
        for i in range(m):
            u = basis[i].astype(int)
            new = set()
            for y in R:
                ya = np.array(y)
                for cand in (ya + u, ya - u):
                    if cand.min() >= 0 and cand.max() <= 1:
                        tc = tuple(cand.tolist())
                        if tc in feasible_set and tc not in R:
                            new.add(tc)
            if new:
                R |= new
                chain.append(i)
                grew_this_round = True
            if len(R) == len(feasible_set):
                return chain
        if not grew_this_round:
            break
    return chain


def bits_to_str(row) -> str:
    return "".join("1" if b else "0" for b in row)


def str_to_bits(key: str) -> np.ndarray:
    return np.frombuffer(key.encode(), dtype=np.uint8) - ord("0")


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------

def decode_counts(pack: InstancePack, counts: dict[str, int], mode: str,
                  rng: np.random.Generator, prior: np.ndarray | None = None,
                  p_flip: float = 0.05) -> dict[str, int]:
    """Map every measured string to a feasible string.

    mode='nearest': argmin Hamming distance to F (ties random).
    mode='map':     argmax log prior(y) + d_H(x,y) * log(p/(1-p)); prior over F.
    Feasible strings pass through unchanged.
    """
    out: dict[str, int] = {}
    if not counts:
        return out
    keys = list(counts.keys())
    X = np.stack([str_to_bits(k) for k in keys]).astype(np.int8)
    D = (X[:, None, :] != pack.F[None, :, :]).sum(axis=2)  # (S_unique, |F|)
    if mode == "nearest":
        score = -D.astype(float)
    elif mode == "map":
        if prior is None:
            prior = np.ones(len(pack.F))
        logp = np.log(prior / prior.sum())
        llr = np.log(max(p_flip, 1e-6) / max(1 - p_flip, 1e-6))
        score = logp[None, :] + D * llr
    else:
        raise ValueError(mode)
    for i, k in enumerate(keys):
        if k in pack._fset:
            tgt = k
        else:
            s = score[i]
            best = np.flatnonzero(s == s.max())
            tgt = pack.F_strings[int(rng.choice(best))]
        out[tgt] = out.get(tgt, 0) + counts[k]
    return out


# ---------------------------------------------------------------------------
# Distribution oracle
# ---------------------------------------------------------------------------
# Simulating each (segment, init) pair per arm/seed is wasteful: the noisy
# output distribution q_t(.|x) is a fixed object. We estimate it ONCE with
# ORACLE_SHOTS shots per (segment t, feasible init x, noise scale), then every
# arm/seed/shot-level draws multinomial samples from the empirical estimate.
# Approximation (documented in the report): tail outcomes with true probability
# < ~1/ORACLE_SHOTS are truncated; registers are restricted to feasible inits,
# which is exact for purify/recover arms (their registers are feasible by
# construction) and means 'raw' reinjection from infeasible strings is not
# simulated (raw counts are still available, pooled from the oracle itself).

class SegmentOracle:
    def __init__(self, pack, runner: NoisyRunner, bodies: list, oracle_shots: int = 2048):
        self.pack = pack
        self.bodies = bodies
        self.oracle_shots = oracle_shots
        self.runner = runner
        self._q: dict[tuple[int, str], tuple[list[str], np.ndarray]] = {}

    def build(self, verbose: bool = False):
        import time as _t
        n = self.pack.n
        for t, body in enumerate(self.bodies):
            t0 = _t.time()
            circuits = []
            for init in self.pack.F_strings:
                qc = QuantumCircuit(body.num_qubits, n)
                for i, ch in enumerate(init):
                    if ch == "1":
                        qc.x(i)
                qc.compose(body, inplace=True)
                circuits.append(qc)
            counts_list = self.runner.run_batch(circuits, [self.oracle_shots] * len(circuits))
            for init, counts in zip(self.pack.F_strings, counts_list):
                keys = list(counts.keys())
                p = np.array([counts[k] for k in keys], dtype=float)
                self._q[(t, init)] = (keys, p / p.sum())
            if verbose:
                print(f"  oracle seg {t}: {len(circuits)} inits, {_t.time()-t0:.1f}s", flush=True)

    def sample(self, t: int, init: str, shots: int, rng) -> dict[str, int]:
        keys, p = self._q[(t, init)]
        draw = rng.multinomial(shots, p)
        return {k: int(v) for k, v in zip(keys, draw) if v > 0}

    def f_of(self, t: int, init: str) -> float:
        keys, p = self._q[(t, init)]
        return float(sum(pi for k, pi in zip(keys, p) if self.pack.is_feasible(k)))


class LoadedOracle:
    """Oracle rehydrated from a dill payload (no runner/bodies needed)."""
    def __init__(self, pack, payload):
        self.pack = pack
        self._q = payload["q"]
        self.n_segments = payload["n_segments"]
        self.bodies = [None] * self.n_segments  # len() compatibility

    def sample(self, t, init, shots, rng):
        keys, p = self._q[(t, init)]
        draw = rng.multinomial(shots, p)
        return {k: int(v) for k, v in zip(keys, draw) if v > 0}

    def f_of(self, t, init):
        keys, p = self._q[(t, init)]
        return float(sum(pi for k, pi in zip(keys, p) if self.pack.is_feasible(k)))


def run_segmented_oracle(pack, oracle: SegmentOracle, shots: int, mode: str, rng,
                         map_p: float = 0.05, keep_raw: bool = False) -> "SegRunResult":
    """Same protocol logic as run_segmented, but sampling from the oracle.
    Register states are always feasible strings (exact for purify/recover arms)."""
    res = SegRunResult()
    seed_str = bits_to_str(np.asarray(pack.feasible_seed, dtype=int))
    register = {seed_str: shots}
    prior = np.ones(len(pack.F)) * 0.5

    for t in range(len(oracle.bodies)):
        states = list(register.keys())
        w = np.array([register[s] for s in states], dtype=float)
        alloc = rng.multinomial(shots, w / w.sum())
        raw: dict[str, int] = {}
        for init, a in zip(states, alloc):
            if a == 0:
                continue
            for k, v in oracle.sample(t, init, int(a), rng).items():
                raw[k] = raw.get(k, 0) + v
        res.total_shots_used += int(alloc.sum())
        if keep_raw:
            res.raw_segments.append(raw)

        tot = sum(raw.values())
        feas = {k: v for k, v in raw.items() if pack.is_feasible(k)}
        res.f_per_segment.append(sum(feas.values()) / tot if tot else 0.0)

        for k in feas:
            e = pack.energy(k)
            if e < res.best_energy_seen:
                res.best_energy_seen, res.best_feasible_seen = e, k
        for k, v in feas.items():
            prior[pack._fidx[k]] += v

        if mode == "purify":
            register = feas
        elif mode == "recover_nearest":
            register = decode_counts(pack, raw, "nearest", rng)
        elif mode == "recover_map":
            register = decode_counts(pack, raw, "map", rng, prior=prior, p_flip=map_p)
        else:
            raise ValueError(mode)

        if not register:
            res.death_segment = t
            break

    res.final_register = register
    return res


# ---------------------------------------------------------------------------
# Instrumented segmented executor
# ---------------------------------------------------------------------------

@dataclass
class SegRunResult:
    f_per_segment: list[float] = field(default_factory=list)
    raw_segments: list[dict] = field(default_factory=list)     # raw counts per segment
    death_segment: int = -1                                     # -1 = survived all
    final_register: dict = field(default_factory=dict)
    best_feasible_seen: str | None = None
    best_energy_seen: float = np.inf
    total_shots_used: int = 0


def segment_bodies(pack: InstancePack, seg_size: int) -> list[QuantumCircuit]:
    """Compose bound transition templates into per-segment bodies (no init, with measure)."""
    n, bodies = pack.n, []
    T = len(pack.chain_templates)
    for start in range(0, T, seg_size):
        end = min(start + seg_size, T)
        body = None
        for t in range(start, end):
            circ = pack.chain_templates[t].assign_parameters([pack.params[t]])
            body = circ if body is None else body.compose(circ)
        body.measure(range(n), list(range(n))[::-1])
        bodies.append(body)
    return bodies


def run_segmented(pack: InstancePack, runner: NoisyRunner, shots: int, mode: str,
                  rng: np.random.Generator, seg_size: int = 1, map_p: float = 0.05,
                  keep_raw: bool = True, bodies: list | None = None) -> SegRunResult:
    """Execute the trained segmented Rasengan chain under noise.

    mode: 'purify'  - keep exact-feasible only (upstream behavior); death if none survive
          'recover_nearest' / 'recover_map' - decode ALL outcomes to feasible, reinject
          'raw'     - no filter: measured strings (feasible or not) are reinjected as-is
    """
    res = SegRunResult()
    seed_str = bits_to_str(np.asarray(pack.feasible_seed, dtype=int))
    register = {seed_str: shots}
    n = pack.n
    if bodies is None:
        bodies = segment_bodies(pack, seg_size)
    prior = np.ones(len(pack.F)) * 0.5  # Laplace-smoothed running prior over F

    for t, body in enumerate(bodies):
        # --- run the segment from each register state, shots via multinomial ---
        states = list(register.keys())
        w = np.array([register[s] for s in states], dtype=float)
        alloc = rng.multinomial(shots, w / w.sum())
        raw: dict[str, int] = {}
        circuits, shots_list = [], []
        for init, a in zip(states, alloc):
            if a == 0:
                continue
            qc = QuantumCircuit(body.num_qubits, n)
            for i, ch in enumerate(init):
                if ch == "1":
                    qc.x(i)
            qc.compose(body, inplace=True)
            circuits.append(qc)
            shots_list.append(int(a))
        for counts in runner.run_batch(circuits, shots_list):
            for k, v in counts.items():
                raw[k] = raw.get(k, 0) + v
        res.total_shots_used += int(alloc.sum())
        if keep_raw:
            res.raw_segments.append(raw)

        tot = sum(raw.values())
        feas = {k: v for k, v in raw.items() if pack.is_feasible(k)}
        nfeas = sum(feas.values())
        res.f_per_segment.append(nfeas / tot if tot else 0.0)

        # update best-feasible-seen (generous bookkeeping for all protocols)
        for k in feas:
            e = pack.energy(k)
            if e < res.best_energy_seen:
                res.best_energy_seen, res.best_feasible_seen = e, k

        # update running prior with observed feasible mass
        for k, v in feas.items():
            prior[pack._fidx[k]] += v

        # --- protocol-specific register update ---
        if mode == "purify":
            register = feas
        elif mode == "recover_nearest":
            register = decode_counts(pack, raw, "nearest", rng)
        elif mode == "recover_map":
            register = decode_counts(pack, raw, "map", rng, prior=prior, p_flip=map_p)
        elif mode == "raw":
            register = raw
        else:
            raise ValueError(mode)

        if not register:
            res.death_segment = t
            break

    res.final_register = register
    return res
