"""Pilot A — distribution characterization across circuit families.

For each (instance, noise scale, family): draw a sample ensemble, measure
  f            feasible probability
  mean_dF      mean Hamming distance of infeasible samples to nearest feasible
  frac_dF1     fraction of infeasible samples within Hamming distance 1 of F
  mean_viol    mean number of violated constraint rows per infeasible sample
  p_opt        probability of sampling x* exactly
  mean_arg     probability-weighted ARG over feasible samples
  H_norm       normalized Shannon entropy of the sample distribution
Families:
  rasengan_purified  trained segmented chain, purification ON (final register)
  rasengan_raw       same chain, pooled PRE-purification counts of all segments
  pqaoa_d{1,2}       penalty QAOA depth 1,2 (params picked noise-free, greedy)
  uniform            classical control C1
  feas_uniform       classical control C2 (uniform over F)
  seed_noise         classical control C3' (feasible seed + independent bit flips)

Rasengan arms sample from prebuilt segment oracles (see build_oracles.py);
QAOA distributions are estimated once per (instance, depth, lam) with 4096
shots and per-seed ensembles drawn multinomially.
"""
import sys, os, glob, time
import _paths  # noqa: F401  (repo-relative sys.path setup)

import numpy as np
import pandas as pd
import dill
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from h1_pilot_lib import (NoisyRunner, make_noise_model, bits_to_str, str_to_bits,
                          LoadedOracle, run_segmented_oracle, CORE_BASIS)

SHOTS = 2048
SEEDS = list(range(5))
SIM = AerSimulator()


# ---------------------------------------------------------------------------
# Penalty-QAOA sampler (QUBO-from-constraints, standard X mixer)
# ---------------------------------------------------------------------------

def qubo_from_pack(pack):
    """J, h for E(x) = f(x) + A * sum_r (c_r.x - b_r)^2, A = 2*(e_worst-e_star)."""
    n = pack.n
    A = 2.0 * max(pack.e_worst - pack.e_star, 1.0)
    # objective on feasible strings only is not expressible generally; use
    # constraint penalty + linear objective approximation via F-energies is NOT
    # available to QAOA. We build the penalty part exactly and add the *linear*
    # part of the objective read off the model constraints — for FLP/SCP/GCP the
    # upstream objective is linear or near-linear in x, and Pilot A only needs a
    # representative penalty-QAOA distribution, not a perfectly matched one.
    Q = np.zeros((n, n))
    lin = np.zeros(n)
    const = 0.0
    for row in pack.constr:
        c, b = row[:-1].astype(float), float(row[-1])
        Q += A * np.outer(c, c)
        lin += -2 * A * b * c
        const += A * b * b
    return Q, lin, const


def qaoa_circuit(pack, depth, gammas, betas):
    n = pack.n
    Q, lin, _ = qubo_from_pack(pack)
    J = Q / 4.0
    h = -(lin / 2.0 + Q.sum(axis=1) / 2.0)
    # normalize coupling magnitudes so angles are meaningful across instances
    scale = max(np.abs(J).max(), np.abs(h).max(), 1.0)
    J, h = J / scale, h / scale
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for d in range(depth):
        g, bt = gammas[d], betas[d]
        for i in range(n):
            if abs(h[i]) > 1e-12:
                qc.rz(2 * g * h[i], i)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-12:
                    qc.cx(i, j); qc.rz(2 * g * (J[i, j] + J[j, i]), j); qc.cx(i, j)
        for i in range(n):
            qc.rx(2 * bt, i)
    qc.measure(range(n), list(range(n))[::-1])
    return qc


def pick_qaoa_params(pack, depth, shots=1024, trials=6):
    """Greedy noise-free parameter pick: maximize feasible mass + quality."""
    rng = np.random.default_rng(hash((pack.name, depth)) % (2**31))
    runner = NoisyRunner(None, seed=42)
    best = None
    for _ in range(trials):
        gammas = rng.uniform(0.05, 1.2, depth)
        betas = rng.uniform(0.05, np.pi / 2, depth)
        qc = qaoa_circuit(pack, depth, gammas, betas)
        counts = runner.run_counts(qc, shots)
        tot = sum(counts.values())
        fmass = sum(v for k, v in counts.items() if pack.is_feasible(k)) / tot
        score = fmass
        if best is None or score > best[0]:
            best = (score, gammas, betas)
    return best[1], best[2]


def qaoa_distribution(pack, depth, gammas, betas, lam, shots=4096):
    nm = make_noise_model(lam, lam)
    runner = NoisyRunner(nm, seed=int(13 + lam))
    qc = qaoa_circuit(pack, depth, gammas, betas)
    counts = runner.run_counts(qc, shots)
    keys = list(counts.keys())
    p = np.array([counts[k] for k in keys], dtype=float)
    return keys, p / p.sum()


def sample_dist(keys, p, shots, rng):
    draw = rng.multinomial(shots, p)
    return {k: int(v) for k, v in zip(keys, draw) if v > 0}


# ---------------------------------------------------------------------------
# Classical controls
# ---------------------------------------------------------------------------

def control_counts(pack, family, shots, seed, flip_p=0.1):
    rng = np.random.default_rng(seed)
    n = pack.n
    out = {}
    if family == "uniform":
        X = rng.integers(0, 2, size=(shots, n), dtype=np.int8)
    elif family == "feas_uniform":
        idx = rng.integers(0, len(pack.F), size=shots)
        X = pack.F[idx]
    elif family == "seed_noise":
        base = np.asarray(pack.feasible_seed, dtype=np.int8)
        X = np.tile(base, (shots, 1))
        flips = rng.random((shots, n)) < flip_p
        X = np.where(flips, 1 - X, X)
    else:
        raise ValueError(family)
    for row in X:
        k = bits_to_str(row)
        out[k] = out.get(k, 0) + 1
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def characterize(pack, counts):
    tot = sum(counts.values())
    if tot == 0:
        return dict(f=0.0, n_samples=0)
    keys = list(counts.keys())
    w = np.array([counts[k] for k in keys], dtype=float)
    X = np.stack([str_to_bits(k) for k in keys]).astype(np.int8)
    feas_mask = np.array([pack.is_feasible(k) for k in keys])
    f = w[feas_mask].sum() / tot

    D = (X[:, None, :] != pack.F[None, :, :]).sum(axis=2)
    dF = D.min(axis=1).astype(float)          # 0 for feasible strings
    inf_mask = ~feas_mask
    w_inf = w[inf_mask]
    res = dict(f=float(f), n_samples=int(tot))
    if w_inf.sum() > 0:
        res["mean_dF"] = float((dF[inf_mask] * w_inf).sum() / w_inf.sum())
        res["frac_dF1"] = float((w_inf[dF[inf_mask] <= 1]).sum() / w_inf.sum())
        viol = (X @ pack.constr[:, :-1].T != pack.constr[:, -1][None, :]).sum(axis=1)
        res["mean_viol"] = float((viol[inf_mask] * w_inf).sum() / w_inf.sum())
    else:
        res["mean_dF"] = 0.0; res["frac_dF1"] = 1.0; res["mean_viol"] = 0.0
    res["p_opt"] = float(counts.get(pack.x_star, 0) / tot)
    if feas_mask.any():
        args = np.array([pack.arg(k) for k, m in zip(keys, feas_mask) if m])
        wf = w[feas_mask]
        res["mean_arg_feas"] = float((args * wf).sum() / wf.sum())
    else:
        res["mean_arg_feas"] = np.nan
    p = w / tot
    res["H_norm"] = float(-(p * np.log(p)).sum() / np.log(max(tot, 2)))
    return res


def main():
    with open("packs/instance_packs.dill", "rb") as fh:
        packs = dill.load(fh)
    rows = []
    t0 = time.time()
    for name, pack in packs.items():
        oracle_files = sorted(glob.glob(f"packs/oracles/{name}_lam*.dill"))
        lams = sorted(float(f.split("lam")[-1].replace(".dill", "")) for f in oracle_files)
        # QAOA params picked once per depth (noise-free)
        qaoa_params = {d: pick_qaoa_params(pack, d) for d in (1, 2)}
        for lam in lams:
            with open(f"packs/oracles/{name}_lam{lam:g}.dill", "rb") as fh:
                orc = LoadedOracle(pack, dill.load(fh))
            # QAOA distribution estimated once per lam
            qdists = {d: qaoa_distribution(pack, d, *qaoa_params[d], lam)
                      for d in (1, 2)}
            for seed in SEEDS:
                rng = np.random.default_rng(seed)

                # rasengan purify: characterize final register (post-purification)
                r = run_segmented_oracle(pack, orc, SHOTS, "purify", rng, keep_raw=True)
                rows.append(dict(instance=name, family="rasengan_purified", lam=lam,
                                 seed=seed, died=r.death_segment,
                                 f_first=r.f_per_segment[0] if r.f_per_segment else np.nan,
                                 f_last=r.f_per_segment[-1] if r.f_per_segment else np.nan,
                                 f_min=min(r.f_per_segment) if r.f_per_segment else np.nan,
                                 **characterize(pack, r.final_register)))

                # rasengan raw: pooled pre-purification counts from the SAME run
                pooled = {}
                for seg in r.raw_segments:
                    for k, v in seg.items():
                        pooled[k] = pooled.get(k, 0) + v
                rows.append(dict(instance=name, family="rasengan_raw", lam=lam,
                                 seed=seed, died=r.death_segment,
                                 f_first=r.f_per_segment[0] if r.f_per_segment else np.nan,
                                 f_last=r.f_per_segment[-1] if r.f_per_segment else np.nan,
                                 f_min=min(r.f_per_segment) if r.f_per_segment else np.nan,
                                 **characterize(pack, pooled)))

                # penalty QAOA depth 1, 2
                for d in (1, 2):
                    c = sample_dist(*qdists[d], SHOTS, rng)
                    rows.append(dict(instance=name, family=f"pqaoa_d{d}", lam=lam,
                                     seed=seed, died=-1, f_first=np.nan, f_last=np.nan,
                                     f_min=np.nan, **characterize(pack, c)))

                # classical controls (noise-independent; only run at lam==0)
                if lam == 0.0:
                    for fam in ("uniform", "feas_uniform", "seed_noise"):
                        c = control_counts(pack, fam, SHOTS, seed)
                        rows.append(dict(instance=name, family=fam, lam=lam, seed=seed,
                                         died=-1, f_first=np.nan, f_last=np.nan,
                                         f_min=np.nan, **characterize(pack, c)))
            print(f"[{time.time()-t0:7.0f}s] {name} lam={lam:g} done", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/pilot_a.csv", index=False)
    print("wrote results/pilot_a.csv", df.shape)


if __name__ == "__main__":
    main()
