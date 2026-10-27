"""Pilot B — purification vs recovery crossover.

Head-to-head on the identical noisy segmented executor:
  arm 'purify'          : upstream Rasengan (hard filter; chain dies if no survivors)
  arm 'recover_nearest' : decode ALL outcomes to nearest-feasible, reinject
  arm 'recover_map'     : MAP decode with running feasible prior, reinject
  arm 'junk_control'    : recover_nearest applied to a matched JUNK chain — the
                          per-segment raw counts are replaced by uniform random
                          strings (same shot count), so the decoder sees ensembles
                          with no quantum signal. Anti-triviality control (A0).
  arm 'feas_control'    : best-of-S uniform feasible samples at the TOTAL shot
                          budget of the chain (classical skyline for projection
                          triviality).

Primary endpoints per (instance, lam, seed, arm):
  survived (bool), death_segment, best ARG of final register, p(x*) in final
  register, expected ARG of final register, total shots used.
Noise grid is denser than Pilot A around the expected cliff.
"""
import sys, os, glob, time
import _paths  # noqa: F401  (repo-relative sys.path setup)

import numpy as np
import pandas as pd
import dill

from h1_pilot_lib import (LoadedOracle, run_segmented_oracle, decode_counts,
                          bits_to_str, SegRunResult)

SHOTS_LEVELS = [64, 256, 1024]
SEEDS = list(range(8))


def junk_chain(pack, n_segments, shots, rng):
    """Matched control: same chain length/shot budget, but every segment's raw
    output is uniform random strings. Decoder = nearest-feasible reinjection."""
    res = SegRunResult()
    n = pack.n
    register = {bits_to_str(np.asarray(pack.feasible_seed, dtype=int)): shots}
    for t in range(n_segments):
        X = rng.integers(0, 2, size=(shots, n), dtype=np.int8)
        raw = {}
        for row in X:
            k = bits_to_str(row)
            raw[k] = raw.get(k, 0) + 1
        res.total_shots_used += shots
        tot = sum(raw.values())
        feas = {k: v for k, v in raw.items() if pack.is_feasible(k)}
        res.f_per_segment.append(sum(feas.values()) / tot)
        for k in feas:
            e = pack.energy(k)
            if e < res.best_energy_seen:
                res.best_energy_seen, res.best_feasible_seen = e, k
        register = decode_counts(pack, raw, "nearest", rng)
        if not register:
            res.death_segment = t
            break
    res.final_register = register
    return res


def endpoint_row(pack, res: SegRunResult):
    reg = res.final_register
    tot = sum(reg.values())
    row = dict(
        survived=res.death_segment < 0,
        death_segment=res.death_segment,
        total_shots=res.total_shots_used,
        f_min=min(res.f_per_segment) if res.f_per_segment else np.nan,
        f_last=res.f_per_segment[-1] if res.f_per_segment else np.nan,
    )
    feas_keys = [k for k in reg if pack.is_feasible(k)]
    if tot > 0 and feas_keys:
        args = np.array([pack.arg(k) for k in feas_keys])
        w = np.array([reg[k] for k in feas_keys], dtype=float)
        row["best_arg"] = float(args.min())
        row["exp_arg"] = float((args * w).sum() / w.sum())
        row["p_opt"] = float(reg.get(pack.x_star, 0) / tot)
        row["hit_opt"] = bool(pack.x_star in reg)
    else:
        row["best_arg"] = np.nan
        row["exp_arg"] = np.nan
        row["p_opt"] = 0.0
        row["hit_opt"] = False
    # best feasible EVER seen during the run (any protocol observes segments)
    row["best_arg_seen"] = (abs((res.best_energy_seen - pack.e_star) /
                                (pack.e_star if pack.e_star else 1.0))
                            if res.best_feasible_seen else np.nan)
    row["hit_opt_seen"] = res.best_feasible_seen == pack.x_star
    return row


def feasible_skyline(pack, total_shots, rng):
    """Best-of-N uniform feasible sampling at matched total shot budget."""
    idx = rng.integers(0, len(pack.F), size=total_shots)
    E = pack.F_energies[idx]
    best = float(E.min())
    return dict(survived=True, death_segment=-1, total_shots=total_shots,
                f_min=1.0, f_last=1.0,
                best_arg=abs((best - pack.e_star) / (pack.e_star or 1.0)),
                exp_arg=abs((E.mean() - pack.e_star) / (pack.e_star or 1.0)),
                p_opt=float((E == pack.e_star).mean()),
                hit_opt=bool((E == pack.e_star).any()),
                best_arg_seen=abs((best - pack.e_star) / (pack.e_star or 1.0)),
                hit_opt_seen=bool((E == pack.e_star).any()))


def main():
    with open("packs/instance_packs.dill", "rb") as fh:
        packs = dill.load(fh)
    rows = []
    t0 = time.time()
    for name, pack in packs.items():
        oracle_files = sorted(glob.glob(f"packs/oracles/{name}_lam*.dill"))
        lams = sorted(float(f.split("lam")[-1].replace(".dill", "")) for f in oracle_files)
        for lam in lams:
            with open(f"packs/oracles/{name}_lam{lam:g}.dill", "rb") as fh:
                orc = LoadedOracle(pack, dill.load(fh))
            n_seg = orc.n_segments
            for shots in SHOTS_LEVELS:
                for seed in SEEDS:
                    rng = np.random.default_rng(10_000 + 97 * seed + int(lam * 7) + shots)
                    for arm in ("purify", "recover_nearest", "recover_map"):
                        r = run_segmented_oracle(pack, orc, shots, arm, rng)
                        rows.append(dict(instance=name, lam=lam, shots=shots, seed=seed,
                                         arm=arm, n_segments=n_seg,
                                         **endpoint_row(pack, r)))
                    # junk control (no quantum execution needed)
                    rj = junk_chain(pack, n_seg, shots, rng)
                    rows.append(dict(instance=name, lam=lam, shots=shots, seed=seed,
                                     arm="junk_control", n_segments=n_seg,
                                     **endpoint_row(pack, rj)))
                    # feasible-uniform skyline at matched budget
                    rows.append(dict(instance=name, lam=lam, shots=shots, seed=seed,
                                     arm="feas_control", n_segments=n_seg,
                                     **feasible_skyline(pack, shots * n_seg, rng)))
            print(f"[{time.time()-t0:7.0f}s] {name} lam={lam:g} done", flush=True)
        pd.DataFrame(rows).to_csv("results/pilot_b_partial.csv", index=False)
    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/pilot_b.csv", index=False)
    print("wrote results/pilot_b.csv", df.shape)


if __name__ == "__main__":
    main()
