"""profile_structure_eval — T7 metrics for the evaluation packs.

Same metrics as profile_structure.py, but feasible strings are sampled by a
null-space random walk from the seed (basis moves that keep bits in {0,1};
200-move burn-in, 20 moves between samples) instead of enumerating F —
family-agnostic and valid at any n. d_min(F) is reported over the sampled
set (upper bound; exact d_min needs enumeration, available only for packs
that kept F_bits).
Output: results/structural_metrics_eval.csv
"""
import sys, os, pickle
import _paths  # noqa: F401  (repo-relative sys.path setup)
import numpy as np
import pandas as pd

from repair_generic import syndrome, repair


def walk_sample(d, rng, n_samples=400, burn=200, gap=20):
    base = [cv for cv in d["chain_vecs"][:len(set(map(bytes, map(np.ndarray.tobytes, d["chain_vecs"]))))]]
    # unique basis vectors
    uniq, seen = [], set()
    for cv in d["chain_vecs"]:
        k = cv.tobytes()
        if k not in seen:
            seen.add(k); uniq.append(cv.astype(np.int64))
    x = d["seed_bits"].astype(np.int64).copy()
    out = []
    steps = burn + n_samples * gap
    for t in range(steps):
        u = uniq[rng.integers(len(uniq))]
        s = 1 if rng.random() < 0.5 else -1
        y = x + s * u
        if y.min() >= 0 and y.max() <= 1:
            x = y
        if t >= burn and (t - burn) % gap == 0:
            out.append(x.astype(np.int8).copy())
    return out


def profile(d, rng, n_trials=3000):
    C = np.asarray(d["constr"], dtype=np.int64)
    A, n = C[:, :-1], d["n"]
    if "F_bits" in d and len(d["F_bits"]) <= 4096:
        F = d["F_bits"]; f_src = "enum"
    else:
        F = walk_sample(d, rng); f_src = "walk"
    Fset = {fb.tobytes() for fb in F}
    row_sup = [np.flatnonzero(A[r]) for r in range(A.shape[0])]
    FA = np.array(F, dtype=np.int8)
    idx = rng.choice(len(FA), min(len(FA), 400), replace=False)
    FA_s = FA[idx]
    dm = min((int(np.abs(FA_s[i] - FA_s[j]).sum())
              for i in range(len(FA_s)) for j in range(i + 1, len(FA_s))
              if np.any(FA_s[i] != FA_s[j])), default=n)

    silent = 0; Ls, Rs, invRs = [], [], []
    rec_ok = 0
    for t in range(n_trials):
        x = F[rng.integers(len(F))]
        i = rng.integers(n)
        y = x.copy(); y[i] = 1 - y[i]
        syn = syndrome(C, y)
        if len(syn) == 0:
            silent += 1
            continue
        sus = sorted(set(int(j) for r in syn for j in row_sup[r]))
        Ls.append(len(sus))
        R = 0
        for j in sus:
            z = y.copy(); z[j] = 1 - z[j]
            if len(syndrome(C, z)) == 0:
                R += 1
        Rs.append(R); invRs.append(1.0 / max(R, 1))
        out = repair(C, y, rng)
        if out is not None and np.array_equal(out, x):
            rec_ok += 1
    n_det = len(Ls)
    return dict(F_source=f_src, F_sampled=len(F), d_min_F_ub=dm,
                silent_frac=silent / n_trials,
                L_med=float(np.median(Ls)) if Ls else np.nan,
                R_med=float(np.median(Rs)) if Rs else np.nan,
                inv_R_mean=float(np.mean(invRs)) if invRs else np.nan,
                rec1_emp=rec_ok / max(n_det, 1), n_detected=n_det)


def main():
    rng = np.random.default_rng(1808)
    rows = []
    with open("packs/families_eval_bundle.pkl", "rb") as fh:
        packs = pickle.load(fh)["paper_kv"]
    for d in packs:
        rows.append(dict(family=d["family"], name=d["name"], n=d["n"], **profile(d, rng)))
        print(rows[-1], flush=True)
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv("results/structural_metrics_eval.csv", index=False)
    print("wrote results/structural_metrics_eval.csv", len(rows))


if __name__ == "__main__":
    main()
