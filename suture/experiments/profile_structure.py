"""profile_structure — T7 predictor variables per instance, from C and F only.

Per pack:
  d_min_F        min pairwise Hamming distance within F
  silent_frac    fraction of (x in F, single flip) landing in F  (T7 D1)
  L_med          median suspect-set size over planted single flips (T7 D2)
  R_med          median count of weight-1 feasibility-restoring repairs (T7 D3)
  inv_R_mean     E[1/R]  (T7.3's predicted exact-recovery rate)
  rec1_emp       empirical planted-single-flip exact recovery, generic decoder
  col_collisions fraction of suspect pairs (i,j) with identical columns C_i=C_j
Covers families_bundle.pkl + the bench10 GCP-KV packs (pooled predictors).
Output: results/structural_metrics.csv
"""
import sys, os, pickle
import _paths  # noqa: F401  (repo-relative sys.path setup)
import numpy as np
import pandas as pd

from repair_generic import syndrome, repair

def profile(d, rng, n_trials=4000):
    C = np.asarray(d["constr"], dtype=np.int64)
    A, n = C[:, :-1], d["n"]
    F = d["F_bits"]
    Fset = {bytes(fb.tolist()) for fb in F}
    row_sup = [np.flatnonzero(A[r]) for r in range(A.shape[0])]
    col_sup = [np.flatnonzero(A[:, i]) for i in range(n)]

    # d_min(F) (subsample if large)
    FA = np.array(F, dtype=np.int8)
    if len(FA) > 600:
        FA_s = FA[rng.choice(len(FA), 600, replace=False)]
    else:
        FA_s = FA
    dm = min(int(np.abs(FA_s[i] - FA_s[j]).sum())
             for i in range(len(FA_s)) for j in range(i + 1, len(FA_s))) if len(FA_s) > 1 else n

    silent = 0; Ls, Rs, invRs = [], [], []
    rec_ok = 0; coll_num = 0; coll_den = 0
    for t in range(n_trials):
        x = F[rng.integers(len(F))]
        i = rng.integers(n)
        y = x.copy(); y[i] = 1 - y[i]
        if bytes(y.tolist()) in Fset:
            silent += 1
            continue
        syn = syndrome(C, y)
        sus = sorted(set(int(j) for r in syn for j in row_sup[r]))
        Ls.append(len(sus))
        # weight-1 restoring repairs within suspects
        R = 0
        for j in sus:
            z = y.copy(); z[j] = 1 - z[j]
            if len(syndrome(C, z)) == 0:
                R += 1
        Rs.append(R); invRs.append(1.0 / max(R, 1))
        # column collisions among suspects
        for a in range(len(sus)):
            for b in range(a + 1, len(sus)):
                coll_den += 1
                if np.array_equal(A[:, sus[a]], A[:, sus[b]]):
                    coll_num += 1
        # empirical decoder
        out = repair(C, y, rng)
        if out is not None and np.array_equal(out, x):
            rec_ok += 1
    n_det = len(Ls)
    return dict(d_min_F=dm, silent_frac=silent / n_trials,
                L_med=float(np.median(Ls)) if Ls else np.nan,
                R_med=float(np.median(Rs)) if Rs else np.nan,
                inv_R_mean=float(np.mean(invRs)) if invRs else np.nan,
                rec1_emp=rec_ok / max(n_det, 1),
                col_collision_frac=coll_num / max(coll_den, 1),
                n_detected=n_det)

def main():
    rng = np.random.default_rng(1707)
    rows = []
    with open("packs/families_bundle.pkl", "rb") as fh:
        fam = pickle.load(fh)["paper_kv"]
    for d in fam:
        rows.append(dict(family=d["family"], name=d["name"], n=d["n"],
                         F_size=len(d["F_bits"]), **profile(d, rng)))
        print(rows[-1], flush=True)
    with open("packs/bench10_bundle.pkl", "rb") as fh:
        kv = pickle.load(fh)["paper_kv"]
    for d in kv:
        rows.append(dict(family="gcp_kv", name=d["name"], n=d["n"],
                         F_size=len(d["F_bits"]), **profile(d, rng)))
        print(rows[-1]["name"], "done", flush=True)
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv("results/structural_metrics.csv", index=False)
    print("wrote results/structural_metrics.csv", len(rows))

if __name__ == "__main__":
    main()
