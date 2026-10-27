"""analyze_eval — report-18 analysis of families_eval_sweep.csv.

Report-15 template per family: survival/cliff vs n, paired ARG (Wilcoxon),
measurement-level optimum sampling (hit_opt_seen, discordant binomial),
advantage-window table; P-18.x scoring hooks. Run AFTER the sweep lands:
    exec(open('scripts/analyze_eval.py').read())
Produces: results/families_eval_paired.csv, results/families_eval_window.csv,
printed scorecard inputs (no figures here — figures are built interactively
with apply_figure_style()).
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest

ev = pd.read_csv("results/families_eval_sweep.csv")
ev["scale"] = ev.instance.str.rsplit("_", n=1).str[0]
NQ = ev.groupby("scale").n.first().to_dict()

# ---------- survival / cliff ----------
surv = ev[ev.arm == "purify"].groupby(["scale", "lam"]).survived.mean().unstack()
cliff = {}
for sc, row in surv.iterrows():
    dead = row[row < 0.5]
    cliff[sc] = float(dead.index.min()) if len(dead) else np.inf
print("=== purify survival cliff (first lam with survival<0.5) ===")
for sc in sorted(cliff, key=lambda s: NQ[s]):
    print(f"  {sc} ({NQ[sc]}q): lam_cliff = {cliff[sc]}")

# ---------- paired ARG ----------
rows_out = []
for (sc, lam), cell in ev.groupby(["scale", "lam"]):
    piv = cell.pivot_table(index=["instance", "seed"], columns="arm", values="ARG")
    if "purify" not in piv.columns or "repair" not in piv.columns:
        continue
    both = piv[["purify", "repair"]].dropna()
    note = ""
    if len(both) < 6:
        note = f"n={len(both)}<6"
        rows_out.append(dict(scale=sc, lam=lam, n_pairs=len(both), note=note))
        continue
    d = both.purify - both.repair
    try:
        p = wilcoxon(d).pvalue if d.abs().sum() > 0 else 1.0
    except ValueError:
        p = np.nan
    rows_out.append(dict(scale=sc, lam=lam, n_pairs=len(both),
                         arg_p=both.purify.median(), arg_r=both.repair.median(),
                         d_med=float(d.median()), p=p, note=note))
P = pd.DataFrame(rows_out)
P.to_csv("results/families_eval_paired.csv", index=False)
print("\n=== paired ARG grid written (families_eval_paired.csv) ===")

# ---------- measurement-level optimum sampling ----------
win_rows = []
for (sc, lam), cell in ev.groupby(["scale", "lam"]):
    pivh = cell.pivot_table(index=["instance", "seed"], columns="arm", values="hit_opt_seen")
    for basearm in ("purify", "junk"):
        if "repair" not in pivh.columns or basearm not in pivh.columns:
            continue
        a, b_ = pivh["repair"].astype(bool), pivh[basearm].astype(bool)
        n10, n01 = int((a & ~b_).sum()), int((~a & b_).sum())
        ph = binomtest(n10, n10 + n01).pvalue if n10 + n01 else 1.0
        win_rows.append(dict(scale=sc, lam=lam, vs=basearm, win=n10, lose=n01, p=ph))
W = pd.DataFrame(win_rows)
W.to_csv("results/families_eval_window.csv", index=False)
print("=== discordant win table written (families_eval_window.csv) ===")

# ---------- KPP endpoint caveat (P-18.5): report raw best cost, not ARG ----------
kpp = ev[ev.scale.str.startswith("kpp")]
if len(kpp):
    print("\n=== KPP raw best-cost medians (ARG degenerate, P-18.5) ===")
    print(kpp.groupby(["scale", "lam", "arm"]).best_cost_seen.median().unstack().round(1).to_string())
