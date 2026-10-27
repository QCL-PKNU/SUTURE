"""curate_paper_data — manuscript-grade data extracts (paper_data/*.csv).

One CSV per figure/table claim in the outline's evaluation figure map:
  profiler_prediction.csv   fig_m6a  (C6: recovery = E[1/R], 70 inst, 4 fam)
  cliff_positions.csv       fig_m6b  (C6: purify death-lam vs n, all families;
                                      repair-arm death at kpp_p16b4 marked)
  window_discordance.csv    fig_m6c  (C6: repair-vs-purify discordant
                                      optimum-sampling wins per scale x lam)
  lownoise_paired.csv       Table E1 (C7-i: low-lam paired ARG cells, kpp
                                      excluded per P-18.5 ARG degeneracy)
  envelope_boundaries.csv   Table E1 (C7-ii/iii: repair weight-budget death;
                                      scp_12s8e both-arms optimum floor)
  predictability_link.csv   text     (C6: peak discordance vs profiler inv_R)

Sources: bench10_kv(.fix).csv, bench10_kv_paired_all.csv (GCP),
families_eval_sweep.csv + families_eval_paired.csv (FLP/SCP/KPP ladders),
structural_metrics(.eval).csv (profiler). Run from suture/: paths below.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import binomtest

RR = "results"
OUT = "paper_data"
os.makedirs(OUT, exist_ok=True)

# ---------- profiler prediction ----------
sm1 = pd.read_csv(f"{RR}/structural_metrics.csv"); sm1["source"] = "report17"
sm2 = pd.read_csv(f"{RR}/structural_metrics_eval.csv"); sm2["source"] = "eval_ladder"
keep = ["family", "name", "n", "inv_R_mean", "rec1_emp", "silent_frac", "source"]
prof = pd.concat([sm1[keep], sm2[keep]], ignore_index=True).drop_duplicates("name")
prof.to_csv(f"{OUT}/profiler_prediction.csv", index=False)

# ---------- sweep frames ----------
kv = pd.concat([pd.read_csv(f"{RR}/bench10_kv.csv"),
                pd.read_csv(f"{RR}/bench10_kv_fix.csv")], ignore_index=True)
kv["scale"] = kv.instance.str.rsplit("_", n=1).str[0]
kv["family"] = "gcp"
ev = pd.read_csv(f"{RR}/families_eval_sweep.csv")
ev["scale"] = ev.instance.str.rsplit("_", n=1).str[0]
ev["family"] = ev.scale.str.split("_").str[0]
allf = pd.concat([kv, ev], ignore_index=True)

# ---------- cliff positions ----------
def cliffs(df, arm, col):
    out = []
    for sc, sub in df[df.arm == arm].groupby("scale"):
        s = sub.groupby("lam").survived.mean()
        dead = s[s < 0.5]
        out.append(dict(scale=sc, family=sub.family.iloc[0], n=int(sub.n.iloc[0]),
                        **{col: float(dead.index.min()) if len(dead) else np.nan},
                        lam_max_grid=float(s.index.max())))
    return pd.DataFrame(out)

cliff = (cliffs(allf, "purify", "purify_cliff")
         .merge(cliffs(allf, "repair", "repair_cliff")[["scale", "repair_cliff"]], on="scale")
         .sort_values("n"))
cliff.to_csv(f"{OUT}/cliff_positions.csv", index=False)

# ---------- window discordance ----------
rows = []
for (sc, lam), cell in allf.groupby(["scale", "lam"]):
    pivh = cell.pivot_table(index=["instance", "seed"], columns="arm", values="hit_opt_seen")
    if "repair" not in pivh.columns or "purify" not in pivh.columns:
        continue
    a, b = pivh["repair"].astype(bool), pivh["purify"].astype(bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    if n10 + n01 == 0:
        continue
    rows.append(dict(family=cell.family.iloc[0], scale=sc, n=int(cell.n.iloc[0]),
                     lam=lam, win=n10, lose=n01, p=binomtest(n10, n10 + n01).pvalue))
disc = pd.DataFrame(rows)
disc.to_csv(f"{OUT}/window_discordance.csv", index=False)

# ---------- low-noise paired ARG (kpp excluded: P-18.5 ARG degeneracy) ----------
pk = pd.read_csv(f"{RR}/bench10_kv_paired_all.csv").rename(columns={"n": "n_pairs"})
pk["family"] = "gcp"
pf = pd.read_csv(f"{RR}/families_eval_paired.csv")
pf["family"] = pf.scale.str.split("_").str[0]
cols = ["family", "scale", "lam", "n_pairs", "d_med", "p"]
low = pd.concat([pk[cols], pf[cols]], ignore_index=True)
low = low[low.lam.between(0.5, 4) & low.n_pairs.ge(6)
          & (low.family != "kpp") & low.d_med.notna()]
low["n_qubits"] = low.scale.map(cliff.set_index("scale").n.to_dict())
low.to_csv(f"{OUT}/lownoise_paired.csv", index=False)

# ---------- envelope boundaries ----------
s12 = ev[ev.scale == "scp_12s8e"]
k = ev[(ev.scale == "kpp_p16b4") & (ev.arm == "repair")].groupby("lam").survived.mean()
pd.DataFrame([
    dict(item="scp_12s8e_floor",
         value=float(s12[s12.arm.isin(["purify", "repair"])].hit_opt_seen.max()),
         detail="max hit_opt_seen over both arms, all lam (37-44q SCP)"),
    dict(item="kpp_p16b4_repair_survival_lam32_64",
         value=float(k.loc[[32.0, 64.0]].max()),
         detail="repair survival at lam 32/64 (64q; corruption exceeds MAX_CORRUPT=6)"),
]).to_csv(f"{OUT}/envelope_boundaries.csv", index=False)

# ---------- predictability link ----------
prof_i = prof.set_index("name")
link = []
for sc, sub in disc.groupby("scale"):
    names = [nm for nm in prof_i.index if nm.startswith(sc)]
    link.append(dict(scale=sc, family=sub.family.iloc[0], n=int(sub.n.iloc[0]),
                     peak_disc=int((sub.win - sub.lose).max()),
                     inv_R=prof_i.loc[names].inv_R_mean.median() if names else np.nan))
pd.DataFrame(link).sort_values("n").to_csv(f"{OUT}/predictability_link.csv", index=False)

print("paper_data/ written:", sorted(os.listdir(OUT)))
