"""prep_all_data — derive each figure's processed plotting data from the
repo's raw results (suture/results, suture/paper_data).

One folder per figure: paper_figs/fig_eN_*/data/*.csv are the ONLY inputs
the plot scripts read — regenerating them from raw results is this script.
Run: python paper_figs/prep_all_data.py  (idempotent; prints per-figure
row counts and refuses to write empty tables).
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GR = os.environ.get("SUTURE", os.path.join(HERE, "..", "suture"))
R, PD_, FD = f"{GR}/results", f"{GR}/paper_data", HERE


def write(df, fig, name):
    assert len(df) > 0, f"{fig}/{name}: empty table"
    path = f"{FD}/{fig}/data/{name}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"{path}: {len(df)} rows")


# ---------- Fig E1 (stage 2 core): go/no-go separation, gcp_v8 S=64 ----------
g = pd.read_csv(f"{R}/gonogo_gcp.csv")
sub = g[(g.instance == "gcp_v8") & (g.shots == 64)]
e1 = (sub.groupby(["arm", "lam"])
      .agg(proper_seen=("proper_seen", "mean"), survived=("survived", "mean"),
           n=("seed", "count")).reset_index())
write(e1, "fig_e1_gonogo", "e1_proper_by_arm_lam.csv")

# ---------- Fig E2 (stage 3): scale grid + completed KV family ----------
sc = pd.read_csv(f"{R}/scale_gcp.csv")
e2a = (sc[sc.shots == 64].groupby(["n", "lam", "arm"])
       .agg(proper_seen=("proper_seen", "mean"), survived=("survived", "mean"),
            n_runs=("seed", "count")).reset_index())
write(e2a, "fig_e2_scale", "e2a_scale_grid.csv")
kv = pd.read_csv(f"{R}/bench10_kv_paired_all.csv").rename(columns={"n": "n_pairs"})
nsq = {"kv_v3e2": 15, "kv_v3e3": 18, "kv_v4e3": 28, "kv_v4e4": 32,
       "kv_v4e5": 36, "kv_v5e4": 45, "kv_v5e6": 55}
kv["n_qubits"] = kv.scale.map(nsq)
write(kv, "fig_e2_scale", "e2b_kv_paired.csv")

# ---------- Fig E3 (stage 3): advantage window + honest low-noise cost ----------
write(kv.dropna(subset=["d_med"]), "fig_e3_window", "e3a_kv_dmed.csv")
low = pd.read_csv(f"{PD_}/lownoise_paired.csv")
write(low, "fig_e3_window", "e3b_lownoise.csv")

# ---------- Fig E4 (stages 2+3): hard-edge + noise-assisted mechanism ----------
he = pd.read_csv(f"{R}/hardedge_gcp.csv")
he_v8 = he[(he.instance == "hard_v8") & (he.shots == 64)]
e4a = (he_v8.groupby(["arm", "lam"])
       .agg(best_cost_med=("best_cost_seen", "median"),
            hit=("hit_opt_seen", "mean"), surv=("survived", "mean"),
            n_finite=("best_cost_seen", lambda s: int(np.isfinite(s).sum())))
       .reset_index())
write(e4a, "fig_e4_hardedge_mechanism", "e4a_hardedge.csv")
ts = pd.read_csv(f"{R}/training_sweep.csv")
e4b = (ts[ts.arm == "rec_synd"].groupby(["budget", "lam"])
       .agg(proper_seen=("proper_seen", "mean"), e_trained=("e_trained", "median"))
       .reset_index())
write(e4b, "fig_e4_hardedge_mechanism", "e4b_training.csv")

# ---------- Fig E5 (stage 1 consequence): encoding co-design ----------
ef = pd.read_csv(f"{R}/encoding_flips.csv")
write(ef, "fig_e5_encoding", "e5a_flip_detectability.csv")
dw = pd.read_csv(f"{R}/dw_chain.csv")
e5b = (dw[dw.arm.isin(["purify", "rec_synd"])]
       .groupby(["encoding", "arm", "lam"])
       .agg(proper_seen=("proper_seen", "mean"), survived=("survived", "mean"))
       .reset_index())
write(e5b, "fig_e5_encoding", "e5b_dw_chain.csv")

# ---------- Fig E6 (generalization): curated tables pass through ----------
for src, dst in (("profiler_prediction.csv", "e6a_profiler.csv"),
                 ("cliff_positions.csv", "e6b_cliffs.csv"),
                 ("window_discordance.csv", "e6c_discordance.csv")):
    write(pd.read_csv(f"{PD_}/{src}"), "fig_e6_generalization", dst)

# ---------- Fig E7 (hardware): device campaign ----------
a1 = pd.read_csv(f"{R}/a1_rows.csv")
rows = []
for inst, nq in (("gcp_v8", 24), ("gcp_p12_s0", 36)):
    s_ = a1[a1.instance == inst]
    tot = s_.n_infeasible.sum()
    rows.append(dict(instance=inst, n_qubits=nq, n_infeasible=int(tot),
                     blk1=s_.blk1.sum() / tot,
                     blk2=(s_.blk1.sum() + s_.blk2.sum()) / tot))
write(pd.DataFrame(rows), "fig_e7_device", "e7a_blocklocality.csv")
dev = []
for tag, nq, segs in (("", 36, 48), ("_p16", 48, 64), ("_p24s64", 72, 96)):
    chd = pd.read_csv(f"{R}/a2_chain{tag}.csv")
    m = (chd.groupby(["arm", "t"])
         .agg(best=("best_conflicts_measured", "median"),
              q25=("best_conflicts_measured", lambda x: x.quantile(.25)),
              q75=("best_conflicts_measured", lambda x: x.quantile(.75)),
              alive=("alive", "mean")).reset_index())
    m["n_qubits"], m["segments"] = nq, segs
    dev.append(m)
write(pd.concat(dev, ignore_index=True), "fig_e7_device", "e7b_chains.csv")
fins = []
for tag, nq in (("", 36), ("_p16", 48), ("_p24s64", 72)):
    f_ = pd.read_csv(f"{R}/a2_final{tag}.csv")
    f_["n_qubits"] = nq
    fins.append(f_)
write(pd.concat(fins, ignore_index=True), "fig_e7_device", "e7c_final.csv")
lat = pd.read_csv(f"{R}/latency_breakdown.csv")
write(lat, "fig_e7_device", "e7d_latency.csv")

# ---------- Table E1: envelope boundaries ----------
env = pd.read_csv(f"{PD_}/envelope_boundaries.csv")
write(env, "table_e1_envelope", "t1_envelope.csv")

# Hard-edge measurement-level results and separate A0 decoder diagnostic.
e8_source = 'hard_ladder.csv'
write(pd.read_csv(f"{R}/{e8_source}"), "fig_e8_blind_ladder", "hard_ladder.csv")
for name in ("a4_chain_b36.csv", "a4_final_b36.csv", "a4_paired_b36.csv",
             "a3_final_hard48.csv", "a3_final_hard78.csv"):
    write(pd.read_csv(f"{R}/{name}"), "fig_e9_blind_device", name)
for source, name in (("fig_e2_scale", "e2a_scale_grid.csv"),
                     ("fig_e3_window", "e3a_kv_dmed.csv"),
                     ("fig_e3_window", "e3b_lownoise.csv")):
    write(pd.read_csv(f"{FD}/{source}/data/{name}"), "fig_q3_regime", name)
print("\nALL FIGURE DATA PREPARED")
