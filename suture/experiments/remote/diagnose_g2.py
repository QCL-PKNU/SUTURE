"""diagnose_g2 — why do scp_12s8e packs fail G2' (S=1024 vs S=8192)?

Hypothesis: MAX_REGISTER truncation (the eval pipeline runs at
MAX_REGISTER=256). At lam=0 these packs' reachable register support is
~|F| >> 256, so the tracked register is a shot-count-dependent truncation;
ARG then legitimately drifts with S. Test: ARG vs
S in {512,1024,2048,4096,8192} x MAX_REGISTER in {64,512,4096} — bracketing
the pipeline's 256 from both sides — 6 seeds,
lam=0, purify arm only, the three failing packs.
If ARG converges as MAX_REGISTER grows (at fixed S) and the S-drift
disappears at large MAX_REGISTER -> truncation confirmed: protocol
property, gate re-specified shot-matched; wiring exonerated.
"""
import os, sys, pickle
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import pandas as pd

import remote_bench10_eval as rb
from remote_bench10_eval import KvChannel, run_chain, make_obj

with open("families_eval_bundle.pkl", "rb") as fh:
    packs = [d for d in pickle.load(fh)["paper_kv"] if d["name"].startswith("scp_12s8e")]
with open("families_eval_channels.pkl", "rb") as fh:
    channels = pickle.load(fh)

rows = []
for d in packs:
    ch0 = channels["kv"][d["name"]][0.0]
    obj = make_obj(d["obj_spec"])
    for mr in (64, 512, 4096):
        rb.MAX_REGISTER = mr
        for S in (512, 1024, 2048, 4096, 8192):
            rb.SHOTS = S
            for seed in range(6):
                rng = np.random.default_rng(53_000 + seed)
                ch = KvChannel(d, ch0, 0.0)
                r = run_chain(d["constr"], d["seed_bits"], len(d["chain_vecs"]),
                              ch, "purify", rng, d.get("F_bits"), obj, d["best_cost"])
                rows.append(dict(name=d["name"], max_reg=mr, shots=S, seed=seed,
                                 ARG=r["ARG"], icp=r["in_constraints_probs"]))
            print(f"{d['name']} mr={mr} S={S}: "
                  f"med={np.median([x['ARG'] for x in rows if x['name']==d['name'] and x['max_reg']==mr and x['shots']==S]):.3f}",
                  flush=True)
pd.DataFrame(rows).to_csv("diagnose_g2.csv", index=False)
print("wrote diagnose_g2.csv")
