"""validate_lam0 — noise-free internal-consistency gate (runs on the server).

For every kv pack, at lam=0:
  1. EXACT: propagate the register distribution through the lam=0 channels
     analytically (probability vectors, no sampling) -> E_exact[cost] on the
     final register; ARG_exact = |E_exact - best| / best.
  2. SAMPLED: run the purify arm of remote_bench10.run_chain at S=1024,
     4 seeds -> paper metrics.
Gates:
  G1  in_constraints_probs == 100 (exactly, lam=0 chains cannot leave F)
  G2  |ARG_sampled_median - ARG_exact| <= 0.05 + 2*sd (sampling error)
  G3  survived == True for all seeds
Output: validate_lam0.csv (one row per pack: exact vs sampled, gate flags).
"""
import os, sys, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from remote_bench10 import KvChannel, run_chain, syndrome

with open("packs/bench10_bundle.pkl", "rb") as fh:
    bundle = pickle.load(fh)
with open("packs/bench10_channels.pkl", "rb") as fh:
    channels = pickle.load(fh)

rows = []
for d in bundle["paper_kv"]:
    ch0 = channels["kv"][d["name"]][0.0]
    obj_cc = np.asarray(d["cost_color"], dtype=float)
    V, K = d["V"], d["K"]
    def obj(bits):
        x = np.asarray(bits[:V * K]).reshape(V, K)
        return float(obj_cc @ (1 - np.prod(1 - x, axis=0)))

    # --- exact propagation ---
    active = [np.asarray(a, dtype=np.int64) for a in d["active_sets"]]
    reg = {d["seed_bits"].tobytes(): 1.0}
    for t in range(len(d["chain_vecs"])):
        A = active[t]
        nxt = {}
        for kb, w in reg.items():
            bits = np.frombuffer(kb, dtype=np.int8)
            entry = ch0.get((t, bytes(bits[A].tolist())))
            if entry is None:
                nxt[kb] = nxt.get(kb, 0.0) + w
                continue
            codes, probs = entry
            for code, p in zip(codes, probs):
                b2 = bits.copy()
                for i in range(len(A)):
                    b2[A[i]] = (code >> i) & 1
                k2 = b2.tobytes()
                nxt[k2] = nxt.get(k2, 0.0) + w * p
        reg = nxt
    E_exact = sum(obj(np.frombuffer(k, dtype=np.int8)) * w for k, w in reg.items())
    w_feas = sum(w for k, w in reg.items()
                 if len(syndrome(d["constr"], np.frombuffer(k, dtype=np.int8))) == 0)
    arg_exact = abs((E_exact - d["best_cost"]) / (d["best_cost"] + 1e-8))

    # --- sampled purify runs ---
    args, icps, survs = [], [], []
    for seed in range(4):
        rng = np.random.default_rng(51_000 + seed)
        ch = KvChannel(d, ch0, 0.0)
        r = run_chain(d["constr"], d["seed_bits"], len(d["chain_vecs"]), ch,
                      "purify", rng, d["F_bits"], obj, d["best_cost"])
        args.append(r["ARG"]); icps.append(r["in_constraints_probs"]); survs.append(r["survived"])

    arg_med = float(np.median(args)); arg_sd = float(np.std(args))
    g1 = all(abs(x - 100.0) < 1e-9 for x in icps)
    g2 = abs(arg_med - arg_exact) <= 0.05 + 2 * arg_sd
    g3 = all(survs)
    rows.append(dict(name=d["name"], n=d["n"], best=d["best_cost"],
                     E_exact=E_exact, arg_exact=arg_exact, w_feas_exact=w_feas,
                     arg_sampled_med=arg_med, arg_sampled_sd=arg_sd,
                     icp_min=min(icps), G1_icp100=g1, G2_arg=g2, G3_surv=g3))
    print(rows[-1], flush=True)

df = pd.DataFrame(rows)
df.to_csv("validate_lam0.csv", index=False)
ok = df.G1_icp100.all() and df.G2_arg.all() and df.G3_surv.all()
print(f"\nGATES {'PASS' if ok else 'FAIL'}: G1={df.G1_icp100.all()} "
      f"G2={df.G2_arg.all()} G3={df.G3_surv.all()}")
