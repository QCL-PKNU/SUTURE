"""Build dw_bundle.pkl: domain-wall 4x4 pair-channel matrices for the encoding
co-design arm, reusing the one-hot trained params from scale_bundle.pkl
(angles live on coloring space, so they transfer; matched-chain comparison).

Instances: gcp_v8, gcp_p16_s0. Channels per (segment, lam): T_dw from
dw_pair_channel_matrix(theta_t, kind_t, lam). Also records transpiled gate
costs for both encodings' transition circuits.
"""
import sys, os, time, pickle
import _paths  # noqa: F401  (repo-relative sys.path setup)

import numpy as np
from gcp_dw import dw_pair_channel_matrix, dw_gate_cost

LAMS = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]

def main():
    with open("packs/scale_bundle.pkl", "rb") as fh:
        sb = pickle.load(fh)
    want = {"gcp_v8", "gcp_p16_s0"}
    out = {"lams": LAMS, "instances": []}
    t0 = time.time()
    for inst in sb["instances"]:
        if inst["name"] not in want:
            continue
        chain, params = inst["chain"], inst["params"]
        kinds = ["01" if (cf, ct) == (0, 1) else "02" for (_, cf, ct) in chain]
        channels = {}
        for lam in LAMS:
            T = np.stack([dw_pair_channel_matrix(params[t], kinds[t], lam, seed=31 + t)
                          for t in range(len(chain))])
            channels[lam] = dict(T=T, p_ro=min(1.0e-2 * lam, 0.5))
        out["instances"].append(dict(
            name=inst["name"], v=inst["v"], edges=inst["edges"],
            chain=chain, params=np.asarray(params), kinds=kinds,
            seed_coloring=np.asarray(inst["seed_coloring"]),
            channels_dw=channels,
            channels_onehot={lam: inst["channels"][lam] for lam in LAMS
                             if lam in inst["channels"]}))
        print(f"[{time.time()-t0:5.0f}s] {inst['name']} done", flush=True)
    # gate-cost comparison at a representative angle
    from gcp_k3 import pair_channel_matrix  # noqa: F401  (import check)
    costs = {"dw_01": dw_gate_cost(np.pi / 4, "01"), "dw_02": dw_gate_cost(np.pi / 4, "02")}
    out["gate_costs_dw"] = costs
    with open("packs/dw_bundle.pkl", "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    print("wrote dw_bundle.pkl", os.path.getsize("packs/dw_bundle.pkl") / 1e6, "MB", costs)


if __name__ == "__main__":
    main()
