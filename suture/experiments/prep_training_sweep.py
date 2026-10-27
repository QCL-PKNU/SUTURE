"""Training-budget sweep prep (robustness arm registered in reports/08).

Question: is the noise-assisted optimum an artifact of undertrained chains?
Sweep COBYLA budget B (maxiter) per instance, record trained E[conflicts],
rebuild exact channels per (budget, lam) INCLUDING lam=0 (noise-free), ship
to the server for the chain grid.

Instances: the same planted graphs as the scale campaign (identical seeds via
prep_scale.planted_graph) at v=12,16,24 (graph seed 0).
Budgets: 0 (untrained pi/4), 30, 100, 300 (= 08's budget), 1000, 3000.
"""
import sys, os, time, pickle
import _paths  # noqa: F401  (repo-relative sys.path setup)

import numpy as np
from scipy.optimize import minimize

from gcp_k3 import GcpK3Pack, pair_channel_matrix
from gcp_gonogo_lib import run_chain
from build_gcp_packs import AnalyticChannel
from h1_pilot_lib import BASE_PRO
from prep_scale import planted_graph, ROUNDS

BUDGETS = [0, 30, 100, 300, 1000, 3000]
LAMS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
SIZES = [12, 16, 24]


def make_pack(name, v, edges):
    chain = []
    for _ in range(ROUNDS):
        for u in range(v):
            chain.append((u, 0, 1)); chain.append((u, 0, 2))
    return GcpK3Pack(name=name, v=v, K=3, edges=list(edges), chain=chain,
                     params=np.full(len(chain), np.pi / 4),
                     seed_coloring=np.zeros(v, dtype=np.int64))


def train(pack, maxiter):
    ach = AnalyticChannel(pack)

    def objective(params):
        pack.params = np.asarray(params)
        r = run_chain(pack, ach, 512, "purify", np.random.default_rng(11))
        reg = r.final_register
        tot = sum(reg.values())
        return 1e9 if tot == 0 else sum(pack.conflicts(k) * w for k, w in reg.items()) / tot

    x0 = np.full(len(pack.chain), np.pi / 4)
    if maxiter == 0:
        pack.params = x0
        return objective(x0)
    res = minimize(objective, x0, method="COBYLA",
                   options={"maxiter": maxiter, "rhobeg": 0.5})
    pack.params = res.x if objective(res.x) <= objective(x0) else x0
    return objective(pack.params)


def main():
    t0 = time.time()
    bundle = {"lams": LAMS, "budgets": BUDGETS, "entries": []}
    for v in SIZES:
        edges, _ = planted_graph(v, 0)
        name = f"gcp_p{v}_s0"
        for B in BUDGETS:
            pack = make_pack(name, v, edges)
            e_tr = train(pack, B)
            channels = {}
            for lam in LAMS:
                T = np.stack([pair_channel_matrix(pack.params[t], lam, seed=17 + t)
                              for t in range(len(pack.chain))])
                channels[lam] = dict(T=T, p_ro=min(BASE_PRO * lam, 0.5))
            bundle["entries"].append(dict(
                name=name, v=v, K=3, budget=B, e_trained=float(e_tr),
                edges=list(edges), chain=list(pack.chain),
                params=np.asarray(pack.params),
                seed_coloring=np.asarray(pack.seed_coloring), channels=channels))
            print(f"[{time.time()-t0:6.0f}s] {name} B={B}: E[conf] trained={e_tr:.2f}",
                  flush=True)
    with open("packs/training_bundle.pkl", "wb") as fh:
        pickle.dump(bundle, fh, protocol=4)
    print("wrote training_bundle.pkl", os.path.getsize("packs/training_bundle.pkl") / 1e6, "MB")


if __name__ == "__main__":
    main()
