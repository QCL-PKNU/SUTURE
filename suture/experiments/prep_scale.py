"""Local prep for the remote scale campaign.

Builds planted 3-colorable graphs at v in {12,16,24,32,40} (3 graph seeds each)
plus the go/no-go gcp_v8, trains transition times (COBYLA on the analytic
noise-free channel), computes the exact 2-qubit pair-channel T-matrices for
every (segment, lam) with qiskit-aer density simulation, and writes a
pure-pickle bundle (dicts + ndarrays only) that the remote sampler consumes
with numpy alone.
"""
import sys, os, time, pickle
import _paths  # noqa: F401  (repo-relative sys.path setup)

import numpy as np
from scipy.optimize import minimize

from gcp_k3 import GcpK3Pack, pair_channel_matrix
from gcp_gonogo_lib import run_chain
from build_gcp_packs import AnalyticChannel
from h1_pilot_lib import BASE_PRO

SIZES = [12, 16, 24, 32, 40]
GRAPH_SEEDS = [0, 1, 2]
LAMS = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
ROUNDS = 2
TARGET_DEG = 4.0


def planted_graph(v, seed):
    """Planted 3-colorable graph: balanced 3-partition, m = TARGET_DEG*v/2 edges
    between parts, uniformly at random, no duplicates."""
    rng = np.random.default_rng(seed * 1000 + v)
    part = np.array([i % 3 for i in range(v)])
    rng.shuffle(part)
    m = int(TARGET_DEG * v / 2)
    edges = set()
    while len(edges) < m:
        u, w = rng.integers(0, v, 2)
        if u != w and part[u] != part[w]:
            edges.add((min(u, w), max(u, w)))
    return sorted(edges), part


def train_pack(name, v, edges):
    chain = []
    for _ in range(ROUNDS):
        for u in range(v):
            chain.append((u, 0, 1)); chain.append((u, 0, 2))
    pack = GcpK3Pack(name=name, v=v, K=3, edges=list(edges), chain=chain,
                     params=np.full(len(chain), np.pi / 4),
                     seed_coloring=np.zeros(v, dtype=np.int64))
    ach = AnalyticChannel(pack)

    def objective(params):
        pack.params = np.asarray(params)
        r = run_chain(pack, ach, 512, "purify", np.random.default_rng(11))
        reg = r.final_register
        tot = sum(reg.values())
        return 1e9 if tot == 0 else sum(pack.conflicts(k) * w for k, w in reg.items()) / tot

    x0 = np.full(len(chain), np.pi / 4)
    res = minimize(objective, x0, method="COBYLA", options={"maxiter": 300, "rhobeg": 0.5})
    pack.params = res.x if objective(res.x) <= objective(x0) else x0
    return pack, objective(pack.params)


def main():
    bundle = {"lams": LAMS, "instances": []}
    t0 = time.time()
    jobs = [("gcp_v8", 8, [(i, (i+1) % 8) for i in range(8)] + [(0, 2), (4, 6)])]
    for v in SIZES:
        for gs in GRAPH_SEEDS:
            edges, part = planted_graph(v, gs)
            jobs.append((f"gcp_p{v}_s{gs}", v, edges))

    for name, v, edges in jobs:
        pack, e_tr = train_pack(name, v, edges)
        channels = {}
        for lam in LAMS:
            T = np.stack([pair_channel_matrix(pack.params[t], lam, seed=17 + t)
                          for t in range(len(pack.chain))])
            channels[lam] = dict(T=T, p_ro=min(BASE_PRO * lam, 0.5))
        bundle["instances"].append(dict(
            name=name, v=v, K=3, edges=list(edges),
            chain=list(pack.chain), params=np.asarray(pack.params),
            seed_coloring=np.asarray(pack.seed_coloring), channels=channels))
        print(f"[{time.time()-t0:6.0f}s] {name}: v={v} n={v*3} chain={len(pack.chain)} "
              f"E[conf] trained={e_tr:.2f}", flush=True)

    with open("packs/scale_bundle.pkl", "wb") as fh:
        pickle.dump(bundle, fh, protocol=4)
    print("wrote scale_bundle.pkl", os.path.getsize("packs/scale_bundle.pkl") / 1e6, "MB")


if __name__ == "__main__":
    main()
