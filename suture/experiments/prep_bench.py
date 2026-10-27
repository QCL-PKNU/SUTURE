"""E6 prep — benchmark broadening (addresses review O4).

Families:
  DIMACS subset (generated procedurally — exact standard constructions):
    myciel3   Mycielskian(C5): 11 v, 20 e, chi=4  -> K=4, n=44
    myciel4   Mycielskian(myciel3): 23 v, 71 e, chi=5 -> K=5, n=115
    queen5_5  5x5 queen graph: 25 v, 160 e, chi=5 -> K=5, n=125
  Degree sweep: planted 3-colorable, v=16, mean degree {2, 4, 6, 8} (K=3, n=48)
    (degree 4 replicates the 08 scale-grid setting)
  K=4 planted: balanced 4-partition, v=12, mean degree 4 (K=4, n=48)

Chain: ROUNDS=2 rounds of (0 -> c) swaps per vertex, c in 1..K-1 (K-generic
extension of the scale-grid chain). Training: COBYLA on the analytic
noise-free channel, expected-conflicts objective (same as prep_scale).
Channels: exact 2q pair matrices per (segment, lam).

Output: bench_bundle.pkl (same schema as scale_bundle.pkl + per-instance K).
"""
import sys, os, time, pickle
import _paths  # noqa: F401  (repo-relative sys.path setup)

import numpy as np
from scipy.optimize import minimize

from gcp_k3 import GcpK3Pack, pair_channel_matrix
from gcp_gonogo_lib import run_chain
from build_gcp_packs import AnalyticChannel
from h1_pilot_lib import BASE_PRO
from prep_scale import planted_graph

LAMS = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
ROUNDS = 2


# --- exact procedural constructions -----------------------------------------

def mycielskian(v, edges):
    """Mycielski construction: v' = 2v+1. chi increases by 1, triangle-free preserved."""
    u_off, w = v, 2 * v
    out = list(edges)
    for (a, b) in edges:
        out.append((a, u_off + b))
        out.append((b, u_off + a))
    for i in range(v):
        out.append((u_off + i, w))
    return 2 * v + 1, sorted(set(tuple(sorted(e)) for e in out))


def queen_graph(n):
    """n x n queen graph: vertices = squares, edges = same row/col/diagonal."""
    edges = set()
    for r1 in range(n):
        for c1 in range(n):
            for r2 in range(n):
                for c2 in range(n):
                    a, b = r1 * n + c1, r2 * n + c2
                    if a >= b:
                        continue
                    if r1 == r2 or c1 == c2 or abs(r1 - r2) == abs(c1 - c2):
                        edges.add((a, b))
    return n * n, sorted(edges)


def planted_k(v, K, seed, target_deg=4.0):
    """Balanced K-partition planted graph (generalizes prep_scale.planted_graph)."""
    rng = np.random.default_rng(seed * 1000 + v + 17 * K)
    part = np.array([i % K for i in range(v)])
    rng.shuffle(part)
    m = int(target_deg * v / 2)
    edges = set()
    tries = 0
    while len(edges) < m and tries < 200 * m:
        u, w = rng.integers(0, v, 2)
        tries += 1
        if u != w and part[u] != part[w]:
            edges.add((min(u, w), max(u, w)))
    return sorted(edges), part


# --- pack build + train ------------------------------------------------------

def make_pack(name, v, K, edges):
    chain = []
    for _ in range(ROUNDS):
        for u in range(v):
            for c in range(1, K):
                chain.append((u, 0, c))
    return GcpK3Pack(name=name, v=v, K=K, edges=list(edges), chain=chain,
                     params=np.full(len(chain), np.pi / 4),
                     seed_coloring=np.zeros(v, dtype=np.int64))


def train(pack, maxiter=300):
    ach = AnalyticChannel(pack)

    def objective(params):
        pack.params = np.asarray(params)
        r = run_chain(pack, ach, 512, "purify", np.random.default_rng(11))
        reg = r.final_register
        tot = sum(reg.values())
        return 1e9 if tot == 0 else sum(pack.conflicts(k) * w for k, w in reg.items()) / tot

    x0 = np.full(len(pack.chain), np.pi / 4)
    res = minimize(objective, x0, method="COBYLA",
                   options={"maxiter": maxiter, "rhobeg": 0.5})
    pack.params = res.x if objective(res.x) <= objective(x0) else x0
    return objective(pack.params)


def main():
    m3_v, m3_e = mycielskian(5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
    m4_v, m4_e = mycielskian(m3_v, m3_e)
    q_v, q_e = queen_graph(5)
    print(f"myciel3: {m3_v}v {len(m3_e)}e | myciel4: {m4_v}v {len(m4_e)}e | "
          f"queen5_5: {q_v}v {len(q_e)}e", flush=True)
    assert (m3_v, len(m3_e)) == (11, 20) and (m4_v, len(m4_e)) == (23, 71) \
        and (q_v, len(q_e)) == (25, 160), "DIMACS construction mismatch"

    jobs = [
        ("myciel3", m3_v, 4, m3_e),
        ("myciel4", m4_v, 5, m4_e),
        ("queen5_5", q_v, 5, q_e),
        ("planted_k4_v12", 12, 4, planted_k(12, 4, seed=0)[0]),
    ]
    for deg in (2, 4, 6, 8):
        edges, _ = planted_graph(16, 0) if deg == 4 else (None, None)
        if edges is None:
            rng = np.random.default_rng(16 + deg)
            part = np.array([i % 3 for i in range(16)]); rng.shuffle(part)
            m = int(deg * 16 / 2)
            es = set()
            while len(es) < m:
                u, w = rng.integers(0, 16, 2)
                if u != w and part[u] != part[w]:
                    es.add((min(u, w), max(u, w)))
            edges = sorted(es)
        jobs.append((f"deg{deg}_v16", 16, 3, edges))

    bundle = {"lams": LAMS, "instances": []}
    t0 = time.time()
    for name, v, K, edges in jobs:
        pack = make_pack(name, v, K, edges)
        e_tr = train(pack)
        channels = {}
        for lam in LAMS:
            T = np.stack([pair_channel_matrix(pack.params[t], lam, seed=17 + t)
                          for t in range(len(pack.chain))])
            channels[lam] = dict(T=T, p_ro=min(BASE_PRO * lam, 0.5))
        bundle["instances"].append(dict(
            name=name, v=v, K=K, edges=list(edges), chain=list(pack.chain),
            params=np.asarray(pack.params),
            seed_coloring=np.asarray(pack.seed_coloring), channels=channels))
        print(f"[{time.time()-t0:6.0f}s] {name}: v={v} K={K} n={v*K} "
              f"|E|={len(edges)} chain={len(pack.chain)} E[conf]={e_tr:.2f}", flush=True)

    with open("packs/bench_bundle.pkl", "wb") as fh:
        pickle.dump(bundle, fh, protocol=4)
    print("wrote bench_bundle.pkl", os.path.getsize("packs/bench_bundle.pkl") / 1e6, "MB")


if __name__ == "__main__":
    main()
