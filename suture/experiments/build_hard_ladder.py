"""build_hard_ladder — hard-edge GCP scale ladder (report 22).

Purpose: supply an OBJECTIVE-BLIND proper-coloring result at scale. In the
hard-edge encoding the edge rows x_uc + x_wc + s_ec = 1 belong to C, so the
list-coloring decoder's neighbour test is a syndrome operation on constraint
rows, not a read of the objective (which is the random vertex-colour weight
matrix W). This is the arm the soft-edge ladder cannot provide.

Ladder: v = 8, 10, 12, 14 -> n = 54, 66, 78, 90 decision+slack qubits.
Cost driver is the number of DISTINCT feasible restrictions of each active
set (6-8 qubits), which grows linearly in the chain length, not with |F|.

Writes packs/hard_ladder_bundle.pkl. Does not touch hard_bundle.pkl.
"""
import os, sys, time, pickle
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: F401  (repo-relative sys.path setup)
import numpy as np
from scipy.optimize import minimize
from multiprocessing import Pool

from gcp_hard import HardPack, AnalyticHardChannel, build_hard_channels

LAMS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
SPECS = [(8, 1), (10, 1), (12, 1), (14, 1)]   # (v, rounds)


def train(pack):
    """COBYLA on the noise-free analytic channel — same protocol as
    build_hard_bundle.py, so the ladder is comparable to hard_v8."""
    ach = AnalyticHardChannel(pack)

    def objective(params):
        pack.params = np.asarray(params)
        rng = np.random.default_rng(11)
        register = {tuple(int(c) for c in pack.seed_coloring): 512}
        for t in range(len(pack.chain)):
            register = ach.step_register(t, register, 512, rng)
        tot = sum(register.values())
        return sum(pack.cost(k) * w for k, w in register.items()) / tot

    x0 = np.full(len(pack.chain), np.pi / 4)
    res = minimize(objective, x0, method="COBYLA",
                   options={"maxiter": 300, "rhobeg": 0.5})
    pack.params = res.x if objective(res.x) <= objective(x0) else x0
    return objective(pack.params)


def edges_for(v):
    """Cycle plus two chords: chi = 3, degree 2-3, same family as hard_v8."""
    e = [(i, (i + 1) % v) for i in range(v)]
    e += [(0, 2), (v // 2, v // 2 + 2)]
    return sorted({(min(a, b), max(a, b)) for a, b in e})


def build_one(spec):
    v, rounds = spec
    t0 = time.time()
    pack = HardPack(f"hardL_v{v}", v, edges_for(v), rounds=rounds)
    e_tr = train(pack)
    ch = build_hard_channels(pack, LAMS, verbose=False)
    rec = dict(name=pack.name, v=v, K=3, edges=pack.edges, n=pack.n,
               W=pack.W, chain=pack.chain, params=np.asarray(pack.params),
               seed_coloring=np.asarray(pack.seed_coloring),
               proper=[tuple(int(x) for x in c) for c in pack.proper],
               x_star=pack.x_star, e_star=float(pack.e_star), e_trained=float(e_tr),
               channels=ch, lams=LAMS, build_s=time.time() - t0)
    print(f"[{pack.name}] n={pack.n} |F|={len(pack.proper)} "
          f"chain={len(pack.chain)} E*={pack.e_star:.0f} E_tr={e_tr:.2f} {rec['build_s']:.0f}s", flush=True)
    return rec


def main():
    with Pool(min(len(SPECS), os.cpu_count() or 4)) as pool:
        recs = pool.map(build_one, SPECS)
    out = {"lams": LAMS, "instances": recs}
    os.makedirs("packs", exist_ok=True)
    with open("packs/hard_ladder_bundle.pkl", "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    print("wrote packs/hard_ladder_bundle.pkl "
          f"({os.path.getsize('packs/hard_ladder_bundle.pkl')/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
