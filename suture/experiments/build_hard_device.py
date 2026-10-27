
import os, sys, pickle, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import _paths  # repository-relative library and upstream imports
import numpy as np
from scipy.optimize import minimize
from gcp_hard import HardPack, AnalyticHardChannel

def edges_for(v):
    e = [(i, (i + 1) % v) for i in range(v)]
    e += [(0, 2), (v // 2, v // 2 + 2)]
    return sorted({(min(a, b), max(a, b)) for a, b in e})

def train(pack):
    """Verbatim protocol from build_hard_ladder.train — COBYLA on the
    noise-free analytic channel, so these packs are comparable to hardL_*."""
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
    res = minimize(objective, x0, method="COBYLA", options={"maxiter": 300, "rhobeg": 0.5})
    pack.params = res.x if objective(res.x) <= objective(x0) else x0
    return objective(pack.params)

recs = []
for v in (5, 7):
    t0 = time.time()
    pack = HardPack(f"hardD_v{v}", v, edges_for(v), rounds=1)
    e_tr = train(pack)
    rec = dict(name=pack.name, v=v, K=3, edges=pack.edges, n=pack.n, W=pack.W,
               chain=pack.chain, params=np.asarray(pack.params),
               seed_coloring=np.asarray(pack.seed_coloring),
               proper=[tuple(int(x) for x in c) for c in pack.proper],
               x_star=pack.x_star, e_star=float(pack.e_star), e_trained=float(e_tr),
               build_s=time.time()-t0)
    print(f"[{pack.name}] n={pack.n} |F|={len(pack.proper)} chain={len(pack.chain)} "
          f"E*={pack.e_star:.0f} E_tr={e_tr:.2f} ({rec['build_s']:.0f}s)")
    recs.append(rec)
with open("hard_device_bundle.pkl", "wb") as fh:
    pickle.dump({"instances": recs, "lams": []}, fh, protocol=4)
print("wrote hard_device_bundle.pkl")
