"""Train hard-edge chains and build exact channels -> hard_bundle.pkl."""
import sys, os, time, pickle
import _paths  # noqa: F401  (repo-relative sys.path setup)

import numpy as np
from scipy.optimize import minimize

from gcp_hard import HardPack, AnalyticHardChannel, build_hard_channels

LAMS = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


def train(pack):
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


def main():
    t0 = time.time()
    bundle = {"lams": LAMS, "instances": []}
    specs = [
        ("hard_c5", 5, [(0,1),(1,2),(2,3),(3,4),(4,0)]),
        ("hard_v8", 8, [(i, (i+1) % 8) for i in range(8)] + [(0, 2), (4, 6)]),
    ]
    for name, v, edges in specs:
        pack = HardPack(name, v, edges)
        e_tr = train(pack)
        print(f"[{name}] v={v} n={pack.n} |F|={len(pack.proper)} chain={len(pack.chain)} "
              f"E*={pack.e_star:.0f} med={pack.e_med:.0f} E_trained={e_tr:.2f}", flush=True)
        ch = build_hard_channels(pack, LAMS)
        bundle["instances"].append(dict(
            name=name, v=v, edges=pack.edges, n=pack.n, W=pack.W,
            chain=pack.chain, params=np.asarray(pack.params),
            seed_coloring=np.asarray(pack.seed_coloring),
            proper=[tuple(int(x) for x in c) for c in pack.proper],
            x_star=pack.x_star, e_star=pack.e_star, e_med=pack.e_med,
            active_sets=[pack.active_set(*tr).tolist() for tr in pack.chain],
            channels=ch,
            p_ro_base=1.0e-2,
        ))
        print(f"  channels done ({time.time()-t0:.0f}s)", flush=True)
    with open("packs/hard_bundle.pkl", "wb") as fh:
        pickle.dump(bundle, fh, protocol=4)
    print("wrote hard_bundle.pkl", os.path.getsize("packs/hard_bundle.pkl") / 1e6, "MB")


if __name__ == "__main__":
    main()
