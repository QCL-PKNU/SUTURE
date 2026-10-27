"""Build per-(instance, noise-scale) segment oracles once; both pilots sample from them.

Oracle = empirical estimate of each segment's noisy output distribution
q_t(. | init) for every feasible init, from ORACLE_SHOTS Aer shots.
"""
import sys, os, time
import _paths  # noqa: F401  (repo-relative sys.path setup)

import numpy as np
import dill

from h1_pilot_lib import NoisyRunner, make_noise_model, segment_bodies, SegmentOracle

LAMS = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
EXTRA_LAMS_CHEAP = [6.0, 12.0, 24.0]
CHEAP = {"flp_1x2", "flp_2x2", "scp_4"}
ORACLE_SHOTS = {"flp_1x2": 4096, "flp_2x2": 4096, "scp_4": 4096,
                "flp_2x3": 1024, "gcp_p3": 1024}


def main():
    with open("packs/instance_packs.dill", "rb") as fh:
        packs = dill.load(fh)
    os.makedirs("packs/oracles", exist_ok=True)
    t00 = time.time()
    for name, pack in packs.items():
        bodies = segment_bodies(pack, 1)
        lams = LAMS + (EXTRA_LAMS_CHEAP if name in CHEAP else [])
        for lam in sorted(lams):
            out_path = f"packs/oracles/{name}_lam{lam:g}.dill"
            if os.path.exists(out_path):
                continue
            t0 = time.time()
            runner = NoisyRunner(make_noise_model(lam, lam), seed=int(31 + lam))
            orc = SegmentOracle(pack, runner, bodies, oracle_shots=ORACLE_SHOTS[name])
            orc.build()
            # strip the unpicklable/heavy parts before saving
            payload = dict(name=name, lam=lam, oracle_shots=ORACLE_SHOTS[name],
                           n_segments=len(bodies), q=orc._q)
            with open(out_path, "wb") as fh:
                dill.dump(payload, fh)
            print(f"[{time.time()-t00:7.0f}s] {name} lam={lam:g}: "
                  f"{time.time()-t0:.0f}s -> {out_path}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
