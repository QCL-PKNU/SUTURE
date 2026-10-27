"""run_a2 — A2 end-to-end 36q chain: purify vs SYND (C3 confirmatory cell).

Design (reports/20 Option A, narrowed per user: NO purify+restart):
gcp_p12_s0 (v=12, 36q, 48 trained segments), 2 arms x N_SEEDS seeds,
S=64 shots per (arm, seed) per segment. Adaptive: segment t+1's init
depends on t's outcome, so rounds are sequential; within one round ALL
(arm, seed) pairs batch into ONE job (2 x N_SEEDS circuits x 64 shots).
Seeds interleaved across arms within every job => calibration drift acts
as shared noise, not arm bias (reports/12 discipline).

Latency instrumentation (Rasengan Fig-12 style), per round per arm:
  qpu_usage_s (from job metrics, apportioned per circuit), queue_s,
  transpile_s, decode_s (SYND repair / purify filter), register_update_s,
  build_s. Written to results/a2_latency.jsonl (one row per round) and
  per-(arm,seed,segment) chain stats to results/a2_chain.csv.

Resume: state checkpointed to device/a2_state.pkl BEFORE each submission;
rerun continues at the first unfinished round. Budget: BudgetGuard aborts
past USAGE_CAP_S cumulative reported usage.

Usage:
  python device/run_a2.py --dry            # Aer smoke, no credentials
  python device/run_a2.py                  # device, least busy >=40q
  python device/run_a2.py --backend NAME --seeds 12
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dev_lib import (K, USAGE_CAP_S, load_soft_pack, coloring_to_bits,
                     segment_circuit_baked, segment_layout, choose_path,
                     rows_from_bitarray, Register, arm_postprocess,
                     BudgetGuard, job_usage_seconds, job_timestamps,
                     append_jsonl)

SHOTS = 64


def fresh_state(pack, arms, seeds):
    st = dict(round=0, used_s=0.0, chains={}, rows=[])
    for arm in arms:
        for s in range(seeds):
            rng = np.random.default_rng(90_000 + 1000 * s + (0 if arm == "purify" else 1))
            st["chains"][(arm, s)] = dict(
                reg=Register(coloring_to_bits(pack["seed_coloring"], pack["v"]), rng),
                alive=True, proper_seen=False, best_conf=None)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=None, help="cap rounds (pilot)")
    ap.add_argument("--instance", default="gcp_p12_s0")
    ap.add_argument("--tag", default="", help="suffix for state/output files")
    ap.add_argument("--used-baseline", type=float, default=0.0,
                    help="QPU seconds already spent this campaign (guard counts from here)")
    ap.add_argument("--bundle", default="bench10_bundle.pkl")
    ap.add_argument("--shots", type=int, default=SHOTS)
    ap.add_argument("--cap", type=float, default=None,
                    help="override BudgetGuard cap (absolute, incl. baseline)")
    args = ap.parse_args()
    shots = args.shots

    tag = args.tag
    STATE = f"device/a2_state{tag}.pkl"
    LATP = f"results/a2_latency{tag}.jsonl"
    CHAIN_CSV = f"results/a2_chain{tag}.csv"
    FINAL_CSV = f"results/a2_final{tag}.csv"

    pack = load_soft_pack(args.instance, bundle_path=args.bundle)
    arms = ["purify", "synd"]
    n_rounds = args.rounds or len(pack["chain"])

    if args.dry:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, ReadoutError
        nmz = NoiseModel()
        ro = ReadoutError([[0.97, 0.03], [0.07, 0.93]])
        for q in range(pack["n"]):
            nmz.add_readout_error(ro, [q])
        backend = AerSimulator(method="matrix_product_state", noise_model=nmz)
        path = list(range(pack["n"]))
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        service = QiskitRuntimeService(
            channel="ibm_cloud",
            token=os.environ["IBM_QUANTUM_TOKEN"],
            instance=os.environ["CRN_IBMQ"])
        backend = (service.backend(args.backend) if args.backend
                   else service.least_busy(operational=True, simulator=False, min_num_qubits=40))
        print("backend:", backend.name)
        path = choose_path(backend, pack["n"])

    # ---- resume ----
    if os.path.exists(STATE):
        with open(STATE, "rb") as fh:
            st = pickle.load(fh)
        print(f"RESUME at round {st['round']} (used {st['used_s']:.0f}s QPU)")
    else:
        st = fresh_state(pack, arms, args.seeds)
    guard = BudgetGuard(used_s=st["used_s"] + args.used_baseline)
    if args.cap is not None:
        guard.cap_s = args.cap

    for t in range(st["round"], n_rounds):
        guard.check()
        t_r0 = time.perf_counter()
        # ---- build round batch (one circuit per live (arm, seed)) ----
        keys = [k for k in st["chains"] if st["chains"][k]["alive"]]
        # interleave arms: sort by seed then arm so consecutive circuits alternate
        keys.sort(key=lambda k: (k[1], k[0]))
        if not any(k[0] == "synd" for k in keys):
            print("all synd chains dead — stop"); break
        inits, circs, lays = [], [], []
        t_b0 = time.perf_counter()
        for karm, ks in keys:
            ch = st["chains"][(karm, ks)]
            init = ch["reg"].sample_rows(1)[0]
            inits.append(init)
            circs.append(segment_circuit_baked(pack, t, init))
            lays.append(segment_layout(pack, t, path))
        build_s = time.perf_counter() - t_b0

        # ---- checkpoint BEFORE submission ----
        st["round"] = t
        st["used_s"] = guard.used_s
        with open(STATE, "wb") as fh:
            pickle.dump(st, fh)

        # ---- run ----
        if args.dry:
            t_q0 = time.perf_counter()
            res = backend.run(circs, shots=shots).result()
            wall_s = time.perf_counter() - t_q0
            usage = wall_s; queue_s = 0.0; transpile_s = 0.0; job_id = f"dry{t}"
            rows_by = []
            for i in range(len(circs)):
                rr = []
                for sbits, ct in res.get_counts(i).items():
                    bits = (np.frombuffer(sbits.replace(" ", "")[::-1].encode(), dtype=np.uint8) - ord("0")).astype(np.int8)
                    rr.extend([bits[:pack["n"]]] * ct)
                rows_by.append(np.stack(rr))
        else:
            t_tp0 = time.perf_counter()
            # all circuits in a round share one segment layout (same t, same
            # path) -- generate the pass manager ONCE per round, not per
            # circuit: pm generation against a 156q target costs ~7s, which
            # at 24 circuits/round was ~170s/round of pure client overhead
            # (the dominant wall cost of the 72q campaign).
            assert all(lay == lays[0] for lay in lays), "per-circuit layouts diverged"
            pm = generate_preset_pass_manager(optimization_level=1, backend=backend,
                                              initial_layout=lays[0])
            isa = pm.run(circs)
            transpile_s = time.perf_counter() - t_tp0
            sampler = SamplerV2(mode=backend)
            t_s0 = time.perf_counter()
            job = sampler.run(isa, shots=shots)
            result = job.result()
            wall_s = time.perf_counter() - t_s0
            usage = guard.add_job(job)
            ts = job_timestamps(job)
            queue_s = None
            try:
                from datetime import datetime
                if ts["created"] and ts["running"]:
                    queue_s = (datetime.fromisoformat(str(ts["running"]).replace("Z", "+00:00"))
                               - datetime.fromisoformat(str(ts["created"]).replace("Z", "+00:00"))).total_seconds()
            except Exception:
                pass
            job_id = job.job_id()
            rows_by = [rows_from_bitarray(pub.data.meas, pack["n"])[:, :pack["n"]] for pub in result]

        # ---- postprocess per chain ----
        dec_by_arm = {a: 0.0 for a in arms}
        t_u0 = time.perf_counter()
        for (karm, ks), rows in zip(keys, rows_by):
            ch = st["chains"][(karm, ks)]
            kept, stats, dec_s = arm_postprocess(karm, rows, pack, ch["reg"].rng)
            dec_by_arm[karm] += dec_s
            ch["proper_seen"] |= stats["proper_seen_measured"]
            b = stats["best_conflicts_measured"]
            if b is not None:
                ch["best_conf"] = b if ch["best_conf"] is None else min(ch["best_conf"], b)
            ch["alive"] = ch["reg"].update(kept)
            st["rows"].append(dict(arm=karm, seed=ks, t=t, **stats, alive=ch["alive"]))
        update_s = time.perf_counter() - t_u0

        append_jsonl(LATP, dict(stage="A2", instance=args.instance, round=t, job_id=job_id,
                                n_circuits=len(circs), shots=shots,
                                build_s=build_s,
                                transpile_s=(0.0 if args.dry else transpile_s),
                                queue_s=queue_s, wall_s=wall_s,
                                qpu_usage_s=usage,
                                decode_s_purify=dec_by_arm["purify"],
                                decode_s_synd=dec_by_arm["synd"],
                                register_update_s=update_s,
                                round_wall_s=time.perf_counter() - t_r0,
                                used_cum_s=guard.used_s))
        pd.DataFrame(st["rows"]).to_csv(CHAIN_CSV, index=False)
        alive_p = sum(1 for k in st["chains"] if k[0] == "purify" and st["chains"][k]["alive"])
        alive_s = sum(1 for k in st["chains"] if k[0] == "synd" and st["chains"][k]["alive"])
        print(f"[round {t:2d}] circs {len(circs):2d} usage {usage if usage else 0:5.1f}s "
              f"cum {guard.used_s:5.1f}s | alive purify {alive_p}/{args.seeds} synd {alive_s}/{args.seeds}")

    st["round"] = n_rounds
    with open(STATE, "wb") as fh:
        pickle.dump(st, fh)

    # ---- endpoint summary ----
    fin = []
    for (karm, ks), ch in st["chains"].items():
        fin.append(dict(arm=karm, seed=ks, survived=ch["alive"],
                        proper_seen=ch["proper_seen"], best_conf=ch["best_conf"]))
    F = pd.DataFrame(fin)
    F.to_csv(FINAL_CSV, index=False)
    print(F.groupby("arm")[["survived", "proper_seen"]].mean().round(3).to_string())
    print("A2 done ->", CHAIN_CSV, FINAL_CSV, LATP)


if __name__ == "__main__":
    main()
