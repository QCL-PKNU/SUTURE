"""run_a4 — soft-edge device campaign with the OBJECTIVE-BLIND arm.

Three arms in ONE job per round (seeds interleaved, so calibration drift is
shared noise rather than arm bias):

  purify   — Rasengan's published mitigation: discard infeasible shots.
  synd_g   — SUTURE_G, the graph-aware specialization (dev_lib.decode_synd_soft):
             prunes candidate colors against intact neighbours. In the
             soft-edge encoding the edges ARE the objective, so this arm is
             NOT objective-blind.
  synd_c   — SUTURE_C, the manuscript's GENERIC decoder, run verbatim from
             lib/repair_generic.py against the soft-edge constraint system
             C (assignment rows sum_c x_uc = 1 only). It never sees the
             graph. This is the objective-blind arm.

The generic objective-blind decoder is tested for measured quality at 36q.
Separate hard-edge experiments in run_a3.py establish liveness only: their
raw measurements contain no feasible bitstrings.
Endpoint discipline unchanged: measurement-level endpoints only; repaired
strings re-enter the register but never score a categorical endpoint.
"""
import argparse, os, pickle, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dev_lib import (K, load_soft_pack, coloring_to_bits, segment_circuit_baked,
                     segment_layout, choose_path, rows_from_bitarray, Register,
                     BudgetGuard, job_timestamps, append_jsonl,
                     is_onehot, conflicts, decode_synd_soft)
import repair_generic as RG

ARMS = ["purify", "synd_g", "synd_c"]


def assignment_system(pack):
    """C|b for the soft-edge feasibility system: one row per vertex,
    sum_c x_uc = 1. Contains NO edge information — this is what makes the
    generic arm objective-blind in this encoding."""
    v, n = pack["v"], pack["v"] * K
    C = np.zeros((v, n + 1), dtype=np.int64)
    for u in range(v):
        C[u, u * K:(u + 1) * K] = 1
        C[u, -1] = 1
    return C


def arm_postprocess3(arm, rows, pack, Csys, rng):
    v = pack["v"]
    kept, n_feas, n_rep = [], 0, 0
    best_conf, proper_seen = None, False
    t0 = time.perf_counter()
    for r in rows:
        if is_onehot(r, v):
            n_feas += 1
            kept.append(r)
            c = conflicts(r, pack)
            best_conf = c if best_conf is None else min(best_conf, c)
            proper_seen |= (c == 0)
        elif arm == "synd_g":
            fixed = decode_synd_soft(v, K, pack["adj"], r, rng)
            if fixed is not None:
                n_rep += 1; kept.append(fixed)
        elif arm == "synd_c":
            fixed = RG.repair(Csys, np.asarray(r, dtype=np.int8), rng)
            if fixed is not None:
                n_rep += 1; kept.append(np.asarray(fixed, dtype=np.int8))
    dt = time.perf_counter() - t0
    return kept, dict(f_t=n_feas / max(len(rows), 1), n_repaired=n_rep,
                      best_conflicts_measured=best_conf,
                      proper_seen_measured=bool(proper_seen)), dt


def fresh_state(pack, seeds):
    st = dict(round=0, used_s=0.0, chains={}, rows=[])
    for ai, arm in enumerate(ARMS):
        for s in range(seeds):
            rng = np.random.default_rng(50_000 + 1000 * s + ai)
            st["chains"][(arm, s)] = dict(
                reg=Register(coloring_to_bits(pack["seed_coloring"], pack["v"]), rng),
                alive=True, proper_seen=False, best_conf=None)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--instance", default="gcp_p12_s0")
    ap.add_argument("--bundle", default="bench10_bundle.pkl")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--shots", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--lam", type=float, default=4.0)
    ap.add_argument("--used-baseline", type=float, default=0.0)
    ap.add_argument("--cap", type=float, default=None)
    args = ap.parse_args()

    pack = load_soft_pack(args.instance, bundle_path=args.bundle)
    Csys = assignment_system(pack)
    n_rounds = args.rounds or len(pack["chain"])
    shots = args.shots
    TAG = args.tag
    STATE, CHAIN = f"device/a4_state{TAG}.pkl", f"results/a4_chain{TAG}.csv"
    FINAL, LAT = f"results/a4_final{TAG}.csv", f"results/a4_latency{TAG}.jsonl"
    os.makedirs("results", exist_ok=True)

    if args.dry:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
        lam = args.lam
        nmz = NoiseModel()
        if lam > 0:
            nmz.add_all_qubit_quantum_error(depolarizing_error(3e-4 * lam, 1), ["x", "sx", "rz", "h"])
            nmz.add_all_qubit_quantum_error(depolarizing_error(3e-3 * lam, 2), ["cx", "cz", "rxx", "ryy"])
            ro = ReadoutError([[1 - 1e-2 * lam, 1e-2 * lam], [1e-2 * lam, 1 - 1e-2 * lam]])
            for q in range(pack["n"]):
                nmz.add_readout_error(ro, [q])
        backend = AerSimulator(method="matrix_product_state", noise_model=nmz)
        path = list(range(pack["n"]))
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        service = QiskitRuntimeService(channel="ibm_cloud",
                                       token=os.environ["IBM_QUANTUM_TOKEN"],
                                       instance=os.environ["CRN_IBMQ"])
        backend = (service.backend(args.backend) if args.backend
                   else service.least_busy(operational=True, simulator=False,
                                           min_num_qubits=pack["n"]))
        print("backend:", backend.name, flush=True)
        path = choose_path(backend, pack["n"])

    if os.path.exists(STATE):
        with open(STATE, "rb") as fh:
            st = pickle.load(fh)
        print(f"RESUME at round {st['round']} (used {st['used_s']:.0f}s QPU)", flush=True)
    else:
        st = fresh_state(pack, args.seeds)
    guard = BudgetGuard(used_s=st["used_s"] + args.used_baseline)
    if args.cap is not None:
        guard.cap_s = args.cap

    for t in range(st["round"], n_rounds):
        guard.check()
        t_r0 = time.perf_counter()
        keys = [k for k in st["chains"] if st["chains"][k]["alive"]]
        keys.sort(key=lambda k: (k[1], k[0]))
        if not any(k[0] in ("synd_c", "synd_g") for k in keys):
            print("all repair chains dead — stop", flush=True); break
        circs, lays = [], []
        t_b0 = time.perf_counter()
        for karm, ks in keys:
            init = st["chains"][(karm, ks)]["reg"].sample_rows(1)[0]
            circs.append(segment_circuit_baked(pack, t, init))
            lays.append(segment_layout(pack, t, path))
        build_s = time.perf_counter() - t_b0

        st["round"] = t; st["used_s"] = guard.used_s
        with open(STATE, "wb") as fh:
            pickle.dump(st, fh)

        if args.dry:
            t_q0 = time.perf_counter()
            res = backend.run(circs, shots=shots).result()
            wall_s = time.perf_counter() - t_q0
            usage, queue_s, transpile_s, job_id = wall_s, 0.0, 0.0, f"dry{t}"
            rows_by = []
            for i in range(len(circs)):
                rr = []
                for sbits, ct in res.get_counts(i).items():
                    bits = (np.frombuffer(sbits.replace(" ", "")[::-1].encode(),
                                          dtype=np.uint8) - ord("0")).astype(np.int8)
                    rr.extend([bits[:pack["n"]]] * ct)
                rows_by.append(np.stack(rr))
        else:
            t_tp0 = time.perf_counter()
            assert all(l == lays[0] for l in lays), "per-circuit layouts diverged"
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
            job_id = job.job_id()
            ts = job_timestamps(job)
            try:
                queue_s = (ts["running"] - ts["created"]).total_seconds()
            except Exception:
                queue_s = None
            rows_by = [rows_from_bitarray(result[i].data.meas, pack["n"])
                       for i in range(len(circs))]

        dec_s = {a: 0.0 for a in ARMS}
        t_u0 = time.perf_counter()
        for (karm, ks), rows in zip(keys, rows_by):
            ch = st["chains"][(karm, ks)]
            kept, stats, dt = arm_postprocess3(karm, rows, pack, Csys, ch["reg"].rng)
            dec_s[karm] += dt
            ch["alive"] = bool(ch["reg"].update(kept))
            ch["proper_seen"] |= stats["proper_seen_measured"]
            if stats["best_conflicts_measured"] is not None:
                ch["best_conf"] = (stats["best_conflicts_measured"] if ch["best_conf"] is None
                                   else min(ch["best_conf"], stats["best_conflicts_measured"]))
            st["rows"].append(dict(arm=karm, seed=ks, t=t, alive=ch["alive"], **stats))
        reg_s = time.perf_counter() - t_u0

        append_jsonl(LAT, dict(round=t, job_id=job_id, n_circuits=len(circs), shots=shots,
                               qpu_usage_s=usage, wall_s=wall_s, queue_s=queue_s,
                               transpile_s=(0.0 if args.dry else transpile_s),
                               build_s=build_s, register_s=reg_s,
                               **{f"decode_s_{a}": dec_s[a] for a in ARMS},
                               round_wall_s=time.perf_counter() - t_r0,
                               cum_qpu_s=guard.used_s))
        pd.DataFrame(st["rows"]).to_csv(CHAIN, index=False)
        n_alive = {a: sum(1 for (ka, _), c in st["chains"].items() if ka == a and c["alive"])
                   for a in ARMS}
        print(f"round {t}: alive {n_alive} | qpu {guard.used_s:.0f}s", flush=True)

    st["round"] = n_rounds
    with open(STATE, "wb") as fh:
        pickle.dump(st, fh)
    pd.DataFrame([dict(arm=a, seed=s, survived=c["alive"], proper_seen=c["proper_seen"],
                       best_conf=c["best_conf"]) for (a, s), c in st["chains"].items()]
                 ).to_csv(FINAL, index=False)
    print(f"A4 done -> {CHAIN} {FINAL} {LAT}")


if __name__ == "__main__":
    main()
