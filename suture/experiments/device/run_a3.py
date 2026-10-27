"""run_a3 — hard-edge OBJECTIVE-BLIND device campaign: purify vs SUTURE_C.

Closes the gap that both headline results (72q hardware, 120q ladder) used
the adjacency-reading SUTURE_G, while the objective-blind SUTURE_C existed
only in simulation. Here the edge rows live in C (hard-edge encoding), so
the decoder's neighbour test is a syndrome operation and the arm is
objective-blind: no decoder reads the weight matrix W, which the harness
touches only for scoring.

Protocol mirrors run_a2 exactly (adaptive chain, both arms batched into ONE
job per round with seeds interleaved so calibration drift is shared noise,
measurement-level endpoints only, checkpoint before every submission,
BudgetGuard hard cap). The decoder is hard_dev.decode, vendored verbatim
from experiments/run_hard_ladder.py and verified identical on 400 planted
corruptions.

Usage:
  python device/run_a3.py --dry --instance hardL_v12 --lam 4
  python device/run_a3.py --backend ibm_marrakesh --instance hardL_v12 \
      --seeds 12 --shots 64 --used-baseline 280 --cap 560
"""
import argparse, os, pickle, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dev_lib import (Register, BudgetGuard, job_usage_seconds, job_timestamps,
                     append_jsonl, choose_path, rows_from_bitarray, MAX_REGISTER)
import hard_dev as HD


def syndrome_stats(r, rec):
    """How far from feasible is this measured string? Reads only C.
    Returns (violated_rows, suspect_blocks) — the two quantities the
    decoder's budget is defined on."""
    v, edges, n = rec["v"], rec["edges"], rec["n"]
    dec = HD.blocks_of(r, v)
    viol = int((dec.sum(axis=1) != 1).sum())
    for i, (a, b) in enumerate(edges):
        for c in range(3):
            if r[3 * a + c] + r[3 * b + c] + r[3 * v + 3 * i + c] != 1:
                viol += 1
    susp = set(np.flatnonzero(dec.sum(axis=1) != 1).tolist())
    score = HD.edge_syndrome_score(r, rec)
    susp |= {u for u in range(v) if score[u] >= 3}
    return viol, len(susp)


def arm_postprocess_hard(arm, rows, rec, rng):
    """rows -> (kept, stats, decode_s). Measurement-level endpoints only:
    proper_seen / best cost are scored on MEASURED feasible strings; repaired
    strings re-enter the register but never a categorical endpoint.

    arm='junk' is the A0 control: the SAME decoder on uniform random strings.
    If it survives too, survival is a decoder artifact rather than evidence
    about the quantum state — which is the only way to read a survival
    result when no feasible string is ever measured.
    """
    if arm == "junk":
        rows = rng.integers(0, 2, size=(len(rows), rec["n"])).astype(np.int8)
    kept, n_feas, n_rep, n_giveup = [], 0, 0, 0
    best_conf, best_cost, proper_seen = None, None, False
    viols, susps = [], []
    t0 = time.perf_counter()
    for r in rows:
        vi, su = syndrome_stats(r, rec)
        viols.append(vi); susps.append(su)
        if HD.feasible_hard(r, rec):
            n_feas += 1
            kept.append(r)
            col = HD.blocks_of(r, rec["v"]).argmax(axis=1)
            c = sum(1 for a, b in rec["edges"] if col[a] == col[b])
            cost = float(sum(rec["W"][u][int(cc)] for u, cc in enumerate(col)))
            best_conf = c if best_conf is None else min(best_conf, c)
            best_cost = cost if best_cost is None else min(best_cost, cost)
            proper_seen |= (c == 0)
        elif arm in ("synd_c", "junk"):
            col = HD.decode(r, rec, rng, True)      # objective-blind repair
            if col is not None:
                n_rep += 1
                kept.append(HD.canon_bits(col, rec))
            else:
                n_giveup += 1
    dt = time.perf_counter() - t0
    return kept, dict(f_t=n_feas / max(len(rows), 1), n_repaired=n_rep,
                      n_giveup=n_giveup,
                      viol_med=float(np.median(viols)), viol_min=int(np.min(viols)),
                      susp_med=float(np.median(susps)), susp_min=int(np.min(susps)),
                      frac_in_budget=float(np.mean(np.asarray(susps) <= HD.MAX_CORRUPT)),
                      best_conflicts_measured=best_conf,
                      best_cost_measured=best_cost,
                      proper_seen_measured=bool(proper_seen)), dt


def fresh_state(rec, arms, seeds):
    st = dict(round=0, used_s=0.0, chains={}, rows=[])
    for arm in arms:
        for s in range(seeds):
            rng = np.random.default_rng(70_000 + 1000 * s + arms.index(arm))
            st["chains"][(arm, s)] = dict(reg=Register(rec["seed_bits"], rng),
                                          alive=True, proper_seen=False,
                                          best_conf=None, best_cost=None)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--instance", default="hardL_v12")
    ap.add_argument("--bundle", default="hard_ladder_bundle.pkl")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--shots", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--lam", type=float, default=4.0, help="dry-run noise scale")
    ap.add_argument("--used-baseline", type=float, default=0.0)
    ap.add_argument("--cap", type=float, default=None)
    args = ap.parse_args()

    rec = HD.load_hard_pack(args.instance, bundle_path=args.bundle)
    arms = ["purify", "synd_c", "junk"]
    n_rounds = args.rounds or len(rec["chain"])
    shots = args.shots
    TAG = args.tag
    STATE = f"device/a3_state{TAG}.pkl"
    CHAIN = f"results/a3_chain{TAG}.csv"
    FINAL = f"results/a3_final{TAG}.csv"
    LAT = f"results/a3_latency{TAG}.jsonl"
    os.makedirs("results", exist_ok=True)

    if args.dry:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
        lam = args.lam
        nmz = NoiseModel()
        p1, p2, pro = 3e-4 * lam, 3e-3 * lam, 1e-2 * lam
        if lam > 0:
            nmz.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["x", "sx", "rz", "h"])
            nmz.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx", "cz", "rxx", "ryy"])
            ro = ReadoutError([[1 - pro, pro], [pro, 1 - pro]])
            for q in range(rec["n"] + 2):
                nmz.add_readout_error(ro, [q])
        backend = AerSimulator(method="matrix_product_state", noise_model=nmz)
        path = list(range(rec["n"] + 2))
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        service = QiskitRuntimeService(channel="ibm_cloud",
                                       token=os.environ["IBM_QUANTUM_TOKEN"],
                                       instance=os.environ["CRN_IBMQ"])
        backend = (service.backend(args.backend) if args.backend
                   else service.least_busy(operational=True, simulator=False,
                                           min_num_qubits=rec["n"] + 2))
        print("backend:", backend.name, flush=True)
        path = choose_path(backend, rec["n"] + 2)

    if os.path.exists(STATE):
        with open(STATE, "rb") as fh:
            st = pickle.load(fh)
        print(f"RESUME at round {st['round']} (used {st['used_s']:.0f}s QPU)", flush=True)
    else:
        st = fresh_state(rec, arms, args.seeds)
    guard = BudgetGuard(used_s=st["used_s"] + args.used_baseline)
    if args.cap is not None:
        guard.cap_s = args.cap

    for t in range(st["round"], n_rounds):
        guard.check()
        t_r0 = time.perf_counter()
        keys = [k for k in st["chains"] if st["chains"][k]["alive"]]
        keys.sort(key=lambda k: (k[1], k[0]))          # interleave arms
        if not any(k[0] == "synd_c" for k in keys):
            print("all synd_c chains dead — stop", flush=True); break
        circs, lays, inits = [], [], []
        t_b0 = time.perf_counter()
        for karm, ks in keys:
            init = st["chains"][(karm, ks)]["reg"].sample_rows(1)[0]
            inits.append(init)
            circs.append(HD.hard_segment_circuit(rec, t, init))   # dry path
            lays.append(HD.hard_segment_layout(rec, t, path))
        build_s = time.perf_counter() - t_b0

        st["round"] = t; st["used_s"] = guard.used_s
        with open(STATE, "wb") as fh:
            pickle.dump(st, fh)

        if args.dry:
            # driver_component emits mcx_recursive, which Aer cannot execute
            # directly; decompose to the same core basis build_hard_channels
            # transpiled to (h1_pilot_lib.CORE_BASIS).
            from qiskit import transpile as _tp
            t_q0 = time.perf_counter()
            circs_run = _tp(circs, basis_gates=["measure","cx","id","rz","sx","x"],
                            optimization_level=0)
            res = backend.run(circs_run, shots=shots).result()
            wall_s = time.perf_counter() - t_q0
            usage, queue_s, transpile_s, job_id = wall_s, 0.0, 0.0, f"dry{t}"
            rows_by = []
            for i in range(len(circs)):
                rr = []
                for sbits, ct in res.get_counts(i).items():
                    bits = (np.frombuffer(sbits.replace(" ", "")[::-1].encode(),
                                          dtype=np.uint8) - ord("0")).astype(np.int8)
                    rr.extend([bits[:rec["n"]]] * ct)
                rows_by.append(np.stack(rr))
        else:
            t_tp0 = time.perf_counter()
            # SAME compilation pipeline as run_a2/run_a4 (all prior device
            # campaigns): optimization_level=1 with the round's pinned layout.
            # Do NOT "improve" this — a different level or a free layout makes
            # this cell incomparable with the soft-edge campaigns.
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
            rows_by = [rows_from_bitarray(result[i].data.meas, rec["n"])
                       for i in range(len(circs))]

        dec_s = 0.0
        t_u0 = time.perf_counter()
        for (karm, ks), rows in zip(keys, rows_by):
            ch = st["chains"][(karm, ks)]
            kept, stats, dt = arm_postprocess_hard(karm, rows, rec, ch["reg"].rng)
            dec_s += dt
            alive = ch["reg"].update(kept)
            ch["alive"] = bool(alive)
            ch["proper_seen"] |= stats["proper_seen_measured"]
            for k_ in ("best_conf", "best_cost"):
                v_ = stats["best_conflicts_measured" if k_ == "best_conf" else "best_cost_measured"]
                if v_ is not None:
                    ch[k_] = v_ if ch[k_] is None else min(ch[k_], v_)
            st["rows"].append(dict(arm=karm, seed=ks, t=t, alive=ch["alive"], **stats))
        reg_s = time.perf_counter() - t_u0

        append_jsonl(LAT, dict(round=t, job_id=job_id, n_circuits=len(circs), shots=shots,
                               qpu_usage_s=usage, wall_s=wall_s, queue_s=queue_s,
                               transpile_s=(0.0 if args.dry else transpile_s),
                               build_s=build_s, decode_s=dec_s, register_s=reg_s,
                               round_wall_s=time.perf_counter() - t_r0,
                               cum_qpu_s=guard.used_s))
        pd.DataFrame(st["rows"]).to_csv(CHAIN, index=False)
        n_alive = {a: sum(1 for (ka, _), c in st["chains"].items()
                          if ka == a and c["alive"]) for a in arms}
        print(f"round {t}: alive {n_alive} | qpu {guard.used_s:.0f}s", flush=True)

    st["round"] = n_rounds
    with open(STATE, "wb") as fh:
        pickle.dump(st, fh)
    fin = [dict(arm=a, seed=s, survived=c["alive"], proper_seen=c["proper_seen"],
                best_conf=c["best_conf"], best_cost=c["best_cost"])
           for (a, s), c in st["chains"].items()]
    pd.DataFrame(fin).to_csv(FINAL, index=False)
    print(f"A3 done -> {CHAIN} {FINAL} {LAT}")


if __name__ == "__main__":
    main()
