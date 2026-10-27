"""run_a1 — A1 block-locality experiment (C2 premise, pre-registered gate).

Design (reports/20 Option A): 2 instances (gcp_v8 24q, gcp_p12_s0 36q)
x 8 segments x 8 proper-coloring inits x 256 shots, ONE batched job per
instance (64 circuits each). Endpoint: among infeasible measured strings,
the fraction with exactly 1 / <=2 corrupted blocks; pre-registered gate:
1-block fraction >= 0.6 at 24q+ (E8 predicts 0.77-0.85).

Usage: python device/run_a1.py [--dry]   (--dry: Aer, no credentials)
State: results/a1_rows.csv, results/a1_latency.jsonl. Budget: ~35s QPU.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dev_lib import (K, load_soft_pack, coloring_to_bits, sample_proper,
                     block_weights, segment_circuit_baked, segment_layout,
                     choose_path, rows_from_bitarray, job_usage_seconds,
                     job_timestamps, append_jsonl)

SHOTS = 256
N_SEG, N_INIT = 8, 8


def build_batch(pack, path, rng):
    segs = sorted(rng.choice(len(pack["chain"]), size=N_SEG, replace=False).tolist())
    inits = list(sample_proper(pack, N_INIT, rng))
    circuits, meta, layouts = [], [], []
    for t in segs:
        for j, col in enumerate(inits):
            init = coloring_to_bits(col, pack["v"])
            circuits.append(segment_circuit_baked(pack, t, init))
            layouts.append(segment_layout(pack, t, path))
            meta.append(dict(t=int(t), init_idx=j))
    return circuits, layouts, meta


def analyze(rows_by_circ, meta, pack, inst):
    out = []
    v = pack["v"]
    for rows, m in zip(rows_by_circ, meta):
        w = (rows.reshape(len(rows), v, K).sum(axis=2) != 1)
        n_corrupt = w.sum(axis=1)
        infeas = n_corrupt > 0
        out.append(dict(instance=inst, **m, shots=len(rows),
                        n_infeasible=int(infeas.sum()),
                        blk1=int((n_corrupt == 1).sum()),
                        blk2=int((n_corrupt == 2).sum()),
                        blk3p=int((n_corrupt >= 3).sum())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--instances", default="gcp_v8,gcp_p12_s0",
                    help="comma-separated pack names")
    ap.add_argument("--bundle", default="bench10_bundle.pkl")
    ap.add_argument("--tag", default="", help="suffix for output files")
    args = ap.parse_args()

    rng = np.random.default_rng(2026)
    rows_all, lat_path = [], f"results/a1_latency{args.tag}.jsonl"

    if args.dry:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, ReadoutError
        nmz = NoiseModel()
        ro = ReadoutError([[0.97, 0.03], [0.07, 0.93]])
        for q in range(96):
            nmz.add_readout_error(ro, [q])
        backend = AerSimulator(method="matrix_product_state", noise_model=nmz)
        run = lambda circs, lays: backend.run(circs, shots=32).result()
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

    for inst in args.instances.split(","):
        pack = load_soft_pack(inst, bundle_path=args.bundle)
        t_build0 = time.perf_counter()
        if args.dry:
            path = list(range(pack["n"]))
        else:
            path = choose_path(backend, pack["n"])
        circuits, layouts, meta = build_batch(pack, path, rng)
        build_s = time.perf_counter() - t_build0

        if args.dry:
            circuits, layouts = circuits[:8], layouts[:8]
            meta = meta[:8]
            t0 = time.perf_counter()
            res = run(circuits, layouts)
            qpu_s = time.perf_counter() - t0
            rows_by_circ = []
            for i in range(len(circuits)):
                counts = res.get_counts(i)
                rr = []
                for s, ct in counts.items():
                    bits = (np.frombuffer(s.replace(" ", "")[::-1].encode(), dtype=np.uint8) - ord("0")).astype(np.int8)
                    rr.extend([bits[:pack["n"]]] * ct)
                rows_by_circ.append(np.stack(rr))
            job_id, usage = "dry", qpu_s
        else:
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            t_tp0 = time.perf_counter()
            isa = []
            for qc, lay in zip(circuits, layouts):
                pm = generate_preset_pass_manager(optimization_level=1, backend=backend,
                                                  initial_layout=lay)
                isa.append(pm.run(qc))
            transpile_s = time.perf_counter() - t_tp0
            sampler = SamplerV2(mode=backend)
            t_sub0 = time.perf_counter()
            job = sampler.run(isa, shots=SHOTS)
            print(f"[{inst}] submitted {len(isa)} circuits, job {job.job_id()}")
            result = job.result()
            wall_s = time.perf_counter() - t_sub0
            usage = job_usage_seconds(job)
            job_id = job.job_id()
            rows_by_circ = []
            for i, pub in enumerate(result):
                ba = pub.data.meas
                rows_full = rows_from_bitarray(ba, backend.num_qubits if ba.num_bits > pack["n"] else pack["n"])
                # measured all n logical qubits; classical register order follows circuit
                rows_by_circ.append(rows_full[:, :pack["n"]])
            append_jsonl(lat_path, dict(stage="A1", instance=inst, job_id=job_id,
                                        n_circuits=len(isa), shots=SHOTS,
                                        build_s=build_s, transpile_s=transpile_s,
                                        wall_s=wall_s, qpu_usage_s=usage,
                                        **job_timestamps(job)))
        rows_all += analyze(rows_by_circ, meta, pack, inst)
        df = pd.DataFrame(rows_all)
        df.to_csv(f"results/a1_rows{args.tag}.csv", index=False)

        sub = df[df.instance == inst]
        tot_inf = sub.n_infeasible.sum()
        if tot_inf:
            f1 = sub.blk1.sum() / tot_inf
            f2 = (sub.blk1.sum() + sub.blk2.sum()) / tot_inf
            print(f"[{inst}] infeasible {tot_inf} | 1-block {f1:.3f} | <=2-block {f2:.3f} "
                  f"| GATE(>=0.6): {'PASS' if f1 >= 0.6 else 'FAIL'}")
        else:
            print(f"[{inst}] no infeasible strings (noise too low for gate; still fine)")

    print(f"A1 done -> results/a1_rows{args.tag}.csv")


if __name__ == "__main__":
    main()
