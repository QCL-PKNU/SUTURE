"""test_connection — step 0: verify IBM Quantum access, spend ~0 QPU time.

Checks (in order, stops at first failure):
  1. credentials load (QiskitRuntimeService())
  2. backends visible; pick least-busy >=40q; print basis gates + queue
  3. transpile ONE 36q segment circuit against the real backend target
     (no execution) — catches ISA/coupling surprises for free
  4. OPTIONAL --submit: run that single circuit with 16 shots (the only
     step that spends QPU time: ~1-2 s) and report usage + timestamps,
     which also pins the per-job overhead constant for the budget model.

Usage:
  python device/test_connection.py            # steps 1-3, zero QPU
  python device/test_connection.py --submit   # + one 16-shot job
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dev_lib import (load_soft_pack, coloring_to_bits, enumerate_proper,
                     segment_circuit_baked, segment_layout, choose_path,
                     rows_from_bitarray, job_usage_seconds, job_timestamps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--backend", default=None)
    args = ap.parse_args()

    print("1) loading credentials...")
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    service = QiskitRuntimeService(
            channel="ibm_cloud",
            token=os.environ["IBM_QUANTUM_TOKEN"],
            instance=os.environ["CRN_IBMQ"])
    print("   OK — channel:", service.channel)

    print("2) listing backends...")
    bks = service.backends(operational=True, simulator=False)
    for b in bks:
        st = b.status()
        print(f"   {b.name}: {b.num_qubits}q pending={st.pending_jobs}")
    backend = (service.backend(args.backend) if args.backend
               else service.least_busy(operational=True, simulator=False, min_num_qubits=40))
    print("   selected:", backend.name, "| basis:", list(backend.target.operation_names)[:12])

    print("3) transpiling one 36q segment (no execution)...")
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    pack = load_soft_pack("gcp_p12_s0")
    path = choose_path(backend, pack["n"])
    proper = enumerate_proper(pack, cap=10)
    init = coloring_to_bits(proper[0], pack["v"])
    qc = segment_circuit_baked(pack, 0, init)
    lay = segment_layout(pack, 0, path)
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend, initial_layout=lay)
    t0 = time.perf_counter()
    tqc = pm.run(qc)
    n2 = sum(1 for inst in tqc.data if inst.operation.num_qubits == 2)
    print(f"   OK — depth {tqc.depth()}, 2q gates {n2}, transpile {time.perf_counter()-t0:.2f}s, "
          f"path head {path[:6]}")
    assert n2 <= 6, f"2q count {n2} unexpectedly high — routing crept in"

    if args.submit:
        print("4) submitting ONE 16-shot job (~1-2 s QPU)...")
        sampler = SamplerV2(mode=backend)
        t0 = time.perf_counter()
        job = sampler.run([tqc], shots=16)
        print("   job:", job.job_id())
        res = job.result()
        wall = time.perf_counter() - t0
        rows = rows_from_bitarray(res[0].data.meas, pack["n"])
        w = rows.reshape(len(rows), pack["v"], 3).sum(axis=2)
        print(f"   OK — wall {wall:.1f}s, usage {job_usage_seconds(job)}s, "
              f"timestamps {job_timestamps(job)}")
        print(f"   sanity: {len(rows)} rows, feasible fraction "
              f"{float((w == 1).all(axis=1).mean()):.2f} (init was proper; expect >0.3)")
    print("\nCONNECTION TEST PASS")


if __name__ == "__main__":
    main()
