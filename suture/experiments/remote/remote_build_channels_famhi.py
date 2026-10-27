"""remote_build_channels — parallel exact-channel build on the GPU server.

Self-contained (ships with hdi_decompose.py + mcx_decompose.py): synthesizes
each transition with the UPSTREAM driver_component, density-matrix sim under
scaled depolarizing noise, per (segment, feasible-restriction) exact output
distributions. Multiprocessing over (instance, segment) tasks.

Output: families_channels_hi.pkl (same schema as prep_bench10_channels).
"""
import os, sys, time, pickle

# MUST precede any numpy/qiskit import: each Pool worker otherwise spawns a
# full OMP thread set (30 workers x 32 threads thrashed the 32-core server to
# load ~580 and stalled the build at 200/1458).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("QISKIT_IN_PARALLEL", "TRUE")   # disable aer's own fork parallelism

from multiprocessing import Pool, cpu_count

import numpy as np

LAMS = [32.0, 64.0]
BASE_P1, BASE_P2 = 3.5e-4, 8.75e-3
CORE_BASIS = ["id", "rz", "sx", "x", "cx"]


def make_noise_model(lam):
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    if lam <= 0:
        return None
    nm = NoiseModel(basis_gates=CORE_BASIS)
    nm.add_all_qubit_quantum_error(depolarizing_error(min(BASE_P1 * lam, 0.5), 1),
                                   ["id", "rz", "sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(min(BASE_P2 * lam, 0.75), 2), ["cx"])
    return nm


def seg_task(args):
    """One (instance, segment): distributions for every restriction x lam.

    args may carry an optional 7th element (device_spec):
      None / "cpu1"  -> CPU, 1 thread   (small tasks, wide pool)
      "cpu4"         -> CPU, 4 threads  (support-8)
      "gpu:<k>"      -> GPU k via CUDA_VISIBLE_DEVICES (support-10: a 12q
                        density matrix is a 24q-statevector-sized object;
                        measured 206s/circuit on 1 CPU thread, ~30s on 8
                        threads — the 4090s are the only way the 100 v5e4
                        support-10 tasks finish in hours, not weeks)
    """
    if len(args) == 7:
        name, kind, vec_or_tr, theta, restr_list, seed, dev = args
    else:
        name, kind, vec_or_tr, theta, restr_list, seed = args
        dev = "cpu1"
    sim_kwargs = {"max_parallel_threads": 1}
    if dev == "cpu4":
        sim_kwargs = {"max_parallel_threads": 4}
    elif dev.startswith("gpu:"):
        os.environ["CUDA_VISIBLE_DEVICES"] = dev.split(":")[1]
        sim_kwargs = {"device": "GPU", "max_parallel_threads": 2}
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Parameter
    from qiskit_aer import AerSimulator
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hdi_decompose import driver_component

    if kind == "kv":
        vec = np.asarray(vec_or_tr, dtype=np.int8)
        A = np.flatnonzero(vec)
        m = len(A)
        par = Parameter("t")
        qc0 = QuantumCircuit(m + 2)
        hdi = [0 if x == -1 else 1 for x in vec[A]]
        driver_component(qc0, list(range(m)), [m, m + 1], hdi, par, "constant")
        bound = qc0.assign_parameters([float(theta)])
    else:   # soft pair: 2-qubit transition, restrictions are the 4 basis states
        par = Parameter("t")
        qc0 = QuantumCircuit(4)   # 2 + 2 ancillas, same synthesis path
        driver_component(qc0, [0, 1], [2, 3], [0, 1], par, "constant")
        bound = qc0.assign_parameters([float(theta)])
        m = 2

    # transpile the BODY once (noise model does not affect transpilation);
    # X-prefixes are already in CORE_BASIS, so per-restriction circuits need
    # no further transpile. Previous version re-transpiled per (restriction,
    # lam) — ~300x per segment — which made the build ~50x slower.
    ref_sim = AerSimulator(method="density_matrix", **sim_kwargs)
    body = transpile(bound, ref_sim, basis_gates=CORE_BASIS, optimization_level=0)

    circuits = []
    for rbits in restr_list:
        qc = QuantumCircuit(body.num_qubits)
        for i, b in enumerate(rbits):
            if b: qc.x(i)
        qc.compose(body, inplace=True)
        qc.save_density_matrix(qubits=list(range(m)))
        circuits.append(qc)

    out = {}
    for lam in LAMS:
        nm = make_noise_model(lam)
        sim = AerSimulator(method="density_matrix", noise_model=nm,
                           seed_simulator=seed, **sim_kwargs)
        res = sim.run(circuits).result()   # one batched job per lam
        for ri, rbits in enumerate(restr_list):
            rho = np.asarray(res.data(ri)["density_matrix"])
            p = np.clip(np.real(np.diag(rho)), 0, None)
            p /= p.sum()
            nz = np.flatnonzero(p > 1e-9)
            codes = []
            for idx in nz:
                code = 0
                for i in range(m):
                    code |= ((idx >> i) & 1) << i
                codes.append(code)
            out[(lam, bytes(list(rbits)))] = (np.array(codes, dtype=np.int64),
                                              p[nz] / p[nz].sum())
    return name, kind, out


def main():
    with open("families_bundle.pkl", "rb") as fh:
        bundle = pickle.load(fh)

    tasks = []
    for d in bundle["paper_kv"]:
        F = d["F_bits"]
        for t, vec in enumerate(d["chain_vecs"]):
            A = np.flatnonzero(vec)
            restr = sorted({tuple(int(b) for b in fb[A]) for fb in F})
            tasks.append(((d["name"], t), "kv", vec, d["params"][t], restr, 101 + t))
    for d in bundle["soft_k3"]:
        for t in range(len(d["chain"])):
            restr = [(0, 0), (0, 1), (1, 0), (1, 1)]
            tasks.append(((d["name"], t), "soft", None, d["params"][t], restr, 101 + t))

    # resume support: shards.pkl (staged from a previous timed-out run) holds
    # already-computed tasks; only the remainder is dispatched.
    results = {}
    if os.path.exists("shards.pkl"):
        with open("shards.pkl", "rb") as fh:
            results = pickle.load(fh)
        print(f"resumed {len(results)} tasks from shards.pkl", flush=True)
    todo = [t for t in tasks if t[0] not in results]

    # tier by support size: support<=6 -> cpu1 wide pool; support 8 -> cpu4;
    # support 10 -> the 3 GPUs round-robin (12q density matrix).
    def support(t):
        return 2 if t[1] == "soft" else int(np.count_nonzero(t[2]))
    small = [t + ("cpu1",) for t in todo if support(t) <= 6]
    mid = [t + ("cpu4",) for t in todo if support(t) == 8]
    big = [t + (f"gpu:{i % 3}",) for i, t in enumerate(t for t in todo if support(t) >= 10)]
    print(f"{len(todo)}/{len(tasks)} tasks: {len(small)} cpu1, {len(mid)} cpu4, "
          f"{len(big)} gpu", flush=True)

    t0 = time.time()
    done_ct = 0

    def drain(pool_size, tset, label):
        nonlocal done_ct
        if not tset:
            return
        # maxtasksperchild=1: CUDA_VISIBLE_DEVICES is read at context init, so
        # a reused worker would stay bound to its first GPU; fresh process per
        # task keeps the round-robin honest (and is noise for CPU tiers).
        with Pool(pool_size, maxtasksperchild=1) as pool:
            for key, kind, out in pool.imap_unordered(seg_task, tset, chunksize=1):
                results[key] = (kind, out)
                done_ct += 1
                if done_ct % 25 == 0:
                    print(f"[{time.time()-t0:6.0f}s] {done_ct}/{len(todo)} ({label})",
                          flush=True)
                    with open("shards.tmp", "wb") as fh:
                        pickle.dump(results, fh, protocol=4)
                    os.replace("shards.tmp", "shards.pkl")   # atomic checkpoint

    drain(3, big, "gpu")          # 3 workers, one per GPU — run first (longest)
    drain(7, mid, "cpu4")         # 7 x 4 threads = 28 cores
    drain(30, small, "cpu1")

    # reassemble into the prep_bench10_channels schema
    chans = {"lams": LAMS, "kv": {}, "soft": {}}
    for d in bundle["paper_kv"]:
        by_lam = {lam: {} for lam in LAMS}
        for t in range(len(d["chain_vecs"])):
            kind, out = results[(d["name"], t)]
            for (lam, r), v in out.items():
                by_lam[lam][(t, r)] = v
        chans["kv"][d["name"]] = by_lam
    for d in bundle["soft_k3"]:
        by_lam = {}
        for lam in LAMS:
            T = np.zeros((len(d["chain"]), 4, 4))
            for t in range(len(d["chain"])):
                _, out = results[(d["name"], t)]
                for b_in in range(4):
                    rb = bytes([(b_in >> 1) & 1, b_in & 1])
                    codes, probs = out[(lam, rb)]
                    for c, p in zip(codes, probs):
                        # code little-endian (q0=bit0) -> b_out (q0<<1)|q1
                        b_out = ((c & 1) << 1) | ((c >> 1) & 1)
                        T[t, b_in, b_out] += p
            by_lam[lam] = T
        chans["soft"][d["name"]] = by_lam

    with open("families_channels_hi.pkl", "wb") as fh:
        pickle.dump(chans, fh, protocol=4)
    print(f"wrote families_channels_hi.pkl "
          f"{os.path.getsize('families_channels_hi.pkl')/1e6:.1f} MB "
          f"in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
