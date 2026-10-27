"""E7a — XY-mixer QAOA + postselection baseline (review O5's in-family gap).

The standard feasibility-preserving alternative outside the Rasengan family:
one-hot GCP with a per-vertex ring-XY mixer (preserves block Hamming weight
=> assignment feasibility), conflict cost Hamiltonian, p=2 layers.

Instance: gcp_v8 (v=8, K=3, n=24; edges from gcp_k3_packs metadata shipped
inline below to avoid dill dependency).

Protocol:
  1. Angle search, noise-free GPU statevector: shared-angle grid
     gamma in linspace(0.1, 1.3, 7) x beta in linspace(0.1, 1.3, 7),
     scored by expected conflicts of postselected feasible mass.
  2. Evaluation at best angles under FakeTorino noise (same model as E5),
     matched total budget 2048 shots x 8 seeds, postselect feasible.
Endpoints: feasible fraction, proper_seen, best_conflicts_seen (matched to
the chain arms' measurement-level endpoints).

Output: xy_baseline.csv (one row per seed + one 'search' row per grid point).
"""
import time
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeTorino

V, K = 8, 3
EDGES = [(i, (i + 1) % 8) for i in range(8)] + [(0, 2), (4, 6)]
N = V * K
SEED_COL = [0, 1, 0, 1, 0, 1, 0, 2]  # any proper-ish start not needed; use W-ish init below
P_LAYERS = 2
BUDGET = 2048
SEEDS = list(range(8))


def q(u, c):
    return u * K + c


def xy_qaoa(gamma, beta):
    qc = QuantumCircuit(N)
    # init: computational basis one-hot seed (all vertices color 0);
    # ring-XY mixer then explores each block's weight-1 manifold.
    for u in range(V):
        qc.x(q(u, 0))
    for _ in range(P_LAYERS):
        # cost: for each edge, same-color ZZ
        for (u, w) in EDGES:
            for c in range(K):
                qc.rzz(2 * gamma, q(u, c), q(w, c))
        # mixer: per-vertex ring XY (c, c+1 mod K)
        for u in range(V):
            for c in range(K):
                a, b = q(u, c), q(u, (c + 1) % K)
                qc.rxx(2 * beta, a, b)
                qc.ryy(2 * beta, a, b)
    qc.measure_all()
    return qc


def counts_to_bits(key):
    s = key.replace(" ", "")[::-1]
    return np.frombuffer(s.encode(), dtype=np.uint8)[:N] - ord("0")


def score_counts(counts):
    tot = sum(counts.values())
    feas_w, best, proper, exp_conf = 0, np.inf, False, 0.0
    for k, w in counts.items():
        bits = counts_to_bits(k)
        blocks = bits.reshape(V, K)
        if not (blocks.sum(axis=1) == 1).all():
            continue
        feas_w += w
        col = blocks.argmax(axis=1)
        cf = sum(1 for a, b in EDGES if col[a] == col[b])
        exp_conf += cf * w
        best = min(best, cf)
        proper |= (cf == 0)
    return dict(feas_frac=feas_w / tot,
                exp_conflicts=(exp_conf / feas_w) if feas_w else np.inf,
                best_conflicts_seen=(float(best) if np.isfinite(best) else np.inf),
                proper_seen=bool(proper))


def main():
    rows = []
    t0 = time.time()
    sv = AerSimulator(device="GPU", method="statevector")
    try:
        sv.run(transpile(QuantumCircuit(2), sv), shots=1).result()
    except Exception:
        sv = AerSimulator(method="statevector")
        print("GPU unavailable for search; CPU statevector", flush=True)

    # 1. noise-free angle search
    best_angles, best_score = None, np.inf
    for gamma in np.linspace(0.1, 1.3, 7):
        for beta in np.linspace(0.1, 1.3, 7):
            qc = transpile(xy_qaoa(gamma, beta), sv, optimization_level=1)
            cnt = sv.run(qc, shots=1024, seed_simulator=7).result().get_counts()
            s = score_counts(cnt)
            rows.append(dict(phase="search", gamma=gamma, beta=beta, seed=-1, **s))
            # prioritize proper-reaching angles, then expected conflicts
            key = (0 if s["proper_seen"] else 1, s["exp_conflicts"])
            if best_angles is None or key < best_score:
                best_angles, best_score = (gamma, beta), key
    print(f"[{time.time()-t0:5.0f}s] search done, best angles {best_angles}", flush=True)

    # 2. noisy evaluation at matched budget — one process per GPU (seed % 3),
    # incremental checkpointing (a timed-out job still leaves partial rows).
    gamma, beta = best_angles
    from multiprocessing import get_context

    ctx = get_context("spawn")
    with ctx.Pool(3) as pool:
        args = [(float(gamma), float(beta), seed) for seed in SEEDS]
        for res in pool.imap_unordered(_noisy_eval_one, args):
            rows.append(res)
            pd.DataFrame(rows).to_csv("xy_baseline.csv", index=False)
            print(f"[{time.time()-t0:5.0f}s] seed {res['seed']} done "
                  f"(feas={res['feas_frac']:.3f})", flush=True)
    print("wrote xy_baseline.csv (final)", flush=True)


def _noisy_eval_one(args):
    """Runs in a spawned worker pinned to one GPU."""
    import os
    gamma, beta, seed = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(seed % 3)
    from qiskit import transpile as _tp
    noise = NoiseModel.from_backend(FakeTorino())
    nsim = AerSimulator(device="GPU", method="statevector", noise_model=noise,
                        batched_shots_gpu=True)
    try:
        nsim.run(_tp(QuantumCircuit(2), nsim), shots=1).result()
    except Exception:
        nsim = AerSimulator(method="statevector", noise_model=noise)
    qc = _tp(xy_qaoa(gamma, beta), FakeTorino(), optimization_level=1,
             seed_transpiler=42)
    cnt = nsim.run(qc, shots=BUDGET, seed_simulator=100 + seed).result().get_counts()
    return dict(phase="noisy_eval", gamma=gamma, beta=beta, seed=seed,
                **score_counts(cnt))


if __name__ == "__main__":
    main()
