"""E8 prep — factorized fake-Heron channels + gate-fold noise amplification.

For each instance (gcp_c5 15q, gcp_v8 24q from fake_heron_bundle; gcp_p12_s0
36q, gcp_p16_s0 48q transpiled here the same way):
  1. per segment, extract the 2q pair subcircuit G from the FakeTorino-
     transpiled body (all instructions on the active physical pair);
  2. fold G -> G (G†G)^m for fold k = 2m+1 in {1,3,5,7} (device-compatible
     noise amplification: same logical unitary, k× the gate noise);
  3. build a REDUCED 2-qubit noise model: filter NoiseModel.from_backend(
     FakeTorino()).to_dict() to entries on the pair, remap indices -> exact
     calibration Kraus/T1/T2 noise on the pair;
  4. T_k[t] = (A_a ⊗ A_b) @ C_gate: 2q density-matrix sim of folded G under
     the reduced model, then the pair's asymmetric readout confusion matrices
     (from the backend roerror entries) applied analytically;
  5. spectator readout: per-qubit flip vector p_q (symmetrized from the
     roerror probabilities) over each instance's physical layout.

Output: heron_channel_bundle.pkl
  {instances: [{name, v, K, edges, chain, seed_coloring, layout,
                p_ro_vec (n,), T: {fold: (n_seg, 4, 4)}}], folds}

Validation (done in the campaign analysis, not here): fold-1 factorized fₜ
and block-locality vs E5's whole-register measurements.
"""
import sys, os, io, time, pickle
import _paths  # noqa: F401  (repo-relative sys.path setup)

import numpy as np
from qiskit import QuantumCircuit, transpile, qpy
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeTorino

from prep_fake_heron import heron_path, segment_body

FOLDS = [1, 3, 5, 7]
BACKEND = FakeTorino()
NM_FULL = NoiseModel.from_backend(BACKEND)
NM_DICT = NM_FULL.to_dict(serializable=True)


def extract_pair_subcircuit(tqc):
    """Instructions acting on the (single) physical 2q-gate pair, remapped to 2q."""
    pair = None
    for inst in tqc.data:
        if inst.operation.num_qubits == 2:
            qs = tuple(sorted(tqc.find_bit(q).index for q in inst.qubits))
            if pair is None:
                pair = qs
            elif qs != pair:
                raise RuntimeError(f"multiple 2q pairs: {pair} vs {qs}")
    assert pair is not None
    a, b = pair
    sub = QuantumCircuit(2)
    for inst in tqc.data:
        if inst.operation.name in ("measure", "barrier"):
            continue
        idx = [tqc.find_bit(q).index for q in inst.qubits]
        if all(i in (a, b) for i in idx):
            sub.append(inst.operation, [0 if i == a else 1 for i in idx])
    return sub, (a, b)


def fold_circuit(sub, k):
    m = (k - 1) // 2
    out = sub.copy()
    inv = sub.inverse()
    for _ in range(m):
        out.compose(inv, inplace=True)
        out.compose(sub, inplace=True)
    return out


def reduced_noise_model(a, b):
    """Filter the backend noise dict to qubits {a,b}, remap -> {0,1}."""
    remap = {a: 0, b: 1}
    errors = []
    for e in NM_DICT["errors"]:
        gq = e.get("gate_qubits", [])
        keep = [tuple(g) for g in gq if all(q in remap for q in g)]
        if not keep:
            continue
        e2 = dict(e)
        e2["gate_qubits"] = [[remap[q] for q in g] for g in keep]
        errors.append(e2)
    return NoiseModel.from_dict({"errors": errors})


def readout_matrices():
    """Per-qubit 2x2 confusion matrices A_q[meas, true] from roerror entries."""
    A = {}
    for e in NM_DICT["errors"]:
        if e["type"] == "roerror":
            q = e["gate_qubits"][0][0]
            P = np.array(e["probabilities"])  # rows: true state, cols: measured
            A[q] = P
    return A


def pair_T(sub_folded, a, b, A):
    """4x4 row-stochastic T[b_in, b_out] incl. pair readout. b = (bit_q0<<1)|bit_q1."""
    nm = reduced_noise_model(a, b)
    sim = AerSimulator(method="density_matrix", noise_model=nm)
    Aa, Ab = A[a], A[b]
    T = np.zeros((4, 4))
    for b_in in range(4):
        qc = QuantumCircuit(2)
        if (b_in >> 1) & 1: qc.x(0)
        if b_in & 1: qc.x(1)
        qc.compose(sub_folded, inplace=True)
        qc.save_density_matrix()
        tqc = transpile(qc, sim, optimization_level=0)
        rho = np.asarray(sim.run(tqc).result().data(0)["density_matrix"])
        p = np.clip(np.real(np.diag(rho)), 0, None); p /= p.sum()
        # rho index little-endian: idx bit0 = q0. our code: (q0<<1)|q1
        pre = np.zeros(4)
        for idx in range(4):
            q0, q1 = idx & 1, (idx >> 1) & 1
            pre[(q0 << 1) | q1] += p[idx]
        # readout: P(meas m0,m1 | true t0,t1) = Aa[t0,m0]*Ab[t1,m1]
        for code_t in range(4):
            t0, t1 = (code_t >> 1) & 1, code_t & 1
            for code_m in range(4):
                m0, m1 = (code_m >> 1) & 1, code_m & 1
                T[b_in, code_m] += pre[code_t] * Aa[t0, m0] * Ab[t1, m1]
    T = np.clip(T, 0, None)
    T /= T.sum(axis=1, keepdims=True)
    return T


def main():
    A = readout_matrices()
    with open("packs/fake_heron_bundle.pkl", "rb") as fh:
        fhb = pickle.load(fh)
    with open("packs/scale_bundle.pkl", "rb") as fh:
        sb = pickle.load(fh)
    by_name = {d["name"]: d for d in sb["instances"]}

    # PER-SEGMENT PLACEMENT: segments are independent device circuits
    # (measurement between segments), so each segment gets its own
    # initial_layout with the active pair on adjacent path slots (L[0], L[1]).
    # This is the placement a per-segment compiler pass would emit; it removes
    # SWAP routing entirely (the fixed-layout E5 transpiles routed distance-2
    # transitions through a third qubit — that overhead is a fixed-layout
    # artifact, documented as a deviation from E5 in the validation analysis).
    jobs = [(d, None) for d in fhb["instances"]] + \
           [(by_name[nm], None) for nm in ("gcp_p12_s0", "gcp_p16_s0")]

    out = {"folds": FOLDS, "backend": BACKEND.name, "instances": []}
    t0 = time.time()
    for d, _ in jobs:
        K = d.get("K") or 3
        n = d["v"] * K
        L = heron_path(n)
        subs, pairs, p_vecs = [], [], []
        for t, (u, cf, ct) in enumerate(d["chain"]):
            i_from, i_to = u * K + cf, u * K + ct
            phys = np.empty(n, dtype=int)
            phys[i_from], phys[i_to] = L[0], L[1]
            rest = [q for q in L if q not in (L[0], L[1])]
            others = [i for i in range(n) if i not in (i_from, i_to)]
            for lg, ph in zip(others, rest):
                phys[lg] = ph
            body = segment_body(n, (i_from, i_to), float(d["params"][t]))
            tqc = transpile(body, backend=BACKEND, initial_layout=list(phys),
                            optimization_level=1, seed_transpiler=42)
            sub, pair = extract_pair_subcircuit(tqc)
            subs.append(sub); pairs.append(pair)
            # spectator flip vector for THIS segment's placement
            p_vecs.append([(A[phys[i]][0, 1] + A[phys[i]][1, 0]) / 2
                           for i in range(n)])
        T_by_fold = {}
        for k in FOLDS:
            T_by_fold[k] = np.stack([pair_T(fold_circuit(subs[t], k), *pairs[t], A)
                                     for t in range(len(subs))])
        p_vecs = np.array(p_vecs)  # (n_seg, n)
        out["instances"].append(dict(
            name=d["name"], v=d["v"], K=K, edges=list(d["edges"]),
            chain=list(d["chain"]), seed_coloring=np.asarray(d["seed_coloring"]),
            layout=list(L), p_ro_vec=p_vecs, pairs=pairs, T=T_by_fold))
        print(f"[{time.time()-t0:5.0f}s] {d['name']}: n={n} segs={len(subs)} "
              f"folds={FOLDS} p_ro med={np.median(p_vecs):.4f} max={p_vecs.max():.4f}",
              flush=True)

    with open("packs/heron_channel_bundle.pkl", "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    print("wrote heron_channel_bundle.pkl",
          os.path.getsize("packs/heron_channel_bundle.pkl") / 1e6, "MB")


if __name__ == "__main__":
    main()
