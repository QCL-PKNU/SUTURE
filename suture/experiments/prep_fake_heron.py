"""Prep for E5 (hardware proxy): transpile GCP chains to the fake Heron target.

For gcp_c5 (15q) and gcp_v8 (24q), soft-edge, trained angles unchanged:
  - per segment, build the n-qubit body (2q transition composed at the active
    pair) + measure_all, transpile to FakeTorino (Heron r1: heavy-hex coupling,
    {cz, rz, sx, x, id} basis) at optimization_level=1 with a fixed
    initial_layout along a connected path of the coupling map;
  - record depth / 2q-gate counts vs the idealized CORE_BASIS circuits
    (gate-overhead table for the paper);
  - serialize QPY + initial_index_layout + pack metadata -> fake_heron_bundle.pkl.

hard_c5 (30q) is EXCLUDED: statevector trajectories at 30q ~ 8-16 GB per
instance on GPU — over the 4090's comfortable budget once noise instructions
multiply; the hardware proxy question (block-locality) is answerable at 15/24q.

The remote runner prepends X gates on physical qubits (basis-native, no
re-transpile) to set register-init states, so one transpiled body per segment
serves every init.
"""
import sys, os, io, time, pickle
import _paths  # noqa: F401  (repo-relative sys.path setup)

import numpy as np
import dill
from qiskit import QuantumCircuit, transpile, qpy
from qiskit_ibm_runtime.fake_provider import FakeTorino

from gcp_k3 import transition_circuit
from h1_pilot_lib import CORE_BASIS

BACKEND = FakeTorino()


def heron_path(n: int) -> list[int]:
    """Longest-first greedy simple path of length n on the coupling map (DFS)."""
    cmap = {tuple(sorted(e)) for e in BACKEND.coupling_map.get_edges()}
    adj = {}
    for a, b in cmap:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    def dfs(node, path, seen):
        if len(path) == n:
            return path
        for nb in sorted(adj.get(node, ())):
            if nb not in seen:
                r = dfs(nb, path + [nb], seen | {nb})
                if r:
                    return r
        return None

    for start in sorted(adj):
        r = dfs(start, [start], {start})
        if r:
            return r
    raise RuntimeError(f"no path of length {n}")


def segment_body(n: int, active: tuple[int, int], theta: float) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.compose(transition_circuit(theta), qubits=list(active), inplace=True)
    qc.measure_all()
    return qc


def main():
    with open("packs/gcp_k3_packs.dill", "rb") as fh:
        packs = dill.load(fh)

    bundle = {"backend": BACKEND.name, "instances": []}
    t0 = time.time()
    for name in ("gcp_c5", "gcp_v8"):
        pack = packs[name]
        n = pack.n
        layout = heron_path(n)
        seg_payloads, overhead = [], []
        for t, (u, cf, ct) in enumerate(pack.chain):
            active = (u * pack.K + cf, u * pack.K + ct)
            body = segment_body(n, active, float(pack.params[t]))
            # idealized reference
            ref = transpile(body, basis_gates=CORE_BASIS, optimization_level=1)
            tqc = transpile(body, backend=BACKEND, initial_layout=layout,
                            optimization_level=1, seed_transpiler=42)
            n2q = sum(1 for inst in tqc.data if inst.operation.num_qubits == 2)
            n2q_ref = sum(1 for inst in ref.data if inst.operation.num_qubits == 2)
            overhead.append(dict(seg=t, depth_ref=ref.depth(), depth_heron=tqc.depth(),
                                 gates2q_ref=n2q_ref, gates2q_heron=n2q))
            buf = io.BytesIO()
            qpy.dump(tqc, buf)
            seg_payloads.append(dict(
                qpy=buf.getvalue(), active=active,
                init_layout=list(tqc.layout.initial_index_layout()[:n])
                    if tqc.layout else layout))
        ov = np.array([[o["gates2q_ref"], o["gates2q_heron"], o["depth_ref"], o["depth_heron"]]
                       for o in overhead])
        print(f"[{time.time()-t0:5.0f}s] {name}: n={n} segs={len(seg_payloads)} "
              f"2q gates ref->heron median {np.median(ov[:,0]):.0f}->{np.median(ov[:,1]):.0f}, "
              f"depth {np.median(ov[:,2]):.0f}->{np.median(ov[:,3]):.0f}", flush=True)
        bundle["instances"].append(dict(
            name=name, v=pack.v, K=pack.K, n=n, edges=list(pack.edges),
            chain=list(pack.chain), params=np.asarray(pack.params),
            seed_coloring=np.asarray(pack.seed_coloring),
            layout=layout, segments=seg_payloads, overhead=overhead))

    with open("packs/fake_heron_bundle.pkl", "wb") as fh:
        pickle.dump(bundle, fh, protocol=4)
    print("wrote fake_heron_bundle.pkl", os.path.getsize("packs/fake_heron_bundle.pkl") / 1e6, "MB")


if __name__ == "__main__":
    main()
