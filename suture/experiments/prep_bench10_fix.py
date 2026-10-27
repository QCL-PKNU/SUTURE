"""prep_bench10_fix — regenerate the three scales the upstream basis defect
excluded from bench10: (3,3), (4,5), (5,6).

Fix (documented in reports/15 §1): when upstream find_basic_solution emits a
basis violating C·u = 0, replace it with the ANALYTIC color-swap basis —
transition (u, 0→c) flips x_u0/x_uc and lets each edge-row's slack absorb
the residual (the construction validated in the hard-edge model, gcp_hard).
The analytic basis is exact null-space by construction, all entries ±1/0,
and full rank = nullity on every instance tested. Everything else (costs,
objective, training, channels, arms) is identical to prep_bench10.

v3e3 note: only ONE edge combination exists (the triangle); its 5 cases are
distinct cost draws — matching upstream generate_gcp's repeat-and-truncate.

Output: bench10_fix_bundle.pkl (same schema as bench10_bundle.pkl paper_kv).
"""
import sys, os, time, pickle, random
import _paths  # noqa: F401  (repo-relative sys.path setup)

import numpy as np
import itertools as it

from rasengan.problems.graph_coloring_problem import GraphColoringProblem
from rasengan.utils.linear_system import greedy_simplification_of_transition_Hamiltonian
import prep_bench10 as pb

SCALES = [(3, 3), (4, 5), (5, 6)]
CASES = 5
LAYERS = 5


def analytic_basis(prb):
    C = np.asarray(prb.lin_constr_mtx, dtype=np.int64)
    A = C[:, :-1]
    n = A.shape[1]
    V, K = prb.num_graphs, prb.num_colors
    basis = []
    for u in range(V):
        for c in range(1, K):
            vec = np.zeros(n, dtype=np.int64)
            vec[u * K + 0], vec[u * K + c] = -1, +1
            for row in A:
                resid = int(row @ vec)
                if resid == 0:
                    continue
                s_cols = np.flatnonzero(row[V * K:]) + V * K
                if len(s_cols) != 1 or vec[s_cols[0]] != 0:
                    return None
                vec[s_cols[0]] = -resid // int(row[s_cols[0]])
            basis.append(vec)
    B = np.array(basis)
    if not (np.all(A @ B.T == 0) and set(np.unique(B)).issubset({-1, 0, 1})):
        return None
    if np.linalg.matrix_rank(B.astype(float)) != len(basis):
        return None
    return [b.astype(np.int8) for b in basis]


def gen_instances(v, e, want, min_value=1, max_value=20):
    """Deterministic edge-combination order; costs drawn per candidate;
    combos repeat with fresh costs when exhausted (upstream behavior)."""
    all_edges = list(it.combinations(range(v), 2))
    combos = list(it.combinations(all_edges, e))
    out = []
    i = 0
    while len(out) < want and i < want * 20:
        edges_comb = combos[i % len(combos)]
        cost_color = [random.randint(min_value, max_value) for _ in range(v)]
        out.append(GraphColoringProblem(v, list(edges_comb), cost_color))
        i += 1
    return out


def main():
    np.random.seed(0xdb); random.seed(0x7f)
    packs = []
    t0 = time.time()
    for (v, e) in SCALES:
        kept = 0
        for ci, prb in enumerate(gen_instances(v, e, CASES * 4)):
            if kept >= CASES:
                break
            V, K = prb.num_graphs, prb.num_colors
            n = len(prb.variables)
            C = np.asarray(prb.lin_constr_mtx, dtype=np.int64)
            # prefer the upstream basis when it happens to be valid; else analytic
            base = [np.asarray(r, dtype=np.int8)
                    for r in greedy_simplification_of_transition_Hamiltonian(prb.driver_bitstr)]
            A = C[:, :-1]
            upstream_ok = (len(base) > 0 and
                           all(np.all(A @ b.astype(np.int64) == 0) for b in base))
            basis_src = "upstream"
            if not upstream_ok:
                base = analytic_basis(prb)
                basis_src = "analytic"
                if base is None:
                    print(f"    SKIP v{v}e{e} c{ci}: analytic construction failed", flush=True)
                    continue
            obj = pb.kv_obj(prb.cost_color, V, K)
            F = pb.enumerate_F_kv(prb, V, K)
            if len(F) == 0:
                continue
            F_costs = np.array([obj(fb) for fb in F])
            chain_vecs = [base[i % len(base)] for i in range(LAYERS * len(base))]
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                seed_bits = np.asarray(prb.get_feasible_solution(), dtype=np.int8)
            assert all(int(r[:-1] @ seed_bits) == int(r[-1]) for r in C)
            params, e_tr = pb.train_vec_chain(chain_vecs, seed_bits, obj)
            kept += 1
            packs.append(dict(
                name=f"kv_v{v}e{e}_c{ci}", scale=(v, e), case=ci, n=n, V=V, K=K,
                cost_color=list(prb.cost_color), constr=C,
                chain_vecs=np.array(chain_vecs, dtype=np.int8),
                active_sets=[np.flatnonzero(cv).tolist() for cv in chain_vecs],
                params=params, seed_bits=seed_bits,
                F_bits=F, F_costs=F_costs, best_cost=float(F_costs.min()),
                basis_src=basis_src))
            sup = sorted({int(np.count_nonzero(cv)) for cv in chain_vecs})
            print(f"[{time.time()-t0:5.0f}s] kv_v{v}e{e}_c{ci}: n={n} |F|={len(F)} "
                  f"chain={len(chain_vecs)} sup={sup} E*={F_costs.min():.0f} "
                  f"E_tr={e_tr:.2f} basis={basis_src}", flush=True)
    with open("packs/bench10_fix_bundle.pkl", "wb") as fh:
        pickle.dump({"paper_kv": packs, "soft_k3": []}, fh, protocol=4)
    n_gpu = sum(1 for d in packs for cv in d["chain_vecs"] if np.count_nonzero(cv) >= 10)
    print(f"wrote bench10_fix_bundle.pkl ({len(packs)} packs; {n_gpu} support-10 segments -> GPU tier)")


if __name__ == "__main__":
    main()
