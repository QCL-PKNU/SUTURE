"""prep_families — FLP/SCP/KPP packs for the generalization study (T7).

Same schema as bench10 paper_kv packs, plus family tag and a generic
objective: F_costs are precomputed for every feasible string at build time,
and the runner looks costs up by string (run_chain only evaluates obj on
feasible strings — raw measured strings are decoded/filtered first).

Scales (all n <= 22 so F is brute-force enumerable):
  FLP: (2,2) n=10, (2,3) n=15, (3,3) n=21     [demand one-hot + linking+slack]
  SCP: (5,4) n=9,  (6,4) n=16, (7,5) n<=22    [>=1 cover rows, unary slack]
  KPP: (4,2) n=8,  (5,2) n=10, (6,3) n=18     [point one-hot + block cardinality]
3 cases per scale; mandatory C.u=0 validity gate (upstream basis kept when
valid — all scouted scales pass, unlike GCP-KV's (3,3)/(4,5)/(5,6)).
"""
import sys, os, io, time, random, pickle, contextlib
import _paths  # noqa: F401  (repo-relative sys.path setup)

import numpy as np

from rasengan.problems.facility_location_problem import FacilityLocationProblem
from rasengan.problems.set_cover_problem import SetCoverProblem
from rasengan.problems.k_partition_problem import KPartitionProblem
from rasengan.utils.linear_system import greedy_simplification_of_transition_Hamiltonian
import prep_bench10 as pb

CASES = 3
LAYERS = 5


def enumerate_F(C, n):
    xs = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)
    A, b = C[:, :-1].astype(np.int64), C[:, -1].astype(np.int64)
    ok = np.all(xs @ A.T == b[None, :], axis=1)
    return [row.copy() for row in xs[ok]]


def valid_base(prb):
    C = np.asarray(prb.lin_constr_mtx, dtype=np.int64)
    A = C[:, :-1]
    base = [np.asarray(r, dtype=np.int8)
            for r in greedy_simplification_of_transition_Hamiltonian(prb.driver_bitstr)]
    if not base:
        return None
    if not all(np.all(A @ b.astype(np.int64) == 0) for b in base):
        return None
    if not all(set(np.unique(b)).issubset({-1, 0, 1}) for b in base):
        return None
    return base


def obj_fn(prb):
    varnames = [v.name for v in prb.variables]
    def f(bits):
        return float(prb.obj_func([int(x) for x in bits]))
    return f


def build_pack(family, name, prb):
    C = np.asarray(prb.lin_constr_mtx, dtype=np.int64)
    n = len(prb.variables)
    base = valid_base(prb)
    if base is None:
        return None
    obj = obj_fn(prb)
    F = enumerate_F(C, n)
    if not (0 < len(F) <= 4096):
        return None
    F_costs = np.array([obj(fb) for fb in F])
    chain_vecs = [base[i % len(base)] for i in range(LAYERS * len(base))]
    with contextlib.redirect_stdout(io.StringIO()):
        seed_bits = np.asarray(prb.get_feasible_solution(), dtype=np.int8)
    assert all(int(r[:-1] @ seed_bits) == int(r[-1]) for r in C), "seed infeasible"
    params, e_tr = pb.train_vec_chain(chain_vecs, seed_bits, obj)
    return dict(
        name=name, family=family, n=n,
        constr=C, chain_vecs=np.array(chain_vecs, dtype=np.int8),
        active_sets=[np.flatnonzero(cv).tolist() for cv in chain_vecs],
        params=params, seed_bits=seed_bits,
        F_bits=F, F_costs=F_costs, best_cost=float(F_costs.min()),
        e_trained=float(e_tr), basis_src="upstream")


def main():
    random.seed(0x17); np.random.seed(0x17)
    packs, t0 = [], time.time()

    # ---- FLP ----
    for (d, f) in ((2, 2), (2, 3), (3, 3)):
        kept = 0
        for ci in range(CASES * 4):
            cs = [[random.randint(1, 20) for _ in range(f)] for _ in range(d)]
            cf = [random.randint(1, 20) for _ in range(f)]
            p = FacilityLocationProblem(d, f, cs, cf)
            pk = build_pack("flp", f"flp_{d}x{f}_c{kept}", p)
            if pk is None: continue
            packs.append(pk); kept += 1
            if kept >= CASES: break

    # ---- SCP ----
    for (s, e) in ((5, 4), (6, 4), (7, 5)):
        kept = 0
        for ci in range(CASES * 6):
            cov = [[] for _ in range(s)]
            pts = list(range(e)); random.shuffle(pts)
            for i, pt in enumerate(pts): cov[i % s].append(pt)
            for i in range(s):
                for pt in range(e):
                    if pt not in cov[i] and random.random() < 0.35: cov[i].append(pt)
            p = SetCoverProblem(s, e, cov)
            if len(p.variables) > 22: continue
            pk = build_pack("scp", f"scp_{s}s{e}e_c{kept}", p)
            if pk is None: continue
            packs.append(pk); kept += 1
            if kept >= CASES: break

    # ---- KPP ----
    for (pts_n, blocks, pairs) in ((4, 2, 3), (5, 2, 4), (6, 3, 5)):
        kept = 0
        for ci in range(CASES * 6):
            allot = [pts_n // blocks + (1 if i < pts_n % blocks else 0) for i in range(blocks)]
            prs, seen = [], set()
            while len(prs) < pairs:
                a, b = random.sample(range(pts_n), 2)
                if (a, b) not in seen and (b, a) not in seen:
                    seen.add((a, b)); prs.append(((a, b), random.randint(1, 20)))
            p = KPartitionProblem(pts_n, allot, prs)
            if len(p.variables) > 22: continue
            pk = build_pack("kpp", f"kpp_p{pts_n}b{blocks}_c{kept}", p)
            if pk is None: continue
            packs.append(pk); kept += 1
            if kept >= CASES: break

    for pk in packs:
        sup = sorted({int(np.count_nonzero(cv)) for cv in pk["chain_vecs"]})
        print(f"[{time.time()-t0:5.0f}s] {pk['name']}: n={pk['n']} |F|={len(pk['F_bits'])} "
              f"chain={len(pk['chain_vecs'])} sup={sup} E*={pk['best_cost']:.0f} "
              f"E_tr={pk['e_trained']:.2f}", flush=True)
    with open("packs/families_bundle.pkl", "wb") as fh:
        pickle.dump({"paper_kv": packs, "soft_k3": []}, fh, protocol=4)
    print(f"wrote families_bundle.pkl ({len(packs)} packs)")


if __name__ == "__main__":
    main()
