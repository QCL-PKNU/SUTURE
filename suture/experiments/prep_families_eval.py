"""prep_families_eval — scale-ladder FLP/SCP/KPP packs for the evaluation
campaign (report 18), bench10-style.

Ladders (scouted; gates: no zero column, C.u=0 valid basis, support <= 10):
  FLP: 3x3 (21q), 3x4 (28q), 3x5 (35q)     [d=4 fails: support 11]
  SCP: 7s5e (~18q), 10s7e (~26q), 12s8e (instance-filtered: the upstream
       basis defect reappears here — draws rejected until C.u=0 passes)
  KPP: p8b4 (32q), p12b3 (36q), p16b4 (64q) [support always 4]

Native objectives (obj_spec in pack, F-lookup abandoned at scale):
  flp: ("linear", c) with c = [cost_facilities | cost_service | 0-slack]
  scp: ("linear", c) with c = [1s over set vars | 0-slack]
  kpp: ("kpp", pairs, V, K)  obj = sum w_ab [a,b same block]
Exact best_cost: FLP/SCP by subset enumeration (2^f / 2^s); KPP by unlabeled
partition enumeration (p16b4: 2.6M partitions).
F_bits kept only where labeled |F| <= 4096 (rec_uniform oracle arm).
Feasible sampling for the T7 profiler: null-space random walk from the seed
(200 basis-move burn-in) — family-agnostic, stays in F by construction.
"""
import sys, os, io, time, random, pickle, contextlib
from itertools import combinations
import _paths  # noqa: F401  (repo-relative sys.path setup)

import numpy as np

from rasengan.problems.facility_location_problem import FacilityLocationProblem
from rasengan.problems.set_cover_problem import SetCoverProblem
from rasengan.problems.k_partition_problem import KPartitionProblem
from rasengan.utils.linear_system import greedy_simplification_of_transition_Hamiltonian
import prep_bench10 as pb

CASES = 3
LAYERS = 5
F_CAP = 4096


def valid_base(prb, max_sup=10):
    C = np.asarray(prb.lin_constr_mtx, dtype=np.int64)
    A = C[:, :-1]
    if np.any(~A.any(axis=0)):
        return None, "zero column"
    base = [np.asarray(r, dtype=np.int8)
            for r in greedy_simplification_of_transition_Hamiltonian(prb.driver_bitstr)]
    if not base:
        return None, "empty"
    if not all(np.all(A @ b.astype(np.int64) == 0) for b in base):
        return None, "C.u!=0"
    if not all(set(np.unique(b)).issubset({-1, 0, 1}) for b in base):
        return None, "entries"
    if max(int(np.count_nonzero(b)) for b in base) > max_sup:
        return None, "support"
    return base, "ok"


def enumerate_F_small(C, n, cap=F_CAP):
    if n > 24:
        return None
    xs = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)
    A, b = C[:, :-1].astype(np.int64), C[:, -1].astype(np.int64)
    ok = np.all(xs @ A.T == b[None, :], axis=1)
    F = [row.copy() for row in xs[ok]]
    return F if 0 < len(F) <= cap else None


def make_obj(spec):
    kind = spec[0]
    if kind == "linear":
        c = np.asarray(spec[1], dtype=float)
        return lambda bits: float(c @ np.asarray(bits, dtype=float))
    if kind == "kpp":
        _, pairs, V, K = spec
        def f(bits):
            x = np.asarray(bits[:V * K]).reshape(V, K)
            blk = np.argmax(x, axis=1)   # valid on feasible strings (one-hot rows)
            return float(sum(w for (a, b), w in pairs if blk[a] == blk[b]))
        return f
    raise ValueError(kind)


def best_flp(d, f, cost_service, cost_facilities):
    best = np.inf
    for r in range(1, f + 1):
        for opens in combinations(range(f), r):
            c = sum(cost_facilities[j] for j in opens)
            c += sum(min(cost_service[i][j] for j in opens) for i in range(d))
            best = min(best, c)
    return float(best)


def best_scp(s, e, cover):
    best = np.inf
    for mask in range(1, 2 ** s):
        sel = [i for i in range(s) if mask >> i & 1]
        if set().union(*(set(cover[i]) for i in sel)) >= set(range(e)):
            best = min(best, len(sel))
    return float(best)


def best_kpp(V, allot, pairs):
    """Unlabeled partition enumeration (recursion over lexicographic blocks)."""
    best = [np.inf]
    W = np.zeros((V, V))
    for (a, b), w in pairs:
        W[a, b] = W[b, a] = w
    def rec(remaining, sizes, acc):
        if acc >= best[0]:
            return
        if not sizes:
            best[0] = acc
            return
        rem = sorted(remaining)
        anchor = rem[0]
        for rest in combinations(rem[1:], sizes[0] - 1):
            blk = (anchor,) + rest
            cost = sum(W[a][b] for a, b in combinations(blk, 2))
            rec(remaining - set(blk), sizes[1:], acc + cost)
    rec(set(range(V)), sorted(allot, reverse=True), 0.0)
    return float(best[0])


def build(family, name, prb, obj_spec, best_cost, extra):
    C = np.asarray(prb.lin_constr_mtx, dtype=np.int64)
    n = len(prb.variables)
    base, why = valid_base(prb)
    if base is None:
        return None, why
    obj = make_obj(obj_spec)
    F = enumerate_F_small(C, n)
    chain_vecs = [base[i % len(base)] for i in range(LAYERS * len(base))]
    with contextlib.redirect_stdout(io.StringIO()):
        seed_bits = np.asarray(prb.get_feasible_solution(), dtype=np.int8)
    assert all(int(r[:-1] @ seed_bits) == int(r[-1]) for r in C), "seed infeasible"
    params, e_tr = pb.train_vec_chain(chain_vecs, seed_bits, obj)
    d = dict(name=name, family=family, n=n, constr=C,
             chain_vecs=np.array(chain_vecs, dtype=np.int8),
             active_sets=[np.flatnonzero(cv).tolist() for cv in chain_vecs],
             params=params, seed_bits=seed_bits,
             obj_spec=obj_spec, best_cost=best_cost,
             e_trained=float(e_tr), basis_src="upstream", **extra)
    if F is not None:
        d["F_bits"] = F
        d["F_costs"] = np.array([obj(fb) for fb in F])
        assert abs(d["F_costs"].min() - best_cost) < 1e-9, \
            f"{name}: enum best {d['F_costs'].min()} != structural {best_cost}"
    return d, "ok"


def main():
    random.seed(0x18E); np.random.seed(0x18E)
    packs, t0 = [], time.time()

    # ---- FLP ----
    for (dd, ff) in ((3, 3), (3, 4), (3, 5)):
        kept = 0
        for ci in range(CASES * 6):
            cs = [[random.randint(1, 20) for _ in range(ff)] for _ in range(dd)]
            cf = [random.randint(1, 20) for _ in range(ff)]
            prb = FacilityLocationProblem(dd, ff, cs, cf)
            nv = len(prb.variables)
            c_vec = np.zeros(nv)
            c_vec[:ff] = cf
            c_vec[ff:ff + dd * ff] = np.asarray(cs, dtype=float).ravel()
            pk, why = build("flp", f"flp_{dd}x{ff}_e{kept}", prb,
                            ("linear", c_vec), best_flp(dd, ff, cs, cf), {})
            if pk is None:
                continue
            packs.append(pk); kept += 1
            if kept >= CASES: break

    # ---- SCP ----
    for (ss, ee) in ((7, 5), (10, 7), (12, 8)):
        kept = 0; rejects = {}
        for ci in range(CASES * 25):
            cov = [[] for _ in range(ss)]
            pts = list(range(ee)); random.shuffle(pts)
            for i, pt in enumerate(pts): cov[i % ss].append(pt)
            for i in range(ss):
                for pt in range(ee):
                    if pt not in cov[i] and random.random() < 0.3: cov[i].append(pt)
            cov = [sorted(set(c)) for c in cov if c]
            prb = SetCoverProblem(len(cov), ee, cov)
            nv = len(prb.variables)
            c_vec = np.zeros(nv); c_vec[:len(cov)] = 1.0
            pk, why = build("scp", f"scp_{ss}s{ee}e_e{kept}", prb,
                            ("linear", c_vec), best_scp(len(cov), ee, cov), {})
            if pk is None:
                rejects[why] = rejects.get(why, 0) + 1
                continue
            packs.append(pk); kept += 1
            if kept >= CASES: break
        print(f"  scp {ss}s{ee}e: kept {kept}, rejects {rejects}", flush=True)

    # ---- KPP ----
    for (V, B) in ((8, 4), (12, 3), (16, 4)):
        kept = 0
        n_pairs = V
        for ci in range(CASES * 6):
            allot = [V // B + (1 if i < V % B else 0) for i in range(B)]
            prs, seen = [], set()
            while len(prs) < n_pairs:
                a, b = random.sample(range(V), 2)
                if (a, b) not in seen and (b, a) not in seen:
                    seen.add((a, b)); prs.append(((a, b), random.randint(1, 20)))
            prb = KPartitionProblem(V, allot, prs)
            pk, why = build("kpp", f"kpp_p{V}b{B}_e{kept}", prb,
                            ("kpp", prs, V, B), best_kpp(V, allot, prs),
                            dict(V=V, K=B))
            if pk is None:
                continue
            packs.append(pk); kept += 1
            if kept >= CASES: break

    for pk in packs:
        sup = sorted({int(np.count_nonzero(cv)) for cv in pk["chain_vecs"]})
        print(f"[{time.time()-t0:6.0f}s] {pk['name']}: n={pk['n']} "
              f"|F|={'enum:'+str(len(pk['F_bits'])) if 'F_bits' in pk else 'large'} "
              f"chain={len(pk['chain_vecs'])} sup={sup} E*={pk['best_cost']:.0f} "
              f"E_tr={pk['e_trained']:.2f}", flush=True)
    with open("packs/families_eval_bundle.pkl", "wb") as fh:
        pickle.dump({"paper_kv": packs, "soft_k3": []}, fh, protocol=4)
    print(f"wrote families_eval_bundle.pkl ({len(packs)} packs)")


if __name__ == "__main__":
    main()
