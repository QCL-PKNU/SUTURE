"""compute_restrictions — exact reachable input patterns per active set.

Register inputs to every segment are FEASIBLE strings in every arm (purify
keeps feasible; repair/junk repair to feasible; rec_uniform reinjects from
F), so the reachable restriction set of segment t = { x_A : x in F } — the
projection of F onto the active set A. For packs without enumerable F this
is computed exactly per pattern via a MILP feasibility check (scipy.milp):
does C x = b admit a 0/1 solution with x_A fixed to the pattern?

Patterns are computed once per UNIQUE active set (chain repeats basis
vectors LAYERS times) and stored in the pack as restr_map
{active_set_tuple: [patterns]}. Packs with F_bits get the same map from
direct projection (and the MILP path is cross-checked against it on the
enumerable packs).
"""
import sys, os, time, pickle
import _paths  # noqa: F401  (repo-relative sys.path setup)
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def reachable_milp(A, b, act, cache):
    k = len(act)
    n = A.shape[1]
    pats = []
    for m in range(2 ** k):
        pat = tuple(int(m >> i & 1) for i in range(k))
        lb = np.zeros(n); ub = np.ones(n)
        for j, bit in zip(act, pat):
            lb[j] = ub[j] = bit
        res = milp(c=np.zeros(n),
                   constraints=LinearConstraint(A, b, b),
                   integrality=np.ones(n), bounds=Bounds(lb, ub))
        if res.success:
            pats.append(pat)
    return sorted(pats)


def main():
    with open("packs/families_eval_bundle.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    t0 = time.time()
    for d in bundle["paper_kv"]:
        C = np.asarray(d["constr"], dtype=np.int64)
        A, b = C[:, :-1].astype(float), C[:, -1].astype(float)
        restr_map = {}
        uniq = []
        seen = set()
        for cv in d["chain_vecs"]:
            key = tuple(np.flatnonzero(cv).tolist())
            if key not in seen:
                seen.add(key); uniq.append(key)
        for act in uniq:
            if "F_bits" in d:
                pats = sorted({tuple(int(x) for x in fb[list(act)]) for fb in d["F_bits"]})
                # cross-check MILP path on enumerable packs (first set only)
                if act == uniq[0]:
                    pats_milp = reachable_milp(A, b, act, None)
                    assert pats_milp == pats, f"{d['name']}: MILP != projection"
            else:
                pats = reachable_milp(A, b, act, None)
            assert pats, f"{d['name']}: empty reachable set for {act}"
            restr_map[act] = pats
        d["restr_map"] = restr_map
        sizes = sorted({len(v) for v in restr_map.values()})
        print(f"[{time.time()-t0:5.0f}s] {d['name']}: {len(uniq)} unique active sets, "
              f"|patterns| in {sizes} (full grid would be "
              f"{sorted({2**len(a) for a in uniq})})", flush=True)
    with open("packs/families_eval_bundle.pkl", "wb") as fh:
        pickle.dump(bundle, fh, protocol=4)
    print("bundle updated with restr_map")


if __name__ == "__main__":
    main()
