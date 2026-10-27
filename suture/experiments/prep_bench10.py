"""prep_bench10 — GCP-only benchmark packs for the Fig 9/10-style evaluation.

Family A ("paper_kv"): instances from the upstream generate_gcp (K = V one-hot
model, x[i,k] assignment rows == 1, edge rows x_ik + x_jk + s = 1 with slack,
random per-color costs; objective = total cost of colors USED). Paper seeds
(np.random.seed(0xdb), random.seed(0x7f), as reproduce/figure_10/run_backend.py
sets them). Scales (v, e) x 5 cases. Chains: upstream driver_bitstr ->
greedy_simplification_of_transition_Hamiltonian, repeated LAYERS=5 times
(paper Fig 10 runs layer=5), one trained angle per chain slot.

Family B ("soft_k3"): our proper-coloring benchmark — soft-edge fixed-K=3
(assignment-only feasibility, cost = edge conflicts): gcp_c5, gcp_v8,
planted 3-colorable v in {12, 16} x 3 graph seeds (prep_scale construction:
balanced 3-partition, mean degree 4).

Both: trained noise-free by COBYLA(300) on the exact analytic channel
(cos^2/sin^2 two-level mixing), identical trained chain shared by all arms.

PROTOCOL NOTE (documented deviation, carried from reports 07-12): both
pipelines run at MAXIMAL segmentation — measurement after every transition
(num_segments = chain length). The paper's num_segments=10 groups several
transitions coherently; per-transition segmentation is what our runtime is
designed for and what the exact active-set channels require. The paper's
Fig 10 numbers are therefore a qualitative anchor, not a bit-exact target;
the lambda=0 validation gate is internal consistency (in_constraints_probs
~= 1, ARG == exact chain expectation), not CSV equality.

Output: bench10_bundle.pkl
  {"paper_kv": [ {name, scale, case, n, V, K, cost_color, constr (rows x n+1),
                  chain_vecs (L x n int8, +-1/0), active_sets, params,
                  seed_bits, F_bits (|F| x n int8), F_costs, best_cost} ],
   "soft_k3":  [ {name, v, K, edges, chain (u,cf,ct), params, seed_coloring} ]}
"""
import sys, os, time, pickle, random
import _paths  # noqa: F401  (repo-relative sys.path setup)

import numpy as np
from itertools import product
from scipy.optimize import minimize

from rasengan.problems.graph_coloring_problem import generate_gcp
from rasengan.utils.linear_system import greedy_simplification_of_transition_Hamiltonian

LAYERS = 5
SCALES = [(3, 2), (3, 3), (4, 3), (4, 4), (4, 5), (5, 4), (5, 6)]
CASES = 5
TRAIN_ITERS = 300


# ---------------------------------------------------------------------------
# Family A: paper K=V packs
# ---------------------------------------------------------------------------

def kv_obj(cost_color, V, K):
    """Upstream objective: sum_k cost_k * (1 - prod_i (1 - x_ik)) — cost of used colors."""
    cc = np.asarray(cost_color, dtype=float)
    def f(bits):
        x = np.asarray(bits[:V * K], dtype=np.int8).reshape(V, K)
        used = 1 - np.prod(1 - x, axis=0)
        return float(cc @ used)
    return f


def enumerate_F_kv(prb, V, K):
    """All feasible bit-vectors: proper colorings with K=V colors, slack filled per row."""
    C = np.asarray(prb.lin_constr_mtx, dtype=np.int64)
    n = C.shape[1] - 1
    nx = V * K
    edges = prb.pairs_adjacent
    F = []
    for col in product(range(K), repeat=V):
        if any(col[a] == col[b] for a, b in edges):
            continue
        bits = np.zeros(n, dtype=np.int8)
        for i, c in enumerate(col):
            bits[i * K + c] = 1
        ok = True
        for row in C:
            lhs_x = int(row[:nx] @ bits[:nx])
            s_cols = np.flatnonzero(row[nx:n]) + nx
            resid = int(row[-1]) - lhs_x
            if len(s_cols) == 0:
                if resid != 0: ok = False; break
            elif len(s_cols) == 1 and row[s_cols[0]] == 1:
                if resid in (0, 1): bits[s_cols[0]] = resid
                else: ok = False; break
            else:  # unexpected slack structure
                ok = False; break
        if ok:
            F.append(bits)
    return np.array(F, dtype=np.int8)


TRAIN_SUPPORT_CAP = 512   # top-M truncation of the register between steps:
                          # without it the exact support doubles per step
                          # (2^chain at v>=12); with it training is a bounded
                          # approximation — identical across arms, so the
                          # matched-chain comparison is unaffected.


class AnalyticVecChannel:
    """Noise-free two-level mixing for +-1 transition vectors (exact register propagation)."""
    def __init__(self, chain_vecs):
        self.vecs = chain_vecs
        self.pats = []
        for vec in chain_vecs:
            A = np.flatnonzero(vec)
            a = (vec[A] == 1).astype(np.int8)   # pattern reached by +vec
            self.pats.append((A, a.tobytes(), (1 - a).tobytes()))

    def step(self, t, register, theta):
        A, pa, pb = self.pats[t]
        c2, s2 = np.cos(theta) ** 2, np.sin(theta) ** 2
        out = {}
        vec = self.vecs[t]
        for kb, w in register.items():
            bits = np.frombuffer(kb, dtype=np.int8)
            r = bits[A].tobytes()
            if r == pa:
                tgt = (bits - vec).astype(np.int8)
            elif r == pb:
                tgt = (bits + vec).astype(np.int8)
            else:
                out[kb] = out.get(kb, 0.0) + w
                continue
            out[kb] = out.get(kb, 0.0) + w * c2
            tb = tgt.tobytes()
            out[tb] = out.get(tb, 0.0) + w * s2
        return out


def train_vec_chain(chain_vecs, seed_bits, obj, iters=TRAIN_ITERS):
    ch = AnalyticVecChannel(chain_vecs)
    seed_kb = np.asarray(seed_bits, dtype=np.int8).tobytes()

    def objective(params):
        reg = {seed_kb: 1.0}
        for t in range(len(chain_vecs)):
            reg = ch.step(t, reg, params[t])
            if len(reg) > TRAIN_SUPPORT_CAP:
                keep = sorted(reg.items(), key=lambda kv: -kv[1])[:TRAIN_SUPPORT_CAP]
                tot = sum(w for _, w in keep)
                reg = {k: w / tot for k, w in keep}
        return sum(obj(np.frombuffer(k, dtype=np.int8)) * w for k, w in reg.items())

    x0 = np.full(len(chain_vecs), np.pi / 4)
    res = minimize(objective, x0, method="COBYLA",
                   options={"maxiter": iters, "rhobeg": 0.5})
    params = res.x if objective(res.x) <= objective(x0) else x0
    return np.asarray(params), objective(params)


def gen_kv_instances(v, e, want, min_value=1, max_value=20):
    """Bounded reimplementation of upstream generate_gcp's inner loop.

    Upstream retries cost_color in a `while True` until the driver-basis
    range check passes — but driver_bitstr depends only on the constraint
    matrix (the edge set), NOT the costs, so a bad edge combination loops
    forever (confirmed: the OVERSAMPLE build hung inside generate_gcp).
    Here: iterate edge combinations in the same deterministic order, draw
    costs ONCE per combination, keep combos passing the upstream range
    check, stop at `want`."""
    import itertools as it
    from rasengan.problems.graph_coloring_problem import GraphColoringProblem
    all_edges = list(it.combinations(range(v), 2))
    out = []
    for edges_comb in it.combinations(all_edges, e):
        cost_color = [random.randint(min_value, max_value) for _ in range(v)]
        prb = GraphColoringProblem(v, list(edges_comb), cost_color)
        if all(x in (-1, 0, 1) for row in prb.driver_bitstr for x in row):
            out.append(prb)
            if len(out) >= want:
                break
    return out


OVERSAMPLE = 30   # candidates per scale passing the upstream range check;
                  # our stricter C u = 0 gate then keeps the first CASES


def build_paper_family():
    # paper seeds (reproduce/figure_10/run_backend.py)
    np.random.seed(0xdb); random.seed(0x7f)
    packs = []
    t0 = time.time()
    for si, (v, e) in enumerate(SCALES):
        problems = gen_kv_instances(v, e, OVERSAMPLE)
        kept = 0
        for ci, prb in enumerate(problems):
            if kept >= CASES:
                break
            V = prb.num_graphs; K = prb.num_colors
            n = len(prb.variables)
            obj = kv_obj(prb.cost_color, V, K)
            F = enumerate_F_kv(prb, V, K)
            assert len(F) > 0, f"empty F at scale {v},{e} case {ci}"
            F_costs = np.array([obj(fb) for fb in F])
            base = greedy_simplification_of_transition_Hamiltonian(prb.driver_bitstr)
            base = [np.asarray(r, dtype=np.int8) for r in base]
            # VALIDITY GATE: every transition vector must be in the null space of C
            # (upstream find_basic_solution has a documented numeric edge case that
            # can emit invalid bases; an invalid vector lets the chain leave F,
            # which shows up as E_tr < E* — silently corrupting every metric).
            A = np.asarray(prb.lin_constr_mtx, dtype=np.int64)[:, :-1]
            bad = [i for i, r in enumerate(base) if np.any(A @ r.astype(np.int64) != 0)]
            if bad:
                print(f"    SKIP kv_v{v}e{e}_c{ci}: {len(bad)}/{len(base)} chain vectors "
                      f"violate C u = 0 (upstream basis edge case)", flush=True)
                continue
            chain_vecs = [base[i % len(base)] for i in range(LAYERS * len(base))]
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                seed_bits = np.asarray(prb.get_feasible_solution(), dtype=np.int8)
            params, e_tr = train_vec_chain(chain_vecs, seed_bits, obj)
            kept += 1
            packs.append(dict(
                name=f"kv_v{v}e{e}_c{ci}", scale=(v, e), case=ci, n=n, V=V, K=K,
                cost_color=list(prb.cost_color),
                constr=np.asarray(prb.lin_constr_mtx, dtype=np.int64),
                chain_vecs=np.array(chain_vecs, dtype=np.int8),
                active_sets=[np.flatnonzero(cv).tolist() for cv in chain_vecs],
                params=params, seed_bits=seed_bits,
                F_bits=F, F_costs=F_costs, best_cost=float(F_costs.min())))
            print(f"[{time.time()-t0:6.0f}s] kv_v{v}e{e}_c{ci}: n={n} |F|={len(F)} "
                  f"chain={len(chain_vecs)} E*={F_costs.min():.0f} E_tr={e_tr:.2f}", flush=True)
    return packs


# ---------------------------------------------------------------------------
# Family B: soft-edge K=3 proper-coloring packs
# ---------------------------------------------------------------------------

def planted_graph(v, seed, target_deg=4.0):
    rng = np.random.default_rng(seed * 1000 + v)
    part = np.array([i % 3 for i in range(v)])
    rng.shuffle(part)
    m = int(target_deg * v / 2)
    edges = set()
    while len(edges) < m:
        u, w = rng.integers(0, v, 2)
        if u != w and part[u] != part[w]:
            edges.add((min(u, w), max(u, w)))
    return sorted(edges)


def build_soft_family():
    from gcp_k3 import GcpK3Pack
    packs = []
    specs = [("gcp_c5", 5, [(0,1),(1,2),(2,3),(3,4),(4,0)]),
             ("gcp_v8", 8, [(i,(i+1)%8) for i in range(8)] + [(0,2),(4,6)])]
    for v in (12, 16):
        for gs in range(3):
            specs.append((f"gcp_p{v}_s{gs}", v, planted_graph(v, gs)))
    t0 = time.time()
    for name, v, edges in specs:
        # chain: 2 rounds x per-vertex swaps (0->1, 0->2), the reports-07/08 design
        chain = [(u, 0, c) for _ in range(2) for u in range(v) for c in (1, 2)]
        pack = GcpK3Pack(name=name, v=v, K=3, edges=list(edges), chain=chain,
                         params=np.full(len(chain), np.pi / 4),
                         seed_coloring=np.zeros(v, dtype=np.int64))
        # chain as +-1 vectors for the shared trainer
        vecs = []
        for (u, cf, ct) in pack.chain:
            vec = np.zeros(pack.n, dtype=np.int8)
            vec[u * 3 + cf], vec[u * 3 + ct] = -1, 1
            vecs.append(vec)
        seed_bits = pack.coloring_to_bits(pack.seed_coloring)
        adj = [[] for _ in range(v)]
        for a, b in edges:
            adj[a].append(b); adj[b].append(a)
        def conf_obj(bits, v=v, adj=adj):
            col = np.argmax(np.asarray(bits).reshape(v, 3), axis=1)
            return float(sum(1 for u in range(v) for w in adj[u] if w > u and col[u] == col[w]))
        params, e_tr = train_vec_chain(vecs, seed_bits, conf_obj)
        pack.params = params
        packs.append(dict(name=name, v=v, K=3, edges=list(edges),
                          chain=list(pack.chain), params=params,
                          seed_coloring=np.asarray(pack.seed_coloring)))
        print(f"[{time.time()-t0:6.0f}s] {name}: n={v*3} chain={len(vecs)} "
              f"E[conf] trained={e_tr:.2f}", flush=True)
    return packs


def main():
    bundle = {"paper_kv": build_paper_family(), "soft_k3": build_soft_family()}
    with open("packs/bench10_bundle.pkl", "wb") as fh:
        pickle.dump(bundle, fh, protocol=4)
    print("wrote bench10_bundle.pkl", os.path.getsize("packs/bench10_bundle.pkl") / 1e6, "MB")


if __name__ == "__main__":
    main()
