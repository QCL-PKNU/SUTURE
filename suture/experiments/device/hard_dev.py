"""hard_dev — hard-edge (objective-blind SUTURE_C) device support.

Adds what run_a2.py needs to drive a hard_ladder_bundle.pkl pack on real
hardware:
  * load_hard_pack  — bundle loader + derived fields (adj, seed_bits)
  * hard_segment_circuit / hard_segment_layout — the SAME transition
    synthesis the simulation channels were built from (driver_component on
    the active set + 2 ancillas, gcp_hard.build_hard_channels), lifted to
    the full n-qubit register with an X-init prefix.
  * decode / feasible_hard / canon_bits — VENDORED VERBATIM from
    experiments/run_hard_ladder.py so the device arm is bit-identical to the
    simulation arm (report 22). Objective-blind: reads only the measured
    string and the edge/assignment ROWS of C; the weight matrix W is never
    touched by any decoder, only by the harness for scoring.

MAX_CORRUPT is the c_max=6 budget of the manuscript's hard-edge SUTURE_C row.
"""
import os, sys, pickle
import numpy as np

# upstream Rasengan clone supplies driver_component (the transition
# synthesis the simulation channels were built from). Resolve it from the
# repo root regardless of where this module is copied to.
_HERE = os.path.dirname(os.path.abspath(__file__))
from pathlib import Path
_UPSTREAM = Path(__file__).resolve().parents[3] / "rasengan"
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))

MAX_CORRUPT = 6


def load_hard_pack(name, bundle_path="hard_ladder_bundle.pkl"):
    with open(bundle_path, "rb") as fh:
        bundle = pickle.load(fh)
    rec = next(r for r in bundle["instances"] if r["name"] == name)
    rec = dict(rec)
    v, edges = rec["v"], rec["edges"]
    rec["adj"] = [[w for (a, b) in edges for w in ((b,) if a == u else (a,) if b == u else ())]
                  for u in range(v)]
    rec["seed_bits"] = canon_bits(np.asarray(rec["seed_coloring"]), rec)
    rec["K"] = 3
    return rec


def hard_transition_vector(rec, u, cf, ct):
    """Null-space vector of the FULL hard-edge system (gcp_hard.HardPack)."""
    v, edges = rec["v"], rec["edges"]
    eidx = {(min(a, b), max(a, b)): i for i, (a, b) in enumerate(edges)}
    vec = np.zeros(rec["n"], dtype=int)
    vec[3 * u + cf] = -1
    vec[3 * u + ct] = +1
    for w in rec["adj"][u]:
        i = eidx[(min(u, w), max(u, w))]
        vec[3 * v + 3 * i + cf] = +1
        vec[3 * v + 3 * i + ct] = -1
    return vec


def hard_active_set(rec, t):
    u, cf, ct = rec["chain"][t]
    return np.flatnonzero(hard_transition_vector(rec, u, cf, ct))


def hard_segment_circuit(rec, t, init_bits):
    """Full-register circuit: X-init, then the driver on the active set + 2
    ancillas, then measure. Same synthesis as build_hard_channels."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from rasengan.solvers.qiskit.circuit.hdi_decompose import driver_component

    n = rec["n"]
    u, cf, ct = rec["chain"][t]
    A = hard_active_set(rec, t)
    m = len(A)
    vec = hard_transition_vector(rec, u, cf, ct)[A]
    hdi = [0 if x == -1 else 1 for x in vec]

    theta = Parameter("t")
    sub = QuantumCircuit(m + 2)
    driver_component(sub, list(range(m)), [m, m + 1], hdi, theta, "constant")
    sub = sub.assign_parameters([float(rec["params"][t])])

    qc = QuantumCircuit(n + 2)                       # 2 trailing ancillas
    for i in np.flatnonzero(np.asarray(init_bits)):
        qc.x(int(i))
    qc.barrier()
    qc.compose(sub, qubits=[int(a) for a in A] + [n, n + 1], inplace=True)
    qc.measure_all()
    return qc


def hard_segment_layout(rec, t, path):
    """Active-set qubits (+2 ancillas) onto CONTIGUOUS path slots; the rest
    fill the remaining slots in order. path must have >= n+2 entries."""
    A = [int(a) for a in hard_active_set(rec, t)]
    n = rec["n"]
    active_slots = list(range(len(A) + 2))
    layout = [None] * (n + 2)
    for k, q in enumerate(A):
        layout[q] = path[active_slots[k]]
    layout[n] = path[len(A)]
    layout[n + 1] = path[len(A) + 1]
    rest = iter(path[len(A) + 2:])
    for q in range(n + 2):
        if layout[q] is None:
            layout[q] = next(rest)
    return layout


def blocks_of(bits, v):
    return np.asarray(bits[:3 * v]).reshape(v, 3)

def edge_syndrome_score(bits, rec):
    """Violated edge rows raise per-vertex suspicion -- a read of C, not f."""
    v = rec["v"]
    score = np.zeros(v, dtype=np.int64)
    for i, (a, b) in enumerate(rec["edges"]):
        for c in range(3):
            if bits[3 * a + c] + bits[3 * b + c] + bits[3 * v + 3 * i + c] != 1:
                score[a] += 1
                score[b] += 1
    return score

def decode(bits, rec, rng, prune):
    """prune=True: list-coloring repair using edge ROWS of C (blind to f).
       prune=False: coset-uniform ablation -- same syndrome, no pruning."""
    v = rec["v"]
    dec = blocks_of(bits, v)
    wts = dec.sum(axis=1)
    col = np.full(v, -1, dtype=np.int64)
    corrupt = set(np.flatnonzero(wts != 1).tolist())
    for u in range(v):
        if u not in corrupt:
            col[u] = int(dec[u].argmax())
    score = edge_syndrome_score(bits, rec)
    for u in range(v):
        if u not in corrupt and score[u] >= 3:
            corrupt.add(u)
            col[u] = -1
    if len(corrupt) > MAX_CORRUPT:
        return None
    adj = rec["adj"]
    cor = sorted(corrupt)
    lists = []
    for u in cor:
        ones = np.flatnonzero(dec[u])
        cands = list(ones) if len(ones) >= 2 else [0, 1, 2]
        rng.shuffle(cands)
        lists.append(cands)

    if not prune:
        # Coset-uniform ablation: identical syndrome and identical candidate
        # lists, drawn uniformly with no use of the edge rows.
        for i, u in enumerate(cor):
            col[u] = lists[i][rng.integers(len(lists[i]))]
        return col

    # T6 list-coloring repair, implementing Definition 5's R(y) exactly:
    # the MINIMUM-Hamming feasible repairs, sampled UNIFORMLY.
    #
    # Two things are needed and both were once missing here.
    # (a) Uniform tie-break: returning the first valid completion from a
    #     backtracking search over shuffled lists is randomized but not
    #     uniform on multi-block corruptions (measured deviations to 0.17).
    # (b) Minimum distance: distance must be measured on the FULL string,
    #     slack bits included. Two edge-consistent completions can differ
    #     sharply in global Hamming cost because each colour choice forces
    #     different slack co-flips (measured pairs at distance 1 vs 5, 2 vs
    #     6, 3 vs 7); 18% of corruptions admit such a spread, and a uniform
    #     draw over all completions would land off the minimum set about
    #     half the time in those cases, putting the decoder outside
    #     Proposition 3.
    # So: enumerate every conflict-free completion, keep only those at
    # minimum global Hamming distance to y, and draw uniformly among them.
    # Bounded by prod|Lambda(u)| <= K^c_max = 729.
    sols = []

    def bt(i):
        if i == len(cor):
            sols.append(col.copy())
            return
        u = cor[i]
        for c in lists[i]:
            if all(col[w] != c for w in adj[u] if col[w] >= 0):
                col[u] = c
                bt(i + 1)
                col[u] = -1

    bt(0)
    if not sols:
        return None
    dists = [int((canon_bits(s, rec) != bits).sum()) for s in sols]
    dmin = min(dists)
    best = [s for s, d in zip(sols, dists) if d == dmin]
    return best[rng.integers(len(best))]

def canon_bits(col, rec):
    """Colouring -> full hard-edge bitstring (assignment + slack), the same
    encoding HardPack.coloring_to_bits produces: slack s_ec = 1 - x_uc - x_wc."""
    v, edges = rec["v"], rec["edges"]
    b = np.zeros(rec["n"], dtype=np.int8)
    for u in range(v):
        b[3 * u + int(col[u])] = 1
    for i, (a, w) in enumerate(edges):
        for c in range(3):
            b[3 * v + 3 * i + c] = 1 - b[3 * a + c] - b[3 * w + c]
    return b

def feasible_hard(bits, rec):
    """Full hard-edge feasibility, identical to HardPack.is_feasible_bits:
    one-hot assignment rows, proper colouring (the edge rows admit no 0/1
    slack when both endpoints share a colour), and slack consistency. Reads
    only C and b -- the objective is the weight matrix W, never touched."""
    v, edges = rec["v"], rec["edges"]
    dec = blocks_of(bits, v)
    if not (dec.sum(axis=1) == 1).all():
        return False
    col = dec.argmax(axis=1)
    if any(col[a] == col[b] for a, b in edges):
        return False
    for i, (a, w) in enumerate(edges):
        for c in range(3):
            if bits[3 * v + 3 * i + c] != 1 - bits[3 * a + c] - bits[3 * w + c]:
                return False
    return True


def hard_segment_template(rec, t):
    """The segment circuit WITHOUT the X-init prefix: driver on the active
    set + 2 ancillas, then measure. Transpiled once per round with a free
    layout (681 two-qubit gates vs 842 when the layout is pinned); the
    per-seed initial product state is then applied as X gates on the
    PHYSICAL qubits, which is exact because the init is a computational
    basis state and X is native."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from rasengan.solvers.qiskit.circuit.hdi_decompose import driver_component

    n = rec["n"]
    u, cf, ct = rec["chain"][t]
    A = hard_active_set(rec, t)
    m = len(A)
    vec = hard_transition_vector(rec, u, cf, ct)[A]
    hdi = [0 if x == -1 else 1 for x in vec]
    theta = Parameter("t")
    sub = QuantumCircuit(m + 2)
    driver_component(sub, list(range(m)), [m, m + 1], hdi, theta, "constant")
    sub = sub.assign_parameters([float(rec["params"][t])])
    qc = QuantumCircuit(n + 2)
    qc.compose(sub, qubits=[int(a) for a in A] + [n, n + 1], inplace=True)
    qc.measure_all()
    return qc


def apply_physical_init(isa_template, init_bits):
    """Prepend X on the physical qubits carrying the virtual qubits that
    start at 1. Uses the transpiled circuit's own initial index layout, so
    no assumption about routing is made."""
    lay = isa_template.layout.initial_index_layout(filter_ancillas=False)
    # copy_empty_like preserves the classical REGISTER NAMES ("meas"), which
    # the primitive result DataBin is keyed on — rebuilding with a bare
    # QuantumCircuit(n, m) silently renames it to "c" and the read fails.
    out = isa_template.copy_empty_like()
    for vq in np.flatnonzero(np.asarray(init_bits)):
        out.x(int(lay[int(vq)]))
    out.barrier()
    out.compose(isa_template, inplace=True)
    out._layout = isa_template.layout
    return out
