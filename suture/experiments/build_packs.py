"""Build trained InstancePacks for the H1-R pilots.

Instances (kept exact-enumerable, n <= 16):
  flp_1x2   FLP 1 demand x 2 facilities        (6 qubits after slack)
  flp_2x2   FLP 2 demands x 2 facilities       (10 qubits after slack)
  flp_2x3   FLP 2 demands x 3 facilities       (15 qubits after slack)
  gcp_p3    GCP on path P3 (edges (0,1),(1,2)) (one-hot, K = V = 3; 15 qubits w/ slack)
  scp_4     Set cover, 4 sets / 3 elements     (>= covering constraints, slack-heavy)

For each: build LcboModel, simplify basis (upstream greedy), build pruned chain
(classical explorer equivalent), synthesize per-transition circuits
(driver_component, mcx constant), transpile templates once, train times with
COBYLA on the noise-free segmented executor (seg_size=1, purify off since
noise-free output is always feasible), save pack via dill.
"""
import sys, os, io, contextlib, time
import _paths  # noqa: F401  (repo-relative sys.path setup)

import numpy as np
import dill
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from rasengan.problems.facility_location_problem import FacilityLocationProblem
from rasengan.problems.graph_coloring_problem import GraphColoringProblem
from rasengan.problems.set_cover_problem import SetCoverProblem
from rasengan.utils.linear_system import greedy_simplification_of_transition_Hamiltonian
from rasengan.solvers.qiskit.circuit.hdi_decompose import driver_component

from h1_pilot_lib import (InstancePack, NoisyRunner, enumerate_feasible, bits_to_str,
                          build_chain, run_segmented, CORE_BASIS)

RNG = np.random.default_rng(20260723)
SIM = AerSimulator()


def synth_template(n: int, u: np.ndarray) -> QuantumCircuit:
    """One-transition parametrized circuit on n+2 qubits, transpiled to core basis."""
    qc = QuantumCircuit(n + 2, n)
    nonzero = np.nonzero(u)[0].tolist()
    hdi_bitstr = [0 if x == -1 else 1 for x in u if x != 0]
    theta = Parameter("t")
    driver_component(qc, nonzero, [n, n + 1], hdi_bitstr, theta, "constant")
    return transpile(qc, SIM, basis_gates=CORE_BASIS, optimization_level=2)


def build_pack(name: str, prb, obj_func, max_rounds=None) -> InstancePack:
    t0 = time.time()
    constr = prb.lin_constr_mtx.astype(int)
    n = len(prb.variables)
    with contextlib.redirect_stdout(io.StringIO()):
        seed = np.asarray(prb.get_feasible_solution(), dtype=int)
    basis = greedy_simplification_of_transition_Hamiltonian(prb.driver_bitstr).astype(int)

    F = enumerate_feasible(constr, n)
    fset = {tuple(r.tolist()) for r in F}
    assert tuple(seed.tolist()) in fset, "seed must be feasible"
    chain = build_chain(basis, seed, fset, max_rounds=max_rounds)

    F_strings = [bits_to_str(r) for r in F]
    F_energies = np.array([obj_func([int(b) for b in r]) for r in F], dtype=float)
    i_star = int(np.argmin(F_energies))

    templates = [synth_template(n, basis[i]) for i in chain]

    pack = InstancePack(
        name=name, feasible_seed=seed, chain_templates=templates, chain_u=chain,
        params=np.full(len(chain), np.pi / 4), F=F, F_strings=F_strings,
        F_energies=F_energies, x_star=F_strings[i_star], e_star=float(F_energies[i_star]),
        e_worst=float(F_energies.max()), n=n, constr=constr,
    )

    # ---- train transition times, noise-free, seg_size=1 (upstream default) ----
    clean = NoisyRunner(None, seed=11)
    from h1_pilot_lib import segment_bodies

    def expected_energy(params):
        pack.params = np.asarray(params)
        bodies = segment_bodies(pack, 1)
        r = run_segmented(pack, clean, shots=256, mode="purify",
                          rng=np.random.default_rng(5), bodies=bodies, keep_raw=False)
        reg = r.final_register
        tot = sum(reg.values())
        if tot == 0:
            return pack.e_worst + 1.0
        return sum(pack.energy(k) * v for k, v in reg.items()) / tot

    x0 = np.full(len(chain), np.pi / 4)
    if len(chain) >= 2:
        try:
            res = minimize(expected_energy, x0, method="COBYLA",
                           options={"maxiter": 150, "rhobeg": 0.4})
            pack.params = res.x if expected_energy(res.x) <= expected_energy(x0) else x0
        except ValueError:
            pack.params = x0
    else:
        # 1-parameter chain: coarse 1-D scan
        grid = np.linspace(0.1, np.pi / 2, 12)
        vals = [expected_energy(np.array([g])) for g in grid]
        pack.params = np.array([grid[int(np.argmin(vals))]])

    e_final = expected_energy(pack.params)
    print(f"[{name}] n={n} |F|={len(F)} m={len(basis)} chain={len(chain)} "
          f"E*={pack.e_star:.1f} E_trained={e_final:.2f} "
          f"depth1={templates[0].depth()} ({time.time()-t0:.0f}s)")
    return pack


def main():
    os.makedirs("packs", exist_ok=True)
    packs = {}
    import random
    random.seed(3)

    # FLP 1x2 (6 qubits)
    p = FacilityLocationProblem(1, 2, [[2, 10]], [2, 1])
    packs["flp_1x2"] = build_pack("flp_1x2", p, p.obj_func)

    # FLP 2x2 (10 qubits)
    p = FacilityLocationProblem(2, 2, [[10, 2], [1, 20]], [1, 1])
    packs["flp_2x2"] = build_pack("flp_2x2", p, p.obj_func)

    # FLP 2x3 (15 qubits)
    p = FacilityLocationProblem(2, 3, [[10, 2, 3], [1, 20, 3]], [1, 1, 2])
    packs["flp_2x3"] = build_pack("flp_2x3", p, p.obj_func)

    # GCP on path P3 (3 vertices, 2 edges) one-hot K=3 -> 9 qubits + slack
    p = GraphColoringProblem(3, [(0, 1), (1, 2)], [1, 2, 3])
    packs["gcp_p3"] = build_pack("gcp_p3", p, p.obj_func)

    # SCP small (element-cover, >= constraints add slack bits).
    # NOTE: the symmetric 3-set config [[0,1],[1,2],[0,2]] produces a driver
    # basis violating C u = 0 (upstream find_basic_solution numeric edge case);
    # this 4-set config yields a valid basis. Documented in the report.
    p = SetCoverProblem(4, 3, [[0], [1], [2], [0, 1, 2]])
    packs["scp_4"] = build_pack("scp_4", p, p.obj_func)

    with open("packs/instance_packs.dill", "wb") as fh:
        dill.dump(packs, fh)
    print("saved packs/instance_packs.dill")


if __name__ == "__main__":
    main()
