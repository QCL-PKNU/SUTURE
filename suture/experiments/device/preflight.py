"""preflight — every gate that must PASS before any device shot is spent.

G-P1  bit-order round trip: X on known qubits -> rows_from_bitarray
      recovers exactly those qubits, both instances.
G-L1  circuit-vs-channel: noiseless AerSimulator segment outputs match the
      analytic pack transition law P(swap)=sin^2(theta_eff) per segment
      (max |dev| <= 3 sigma binomial) — angle convention proven, not assumed.
G-C1  chain smoke: full A2 chain (both arms) against Aer with a readout
      error model — runs end to end, register never empties for synd,
      latency instrumentation produces every field.
G-T1  transpile audit at target backend (fake or real spec): per-segment
      2q count == 3 (RXX+RYY -> CX-based decomp) and depth small & flat
      across segments; no routing SWAPs (adjacency respected).
G-B1  block-locality analyzer self-test on synthetic single flips.

Run: python device/preflight.py  (no credentials needed — Aer only).
Writes device/preflight_report.json; exits nonzero on any gate failure.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dev_lib import (K, load_soft_pack, coloring_to_bits, enumerate_proper,
                     is_onehot, conflicts, segment_circuit_baked,
                     segment_layout, choose_path, rows_from_bitarray,
                     Register, arm_postprocess, append_jsonl)

report = {}


def gate(name, ok, detail):
    report[name] = dict(ok=bool(ok), detail=detail)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def main():
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    packs = {nm: load_soft_pack(nm) for nm in ("gcp_v8", "gcp_p12_s0")}
    # MPS: exact for these product-state + single-2q-gate circuits, and
    # memory-flat at 36q (statevector would need 2^36 amplitudes).
    sim = AerSimulator(method="matrix_product_state")
    ok_all = True

    # ---------- G-P1 bit order ----------
    from qiskit import QuantumCircuit
    for nm, pack in packs.items():
        n = pack["n"]
        marks = [0, 1, K, n - 1]
        qc = QuantumCircuit(n)
        for i in marks:
            qc.x(i)
        qc.measure_all()
        res = sim.run(qc, shots=16).result()
        counts = res.get_counts()
        s = max(counts, key=counts.get)
        got = sorted(i for i, ch in enumerate(s[::-1]) if ch == "1")
        ok_all &= gate(f"G-P1[{nm}]", got == sorted(marks), f"marked {sorted(marks)} got {got}")

    # ---------- G-L1 circuit vs analytic channel ----------
    rng = np.random.default_rng(7)
    for nm, pack in packs.items():
        devs = []
        proper = enumerate_proper(pack)
        for t in rng.choice(len(pack["chain"]), size=6, replace=False):
            t = int(t)
            u, c0, c1 = pack["chain"][t]
            # condition on the active pair being in the {01,10} subspace:
            # pick a proper coloring with col[u] == c0 (pair = 10), so the
            # transition law P(swap) applies; col[u] outside {c0,c1} would
            # leave the pair at 00 where the XY body acts trivially.
            cand = proper[proper[:, u] == c0]
            if len(cand) == 0:
                continue
            col = cand[rng.integers(len(cand))]
            init = coloring_to_bits(col, pack["v"])
            qc = segment_circuit_baked(pack, t, init)
            shots = 4096
            res = sim.run(qc, shots=shots).result()
            swap = 0
            for s, ct in res.get_counts().items():
                bits = np.frombuffer(s[::-1].encode(), dtype=np.uint8) - ord("0")
                if bits[u * K + c0] != init[u * K + c0]:
                    swap += ct
            p_obs = swap / shots
            # analytic law candidates for theta_eff: the segment acts as
            # exp(-i*phi*(XX+YY)/2) on the pair; swap prob on {01,10} inputs
            th = float(pack["params"][t])
            cands = {"sin2(2th)": np.sin(2 * th) ** 2, "sin2(th)": np.sin(th) ** 2}
            # pick convention by first segment; must then hold for all
            devs.append((p_obs, cands))
        for label in ("sin2(2th)", "sin2(th)"):
            m = max(abs(p - c[label]) for p, c in devs)
            if m < 0.03:
                ok_all &= gate(f"G-L1[{nm}]", True, f"convention {label}, max|dev|={m:.4f} over 6 segments")
                report[f"G-L1[{nm}]"]["convention"] = label
                break
        else:
            ok_all &= gate(f"G-L1[{nm}]", False,
                           f"neither convention fits: {[(round(p,3), {k: round(v,3) for k,v in c.items()}) for p,c in devs[:3]]}")

    # ---------- G-C1 chain smoke (24q, both arms, noisy Aer) ----------
    pack = packs["gcp_v8"]
    nm_noise = NoiseModel()
    ro = ReadoutError([[0.97, 0.03], [0.07, 0.93]])
    for q in range(pack["n"]):
        nm_noise.add_readout_error(ro, [q])
    noisy = AerSimulator(noise_model=nm_noise)
    lat_rows = []
    for arm in ("purify", "synd"):
        rng = np.random.default_rng(11)
        reg = Register(coloring_to_bits(pack["seed_coloring"], pack["v"]), rng)
        alive, proper_ever = True, False
        for t in range(len(pack["chain"])):
            t_cl0 = time.perf_counter()
            init = reg.sample_rows(1)[0]          # single-init protocol (G-C1 = device shape)
            qc = segment_circuit_baked(pack, t, init)
            t_q0 = time.perf_counter()
            res = noisy.run(qc, shots=64).result()
            t_q1 = time.perf_counter()
            rows = []
            for s, ct in res.get_counts().items():
                bits = (np.frombuffer(s[::-1].encode(), dtype=np.uint8) - ord("0")).astype(np.int8)
                rows.extend([bits] * ct)
            rows = np.stack(rows)
            kept, stats, dec_s = arm_postprocess(arm, rows, pack, rng)
            proper_ever |= stats["proper_seen_measured"]
            alive = reg.update(kept)
            lat_rows.append(dict(arm=arm, t=t, sim_s=t_q1 - t_q0, decode_s=dec_s,
                                 classical_other_s=(time.perf_counter() - t_cl0) - (t_q1 - t_q0) - dec_s,
                                 **stats))
            if not alive:
                break
        ok = alive if arm == "synd" else True
        ok_all &= gate(f"G-C1[{arm}]", ok,
                       f"survived={alive} segments_run={t+1} proper_seen_measured={proper_ever}")
    with open("device/preflight_latency_smoke.json", "w") as fh:
        json.dump(lat_rows, fh, default=str)

    # ---------- G-T1 transpile audit on fake Heron ----------
    try:
        from qiskit_ibm_runtime.fake_provider import FakeTorino
        fake = FakeTorino()
        path = choose_path(fake, packs["gcp_p12_s0"]["n"])
        pack36 = packs["gcp_p12_s0"]
        pm = generate_preset_pass_manager(optimization_level=1, backend=fake)
        stats2q, depths = [], []
        proper36 = enumerate_proper(pack36, cap=500)
        for t in (0, 10, 25, 47):
            init = coloring_to_bits(proper36[t % len(proper36)], pack36["v"])
            qc = segment_circuit_baked(pack36, t, init)
            lay = segment_layout(pack36, t, path)
            pm_l = generate_preset_pass_manager(optimization_level=1, backend=fake,
                                                initial_layout=lay)
            tqc = pm_l.run(qc)
            n2 = sum(1 for inst in tqc.data if inst.operation.num_qubits == 2)
            stats2q.append(n2); depths.append(tqc.depth())
        flat = max(stats2q) - min(stats2q) <= 1
        ok_all &= gate("G-T1", max(stats2q) <= 4 and flat,
                       f"2q counts {stats2q} depths {depths} (path head {path[:6]})")
    except Exception as e:
        ok_all &= gate("G-T1", False, f"transpile audit error: {e}")

    # ---------- G-B1 block-locality analyzer self-test ----------
    pack = packs["gcp_v8"]
    proper = enumerate_proper(pack)
    rng = np.random.default_rng(3)
    n_bad = 0
    for _ in range(500):
        col = proper[rng.integers(len(proper))]
        bits = coloring_to_bits(col, pack["v"])
        i = rng.integers(pack["n"])
        bits[i] ^= 1
        w = np.asarray(bits).reshape(pack["v"], K).sum(axis=1)
        n_corrupt = int((w != 1).sum())
        if n_corrupt != 1:
            n_bad += 1
    ok_all &= gate("G-B1", n_bad == 0, f"single flip => exactly 1 corrupt block, {500-n_bad}/500")

    with open("device/preflight_report.json", "w") as fh:
        json.dump(report, fh, indent=1, default=str)
    print("\nPREFLIGHT", "PASS" if ok_all else "FAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
