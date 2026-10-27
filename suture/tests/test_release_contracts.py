"""Regression tests for the hybrid-output and bounded-repair contracts."""
import importlib.util
from pathlib import Path
import numpy as np
from repair_generic import repair


def _ladder():
    path = Path(__file__).resolve().parents[1] / 'experiments/run_hard_ladder.py'
    spec = importlib.util.spec_from_file_location('ladder_under_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_repair_counts_as_hybrid_success():
    mod = _ladder()
    rec = dict(v=1, n=3, edges=[], adj=[[]], W=np.array([[1., 2., 3.]]),
               chain=[(0, 0, 1)], seed_bits=np.array([1, 0, 0], dtype=np.int8))

    class CorruptChannel:
        def run_segment(self, t, init, shots, rng):
            return {np.array([0, 0, 0], dtype=np.int8).tobytes(): shots}

    # Deterministic repair makes the distinction independent of decoder tie breaks.
    mod.decode = lambda *args, **kwargs: np.array([0])
    out = mod.run_chain(rec, CorruptChannel(), 'synd', np.random.default_rng(0))
    assert out['survived'] and out['decoder_proper_seen']
    assert out['proper_seen']
    assert out['best_conflicts'] == 0.
    assert out['best_cost'] == 1.
    assert out['endpoint_schema'] == 'hybrid-v1'


def test_feasible_measurement_is_scored():
    mod = _ladder()
    rec = dict(v=1, n=3, edges=[], W=np.array([[1., 2., 3.]]),
               chain=[(0, 0, 1)], seed_bits=np.array([1, 0, 0], dtype=np.int8))

    class CleanChannel:
        def run_segment(self, t, init, shots, rng):
            return {init.tobytes(): shots}

    out = mod.run_chain(rec, CleanChannel(), 'purify', np.random.default_rng(0))
    assert out['proper_seen'] and out['best_cost'] == 1.


def test_combination_cap_can_bind_before_weight_limit():
    # A feasible five-flip correction exists, but C(40,5) exceeds MAX_WORK.
    C = np.array([[1] * 40 + [5]])
    assert repair(C, np.zeros(40, dtype=np.int8), np.random.default_rng(0)) is None
