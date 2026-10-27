"""Boundary API tests cover integration, accounting and actionable failures."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from suture import repair_boundary


def test_batch_repair_preserves_inputs_and_produces_feasible_register():
    C, b = np.array([[1, 1, 1]]), np.array([1])
    Y = np.array([[1, 0, 0], [0, 0, 0], [1, 1, 0]], dtype=np.int8)
    original = Y.copy()
    result = repair_boundary(Y, C, b, rng=np.random.default_rng(4), shots=9)
    np.testing.assert_array_equal(Y, original)
    assert (result.n_feasible_measured, result.n_repaired, result.n_rejected) == (1, 2, 0)
    assert result.next_register.shape == (9, 3)
    assert np.all(result.next_register @ C.T == b)
    np.testing.assert_array_equal(result.accepted[0], original[0])


def test_partial_survival_resamples_only_accepted_occurrences():
    result = repair_boundary(np.array([[1, 0], [0, 0]]), np.array([[1, 1]]),
                             np.array([1]), rng=np.random.default_rng(0), max_flips=0)
    assert (result.n_feasible_measured, result.n_repaired, result.n_rejected) == (1, 0, 1)
    np.testing.assert_array_equal(result.next_register, [[1, 0], [1, 0]])


def test_empty_pool_returns_control_to_caller():
    result = repair_boundary(np.array([[0, 0]]), np.array([[1, 1]]),
                             np.array([1]), rng=np.random.default_rng(0), max_flips=0)
    assert not result.can_continue and result.next_register is None
    assert result.accepted.shape == (0, 2) and result.n_rejected == 1


def test_empty_measurement_batch():
    result = repair_boundary(np.empty((0, 2), dtype=int), np.array([[1, 1]]),
                             np.array([1]), rng=np.random.default_rng(0))
    assert result.n_measured == 0 and not result.can_continue


def test_reproducible_with_explicit_generator():
    args = np.zeros((5, 3), dtype=int), np.array([[1, 1, 1]]), np.array([1])
    a = repair_boundary(*args, rng=np.random.default_rng(3), shots=12)
    b = repair_boundary(*args, rng=np.random.default_rng(3), shots=12)
    np.testing.assert_array_equal(a.next_register, b.next_register)


@pytest.mark.parametrize('measurements,C,b,exception', [
    ([[0, 2]], [[1, 1]], [1], ValueError),
    ([[0., 1.]], [[1, 1]], [1], TypeError),
    ([[0, 1]], [[1]], [1], ValueError),
    ([[0, 1]], [[1, 1]], [[1]], ValueError),
])
def test_invalid_inputs_fail_before_decoding(measurements, C, b, exception):
    with pytest.raises(exception):
        repair_boundary(measurements, C, b, rng=np.random.default_rng(0))
