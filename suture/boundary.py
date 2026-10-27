"""Objective-blind repair and resampling at a measured segment boundary."""
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .lib.repair_generic import MAX_CORRUPT, repair


@dataclass(frozen=True)
class BoundaryResult:
    """Accepted pool, next register, and per-input-shot accounting.

    next_register is None when no shot survives. Otherwise its rows initialize
    the next segment. accepted contains each retained/repaired occurrence once,
    before resampling. Counts refer to input shots, not resampled occurrences.
    """

    accepted: np.ndarray
    next_register: np.ndarray | None
    n_measured: int
    n_feasible_measured: int
    n_repaired: int
    n_rejected: int

    @property
    def can_continue(self) -> bool:
        return self.next_register is not None


def repair_boundary(measurements, C, b, *, rng, shots=None, max_flips=MAX_CORRUPT):
    """Repair binary measurements satisfying integer constraints C @ x == b.

    measurements: dense integer/bool array (number of shots, number of bits).
    C: dense integer constraint matrix (number of rows, number of bits).
    b: integer right-hand-side vector (number of rows,).
    rng: numpy.random.Generator, used for repair ties and resampling.
    shots: next-segment register size; defaults to the input shot count.
    max_flips: maximum Hamming weight searched, default six. The generic decoder
        also enforces its fixed 200,000-combinations-per-weight work cap.

    Feasible shots pass unchanged. Infeasible shots are repaired or rejected.
    A nonempty accepted pool is resampled uniformly by occurrence to fill the
    requested register. Empty pools return next_register=None, leaving the
    caller to stop or apply its own fallback. Input arrays are never modified.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError('rng must be a numpy.random.Generator')
    if not isinstance(max_flips, Integral) or isinstance(max_flips, bool) or max_flips < 0:
        raise ValueError('max_flips must be a nonnegative integer')
    Y, A, rhs = np.asarray(measurements), np.asarray(C), np.asarray(b)
    if Y.ndim != 2 or A.ndim != 2 or rhs.ndim != 1:
        raise ValueError('measurements and C must be 2-D; b must be 1-D')
    if Y.shape[1] != A.shape[1] or rhs.shape[0] != A.shape[0] or A.shape[1] == 0:
        raise ValueError('C, b and measurements have incompatible shapes')
    for name, value in [('measurements', Y), ('C', A), ('b', rhs)]:
        if value.dtype.kind not in 'biu':
            raise TypeError(f'{name} must contain integers or booleans')
        if value.dtype.kind == 'u' and value.size and value.max() > np.iinfo(np.int64).max:
            raise ValueError(f'{name} exceeds the supported signed 64-bit range')
    arithmetic_bound = np.abs(A.astype(object)).sum(axis=1) + np.abs(rhs.astype(object))
    if np.any(arithmetic_bound > np.iinfo(np.int64).max):
        raise ValueError('constraint coefficients exceed safe signed 64-bit arithmetic')
    if np.any((Y != 0) & (Y != 1)):
        raise ValueError('measurements must contain only zero and one')
    target = len(Y) if shots is None else shots
    if not isinstance(target, Integral) or isinstance(target, bool) or target < 0:
        raise ValueError('shots must be a nonnegative integer')
    if target == 0 and len(Y):
        raise ValueError('shots must be positive for a nonempty measurement batch')
    # The decoder's arithmetic uses int64; use small integer constraint weights.
    system = np.column_stack([A, rhs]).astype(np.int64)
    feasible = (Y.astype(np.int64) @ system[:, :-1].T == system[:, -1]).all(axis=1)
    accepted = []
    n_repaired = 0
    for row, is_feasible in zip(Y, feasible):
        if is_feasible:
            accepted.append(row.astype(np.int8, copy=True))
        else:
            result = repair(system, row, rng, max_corrupt=max_flips)
            if result is not None:
                accepted.append(result)
                n_repaired += 1
    pool = np.stack(accepted) if accepted else np.empty((0, Y.shape[1]), dtype=np.int8)
    register = None
    if len(pool):
        register = pool.copy() if len(pool) == target else pool[rng.integers(len(pool), size=target)].copy()
    return BoundaryResult(pool, register, len(Y), int(feasible.sum()), n_repaired,
                          len(Y)-len(pool))
