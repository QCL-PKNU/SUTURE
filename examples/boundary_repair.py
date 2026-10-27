"""Run from the repository root: python -m examples.boundary_repair"""
import numpy as np
from suture import repair_boundary


def main():
    # Two one-hot blocks of three bits. Each block must select one entry.
    C = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]])
    b = np.array([1, 1])
    measured = np.array([
        [1, 0, 0, 0, 1, 0],  # feasible
        [1, 1, 0, 0, 1, 0],  # two entries selected in the first block
        [0, 0, 0, 1, 0, 0],  # first block empty
        [0, 1, 0, 0, 1, 1],  # two entries selected in the second block
    ], dtype=np.int8)
    result = repair_boundary(measured, C, b, rng=np.random.default_rng(7))
    print(f'Measured: {result.n_measured}; already feasible: {result.n_feasible_measured}; '
          f'repaired: {result.n_repaired}; rejected: {result.n_rejected}')
    if result.can_continue:
        assert np.all(result.next_register @ C.T == b)
        print('Next-segment register:')
        print(result.next_register)
    else:
        print('No feasible initialization state: stop or invoke your chosen fallback.')


if __name__ == '__main__':
    main()
