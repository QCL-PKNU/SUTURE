"""Fast unit tests for the suture core (run: pytest tests/ from suture/).

Covers: one-hot codec + feasibility, T6 syndrome decoder on planted single flips,
domain-wall codec + unitarity, hard-edge encoding + slack consistency, and
channel-matrix stochasticity. All tests are seed-fixed and run in seconds
(the two qiskit tests are the slowest).
"""
import numpy as np
import pytest

import _paths  # noqa: F401

EDGES_V8 = [(i, (i + 1) % 8) for i in range(8)] + [(0, 2), (4, 6)]


def _mk_pack():
    from gcp_k3 import GcpK3Pack
    chain = [(u, 0, c) for _ in range(2) for u in range(8) for c in (1, 2)]
    return GcpK3Pack(name="t", v=8, K=3, edges=EDGES_V8, chain=chain,
                     params=np.full(len(chain), np.pi / 4),
                     seed_coloring=np.zeros(8, dtype=np.int64))


def _proper_colorings(v, edges, n=50, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        col = rng.integers(0, 3, size=v)
        if all(col[a] != col[b] for a, b in edges):
            out.append(col)
    return out


class TestOneHotCodec:
    def test_roundtrip_and_feasibility(self):
        pack = _mk_pack()
        for col in _proper_colorings(8, EDGES_V8, n=20):
            bits = pack.coloring_to_bits(col)
            assert pack.is_feasible_bits(bits)
            assert np.array_equal(pack.decode_coloring(bits), col)

    def test_single_flip_always_detected(self):
        pack = _mk_pack()
        col = _proper_colorings(8, EDGES_V8, n=1)[0]
        bits = pack.coloring_to_bits(col)
        for i in range(pack.n):
            b2 = bits.copy()
            b2[i] = 1 - b2[i]
            assert not pack.is_feasible_bits(b2), f"flip at {i} undetected"


class TestSyndromeDecoder:
    def test_planted_single_flip_recovery(self):
        """T6 predicts 0.955 exact recovery for SYND on gcp_v8; assert > 0.88."""
        from gcp_gonogo_lib import decode_synd
        pack = _mk_pack()
        rng = np.random.default_rng(3)
        n_ok = 0
        trials = 400
        for _ in range(trials):
            col = _proper_colorings(8, EDGES_V8, n=1, seed=rng.integers(1 << 30))[0]
            bits = pack.coloring_to_bits(col)
            i = rng.integers(pack.n)
            bits[i] = 1 - bits[i]
            out = decode_synd(pack, bits, rng)
            n_ok += int(out is not None and np.array_equal(out, col))
        assert n_ok / trials > 0.88

    def test_beats_nearest(self):
        from gcp_gonogo_lib import decode_synd, decode_nearest
        pack = _mk_pack()
        rng = np.random.default_rng(4)
        wins = {"synd": 0, "near": 0}
        for _ in range(300):
            col = _proper_colorings(8, EDGES_V8, n=1, seed=rng.integers(1 << 30))[0]
            bits = pack.coloring_to_bits(col)
            for i in rng.choice(pack.n, size=2, replace=False):
                bits[i] = 1 - bits[i]
            for name, dec in (("synd", decode_synd), ("near", decode_nearest)):
                out = dec(pack, bits.copy(), rng)
                wins[name] += int(out is not None and np.array_equal(out, col))
        assert wins["synd"] > wins["near"]


class TestDomainWall:
    def test_codec(self):
        from gcp_dw import CODE, DECODE
        for c, b in CODE.items():
            assert DECODE[b] == c
        assert DECODE[(0, 1)] == -1

    def test_transition_unitary(self):
        from gcp_dw import dw_transition_unitary
        for kind in ("01", "02"):
            U = dw_transition_unitary(0.7, kind)
            assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-12)

    def test_silent_flip_fraction(self):
        """4/6 single flips on valid dw codewords stay valid (silent)."""
        from gcp_dw import CODE, DECODE
        silent = total = 0
        for c, (b1, b2) in CODE.items():
            for pos in range(2):
                nb = (1 - b1, b2) if pos == 0 else (b1, 1 - b2)
                total += 1
                silent += int(DECODE[nb] >= 0)
        assert (silent, total) == (4, 6)


class TestHardEdge:
    def test_encoding_and_slack(self):
        from gcp_hard import HardPack
        pack = HardPack("t", 5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
        for col in pack.proper[:10]:
            bits = pack.coloring_to_bits(col)
            assert pack.is_feasible_bits(bits)
            b2 = bits.copy()
            b2[0] = 1 - b2[0]
            assert not pack.is_feasible_bits(b2)

    def test_transition_nullspace(self):
        """Transition vectors preserve every constraint row (Cx=b invariance)."""
        from gcp_hard import HardPack
        pack = HardPack("t", 5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
        # feasibility of (bits + vec) whenever the move is legal
        col = pack.proper[0]
        bits = pack.coloring_to_bits(col).astype(int)
        for (u, cf, ct) in pack.chain[:10]:
            vec = pack.transition_vector(u, cf, ct)
            if col[u] == cf and all(col[w] != ct for w in pack.adj[u]):
                moved = bits + vec
                assert set(np.unique(moved)) <= {0, 1}
                assert pack.is_feasible_bits(moved.astype(np.int8))


@pytest.mark.slow
class TestChannels:
    def test_onehot_pair_channel_stochastic(self):
        from gcp_k3 import pair_channel_matrix
        T = pair_channel_matrix(np.pi / 4, 2.0, seed=1)
        assert T.shape == (4, 4)
        assert np.allclose(T.sum(axis=1), 1.0, atol=1e-9)
        assert (T >= -1e-12).all()

    def test_dw_pair_channel_stochastic(self):
        from gcp_dw import dw_pair_channel_matrix
        T = dw_pair_channel_matrix(np.pi / 4, "01", 2.0)
        assert np.allclose(T.sum(axis=1), 1.0, atol=1e-9)
        # noise-free limit: swap prob = sin^2(theta) between 00 and 10
        T0 = dw_pair_channel_matrix(np.pi / 4, "01", 0.0)
        assert abs(T0[0, 2] - np.sin(np.pi / 4) ** 2) < 1e-6
