import numpy as np
import pytest

from sandwich._linalg import RankDeficientError, leverage, qr_solve, sandwich


def test_qr_solve_matches_normal_equations():
    rng = np.random.default_rng(0)
    x = np.column_stack([np.ones(200), rng.normal(size=(200, 3))])
    y = x @ np.array([1.0, 2.0, -0.5, 0.25]) + rng.normal(size=200)
    beta, xtx_inv = qr_solve(x, y)
    np.testing.assert_allclose(beta, np.linalg.solve(x.T @ x, x.T @ y), rtol=1e-10)
    np.testing.assert_allclose(xtx_inv, np.linalg.inv(x.T @ x), rtol=1e-8)


def test_rank_deficiency_is_detected():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(50, 2))
    x = np.column_stack([np.ones(50), z, z[:, 0] + z[:, 1]])
    with pytest.raises(RankDeficientError):
        qr_solve(x, rng.normal(size=50))


def test_more_regressors_than_rows_is_rejected():
    with pytest.raises(RankDeficientError):
        qr_solve(np.ones((3, 5)), np.ones(3))


def test_leverage_sums_to_k():
    rng = np.random.default_rng(2)
    x = np.column_stack([np.ones(100), rng.normal(size=(100, 4))])
    _, xtx_inv = qr_solve(x, rng.normal(size=100))
    h = leverage(x, xtx_inv)
    assert h.shape == (100,)
    assert np.all(h > 0) and np.all(h < 1)
    assert h.sum() == pytest.approx(5.0)


def test_sandwich_is_symmetric():
    b = np.array([[2.0, 0.1], [0.1, 1.0]])
    m = np.array([[1.0, 0.3], [0.3, 4.0]])
    v = sandwich(b, m, scale=2.0)
    np.testing.assert_allclose(v, v.T)
    np.testing.assert_allclose(v, 2.0 * b @ m @ b)
