"""Every covariance estimator is checked against statsmodels on the same X and residuals."""

import numpy as np
import pytest
import statsmodels.api as sm

from sandwich._linalg import qr_solve
from sandwich.covariance import (
    bartlett_weights,
    compute_covariance,
    cov_cluster,
    cov_hac,
    cov_hc,
    cov_nonrobust,
)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(42)
    n = 400
    x = np.column_stack([np.ones(n), rng.normal(size=(n, 3))])
    g1 = rng.integers(0, 20, size=n)  # 20 clusters
    g2 = rng.integers(0, 8, size=n)
    # heteroskedastic + cluster-correlated + serially-correlated errors
    e = 0.5 * rng.normal(size=n) * (1 + np.abs(x[:, 1])) + rng.normal(size=20)[g1]
    e = e + np.r_[0.0, 0.6 * e[:-1]]
    y = x @ np.array([1.0, 2.0, -1.0, 0.5]) + e
    beta, xtx_inv = qr_solve(x, y)
    resid = y - x @ beta
    return x, y, resid, xtx_inv, g1, g2


def _sm_bse(x, y, **fit_kwargs):
    return sm.OLS(y, x).fit(**fit_kwargs).bse


def test_nonrobust(data):
    x, y, resid, xtx_inv, *_ = data
    res = cov_nonrobust(x, resid, xtx_inv, df_resid=x.shape[0] - x.shape[1])
    np.testing.assert_allclose(np.sqrt(np.diag(res.cov)), _sm_bse(x, y), rtol=1e-10)
    assert res.df_inference == 396


@pytest.mark.parametrize("kind", ["HC0", "HC1", "HC2", "HC3"])
def test_hc(data, kind):
    x, y, resid, xtx_inv, *_ = data
    res = cov_hc(x, resid, xtx_inv, df_resid=x.shape[0] - x.shape[1], kind=kind)
    np.testing.assert_allclose(np.sqrt(np.diag(res.cov)), _sm_bse(x, y, cov_type=kind), rtol=1e-9)
    assert res.cov_type == kind


def test_cluster_one_way(data):
    x, y, resid, xtx_inv, g1, _ = data
    res = cov_cluster(x, resid, xtx_inv, g1, df_resid=x.shape[0] - x.shape[1])
    expected = _sm_bse(x, y, cov_type="cluster", cov_kwds={"groups": g1})
    np.testing.assert_allclose(np.sqrt(np.diag(res.cov)), expected, rtol=1e-9)
    assert res.df_inference == 19
    assert res.info["n_clusters"] == 20


def test_cluster_without_correction(data):
    x, y, resid, xtx_inv, g1, _ = data
    res = cov_cluster(x, resid, xtx_inv, g1, df_resid=396, use_correction=False)
    expected = _sm_bse(x, y, cov_type="cluster", cov_kwds={"groups": g1, "use_correction": False})
    np.testing.assert_allclose(np.sqrt(np.diag(res.cov)), expected, rtol=1e-9)


def test_cluster_two_way(data):
    x, y, resid, xtx_inv, g1, g2 = data
    res = cov_cluster(x, resid, xtx_inv, (g1, g2), df_resid=396)
    expected = _sm_bse(x, y, cov_type="cluster", cov_kwds={"groups": np.column_stack([g1, g2])})
    np.testing.assert_allclose(np.sqrt(np.diag(res.cov)), expected, rtol=1e-9)
    assert res.info["n_clusters"] == (20, 8)
    assert res.df_inference == 7


def test_cluster_labels_can_be_strings(data):
    x, y, resid, xtx_inv, g1, _ = data
    labels = np.array([f"firm-{g}" for g in g1])
    a = cov_cluster(x, resid, xtx_inv, labels, df_resid=396).cov
    b = cov_cluster(x, resid, xtx_inv, g1, df_resid=396).cov
    np.testing.assert_allclose(a, b)


@pytest.mark.parametrize("maxlags", [0, 1, 4])
def test_hac(data, maxlags):
    x, y, resid, xtx_inv, *_ = data
    res = cov_hac(x, resid, xtx_inv, df_resid=396, maxlags=maxlags)
    expected = _sm_bse(x, y, cov_type="HAC", cov_kwds={"maxlags": maxlags})
    np.testing.assert_allclose(np.sqrt(np.diag(res.cov)), expected, rtol=1e-9)


def test_hac_with_correction(data):
    x, y, resid, xtx_inv, *_ = data
    res = cov_hac(x, resid, xtx_inv, df_resid=396, maxlags=2, use_correction=True)
    expected = _sm_bse(x, y, cov_type="HAC", cov_kwds={"maxlags": 2, "use_correction": True})
    np.testing.assert_allclose(np.sqrt(np.diag(res.cov)), expected, rtol=1e-9)


def test_hac_zero_lags_equals_hc0(data):
    x, y, resid, xtx_inv, *_ = data
    hac = cov_hac(x, resid, xtx_inv, df_resid=396, maxlags=0).cov
    hc0 = cov_hc(x, resid, xtx_inv, df_resid=396, kind="HC0").cov
    np.testing.assert_allclose(hac, hc0)


def test_bartlett_weights():
    np.testing.assert_allclose(bartlett_weights(3), [0.75, 0.5, 0.25])
    assert bartlett_weights(0).size == 0


def test_dispatch_and_errors(data):
    x, y, resid, xtx_inv, g1, _ = data
    assert compute_covariance("hc2", x, resid, xtx_inv, 396).cov_type == "HC2"
    assert compute_covariance("cluster", x, resid, xtx_inv, 396, groups=g1).cov_type == "cluster"
    assert compute_covariance("HAC", x, resid, xtx_inv, 396, maxlags=1).cov_type == "HAC"
    with pytest.raises(ValueError, match="requires groups"):
        compute_covariance("cluster", x, resid, xtx_inv, 396)
    with pytest.raises(ValueError, match="requires maxlags"):
        compute_covariance("HAC", x, resid, xtx_inv, 396)
    with pytest.raises(ValueError, match="unknown cov_type"):
        compute_covariance("bootstrap", x, resid, xtx_inv, 396)
