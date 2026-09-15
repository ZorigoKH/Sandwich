import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from sandwich.ols import OLS, add_constant


@pytest.fixture(scope="module")
def wage_like():
    rng = np.random.default_rng(7)
    n = 500
    educ = rng.integers(8, 20, size=n).astype(float)
    exper = rng.uniform(0, 30, size=n)
    female = rng.integers(0, 2, size=n).astype(float)
    firm = rng.integers(0, 25, size=n)
    u = rng.normal(size=n) * (0.5 + 0.05 * exper) + rng.normal(size=25)[firm] * 0.3
    lwage = 0.5 + 0.08 * educ + 0.02 * exper - 0.15 * female + u
    df = pd.DataFrame({"lwage": lwage, "educ": educ, "exper": exper, "female": female})
    return df, firm


def test_params_and_nonrobust_inference_match_statsmodels(wage_like):
    df, _ = wage_like
    x = add_constant(df[["educ", "exper", "female"]])
    res = OLS(df["lwage"], x).fit()
    ref = sm.OLS(df["lwage"], x).fit()
    np.testing.assert_allclose(res.params, ref.params.to_numpy(), rtol=1e-10)
    np.testing.assert_allclose(res.bse, ref.bse.to_numpy(), rtol=1e-10)
    np.testing.assert_allclose(res.tvalues, ref.tvalues.to_numpy(), rtol=1e-10)
    np.testing.assert_allclose(res.pvalues, ref.pvalues.to_numpy(), rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(res.conf_int(), ref.conf_int().to_numpy(), rtol=1e-10)
    assert res.r2 == pytest.approx(ref.rsquared)
    assert res.r2_adj == pytest.approx(ref.rsquared_adj)
    assert res.nobs == 500 and res.df_resid == 496 and res.df_model == 3
    assert res.names == ["const", "educ", "exper", "female"]
    assert res.dep_name == "lwage"


def test_f_test_matches_statsmodels(wage_like):
    df, _ = wage_like
    x = add_constant(df[["educ", "exper", "female"]])
    res = OLS(df["lwage"], x).fit()
    ref = sm.OLS(df["lwage"], x).fit()
    f = res.f_test()
    assert f.statistic == pytest.approx(ref.fvalue)
    assert f.pvalue == pytest.approx(ref.f_pvalue, abs=1e-12)
    assert f.df_num == 3 and f.df_denom == 496


def test_wald_test_with_restriction_values(wage_like):
    df, _ = wage_like
    x = add_constant(df[["educ", "exper", "female"]])
    res = OLS(df["lwage"], x).fit(cov_type="HC1")
    ref = sm.OLS(df["lwage"], x).fit(cov_type="HC1", use_t=True)
    r_mat = np.array([[0, 1, 0, 0], [0, 0, 1, -1]])
    r_val = np.array([0.08, 0.1])
    ours = res.wald_test(r_mat, r_val)
    theirs = ref.wald_test((r_mat, r_val), use_f=True, scalar=True)
    assert ours.statistic == pytest.approx(float(theirs.statistic))
    assert ours.pvalue == pytest.approx(float(theirs.pvalue), abs=1e-12)


def test_robust_and_cluster_fits(wage_like):
    df, firm = wage_like
    x = add_constant(df[["educ", "exper", "female"]])
    for cov_type, kwds in [("HC3", {}), ("cluster", {"groups": firm})]:
        res = OLS(df["lwage"], x).fit(cov_type=cov_type, **kwds)
        cov_kwds = {"groups": firm} if cov_type == "cluster" else {}
        ref = sm.OLS(df["lwage"], x).fit(cov_type=cov_type, cov_kwds=cov_kwds, use_t=True)
        np.testing.assert_allclose(res.bse, ref.bse.to_numpy(), rtol=1e-9)
        np.testing.assert_allclose(res.pvalues, ref.pvalues.to_numpy(), rtol=1e-7, atol=1e-12)


def test_use_t_false_gives_normal_pvalues(wage_like):
    df, _ = wage_like
    x = add_constant(df[["educ", "exper", "female"]])
    res = OLS(df["lwage"], x).fit(cov_type="HC0", use_t=False)
    ref = sm.OLS(df["lwage"], x).fit(cov_type="HC0", use_t=False)
    np.testing.assert_allclose(res.pvalues, ref.pvalues.to_numpy(), rtol=1e-7, atol=1e-12)
    w = res.wald_test(np.eye(4)[1:])
    assert w.distribution == "chi2"


def test_numpy_inputs_get_default_names_and_constant_detection():
    rng = np.random.default_rng(3)
    x = add_constant(rng.normal(size=(50, 2)))
    y = x @ [1.0, 2.0, 3.0] + rng.normal(size=50)
    res = OLS(y, x).fit()
    assert res.names == ["const", "x1", "x2"]
    assert res.has_constant
    assert res.predict([1.0, 0.0, 0.0]).shape == (1,)
    assert res.to_frame().shape == (3, 6)
    text = res.summary()
    assert "OLS Regression Results" in text and "const" in text


def test_add_constant_is_idempotent():
    x = np.ones((10, 1))
    assert add_constant(x).shape == (10, 1)
    df = pd.DataFrame({"a": np.arange(10.0)})
    out = add_constant(df)
    assert list(out.columns) == ["const", "a"]
    assert add_constant(out).shape == (10, 2)


def test_shape_mismatch_and_nan_are_rejected():
    with pytest.raises(ValueError, match="rows"):
        OLS(np.ones(5), np.ones((6, 1)))
    with pytest.raises(ValueError, match="NaN"):
        OLS(np.array([1.0, np.nan]), np.ones((2, 1)))
