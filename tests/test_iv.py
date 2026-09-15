import numpy as np
import pytest
from linearmodels.iv import IV2SLS as LM_IV2SLS

from sandwich.iv import IV2SLS


@pytest.fixture(scope="module")
def iv_data():
    """Returns-to-schooling style DGP: schooling is endogenous, two valid instruments."""
    rng = np.random.default_rng(11)
    n = 600
    ability = rng.normal(size=n)  # unobserved, drives both schooling and wages
    z1 = rng.normal(size=n)  # distance to college
    z2 = rng.normal(size=n)  # tuition
    exper = rng.uniform(0, 20, size=n)
    school = 12 + 0.8 * z1 - 0.5 * z2 + 0.6 * ability + rng.normal(size=n)
    het = 0.4 + 0.03 * exper
    lwage = 1.0 + 0.09 * school + 0.03 * exper + 0.5 * ability + het * rng.normal(size=n)
    cluster = rng.integers(0, 30, size=n)
    exog = np.column_stack([np.ones(n), exper])
    return lwage, exog, school, np.column_stack([z1, z2]), cluster


def test_2sls_matches_linearmodels_unadjusted(iv_data):
    y, exog, endog, inst, _ = iv_data
    res = IV2SLS(y, exog, endog, inst).fit()
    ref = LM_IV2SLS(y, exog, endog, inst).fit(cov_type="unadjusted", debiased=True)
    np.testing.assert_allclose(res.params, ref.params.to_numpy(), rtol=1e-9)
    np.testing.assert_allclose(res.bse, ref.std_errors.to_numpy(), rtol=1e-9)
    np.testing.assert_allclose(res.pvalues, ref.pvalues.to_numpy(), rtol=1e-6, atol=1e-12)
    assert res.names == ["const", "x1", "endog0"]
    assert res.model == "IV-2SLS"


def test_2sls_not_debiased_matches_linearmodels_default(iv_data):
    y, exog, endog, inst, _ = iv_data
    res = IV2SLS(y, exog, endog, inst).fit(debiased=False, use_t=False)
    ref = LM_IV2SLS(y, exog, endog, inst).fit(cov_type="unadjusted")
    np.testing.assert_allclose(res.bse, ref.std_errors.to_numpy(), rtol=1e-9)
    np.testing.assert_allclose(res.pvalues, ref.pvalues.to_numpy(), rtol=1e-6, atol=1e-12)


def test_2sls_robust_and_clustered(iv_data):
    y, exog, endog, inst, cluster = iv_data
    res = IV2SLS(y, exog, endog, inst).fit(cov_type="HC0", debiased=False)
    ref = LM_IV2SLS(y, exog, endog, inst).fit(cov_type="robust")
    np.testing.assert_allclose(res.bse, ref.std_errors.to_numpy(), rtol=1e-9)

    res_c = IV2SLS(y, exog, endog, inst).fit(
        cov_type="cluster", groups=cluster, use_correction=False
    )
    ref_c = LM_IV2SLS(y, exog, endog, inst).fit(cov_type="clustered", clusters=cluster)
    np.testing.assert_allclose(res_c.bse, ref_c.std_errors.to_numpy(), rtol=1e-9)
    assert res_c.df_inference == 29


def test_2sls_recovers_the_true_effect_where_ols_does_not(iv_data):
    y, exog, endog, inst, _ = iv_data
    from sandwich.ols import OLS

    ols = OLS(y, np.column_stack([exog, endog])).fit()
    iv = IV2SLS(y, exog, endog, inst).fit()
    assert abs(iv.params[-1] - 0.09) < 3 * iv.bse[-1]
    assert abs(ols.params[-1] - 0.09) > 3 * ols.bse[-1]  # ability bias


def test_diagnostics_match_linearmodels(iv_data):
    y, exog, endog, inst, _ = iv_data
    res = IV2SLS(y, exog, endog, inst).fit()
    ref = LM_IV2SLS(y, exog, endog, inst).fit(cov_type="unadjusted", debiased=True)
    diag = res.info["diagnostics"]
    fs = diag.first_stage[0]
    ref_fs = ref.first_stage.diagnostics.iloc[0]
    assert fs.r2 == pytest.approx(ref_fs["rsquared"])
    assert fs.partial_r2 == pytest.approx(ref_fs["partial.rsquared"])
    assert fs.f_stat == pytest.approx(ref_fs["f.stat"], rel=1e-6)
    assert not fs.weak
    s_stat, s_p, df = diag.sargan
    assert df == 1
    assert s_stat == pytest.approx(ref.sargan.stat, rel=1e-6)
    assert s_p == pytest.approx(ref.sargan.pval, abs=1e-6)
    # linearmodels uses a slightly different small-sample convention for Wu-Hausman,
    # so agree to ~1% rather than to machine precision
    wh = diag.wu_hausman
    assert wh.statistic == pytest.approx(ref.wu_hausman().stat, rel=1e-2)
    assert wh.pvalue == pytest.approx(ref.wu_hausman().pval, abs=1e-3)
    assert "Wu-Hausman" in repr(diag) and "Hansen J" in repr(diag)


def test_exactly_identified_has_no_overid_test(iv_data):
    y, exog, endog, inst, _ = iv_data
    res = IV2SLS(y, exog, endog, inst[:, :1]).fit()
    assert res.info["diagnostics"].sargan is None
    assert "exactly identified" in repr(res.info["diagnostics"])


def test_under_identified_is_rejected(iv_data):
    y, exog, endog, inst, _ = iv_data
    with pytest.raises(ValueError, match="under-identified"):
        IV2SLS(y, exog, np.column_stack([endog, endog**2]), inst[:, :1])
