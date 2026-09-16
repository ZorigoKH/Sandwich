"""Fixed effects and first differences are checked against linearmodels on the same
unbalanced panel, and the clustered degrees-of-freedom rule against a statsmodels
regression on the equivalent dummy-variable design."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from linearmodels.panel import FirstDifferenceOLS as LMFirstDifferenceOLS
from linearmodels.panel import PanelOLS as LMPanelOLS

from sandwich.ols import OLS
from sandwich.panel import FirstDifferenceOLS, PanelOLS, _demean


@pytest.fixture(scope="module")
def panel():
    """40 firms x 8 years, ~15% of cells missing, effects correlated with the regressors."""
    rng = np.random.default_rng(5)
    n_firms, n_years = 40, 8
    firm = np.repeat(np.arange(n_firms), n_years)
    year = np.tile(np.arange(2000, 2000 + n_years), n_firms)
    n = n_firms * n_years
    alpha = rng.normal(size=n_firms)[firm]
    lam = rng.normal(size=n_years)[year - 2000]
    x1 = rng.normal(size=n) + alpha  # pooled OLS is biased for this one
    x2 = rng.normal(size=n) + 0.3 * lam
    e = rng.normal(size=n) * (1 + 0.3 * np.abs(x1)) + 0.4 * rng.normal(size=n_firms)[firm]
    y = 1.0 + 0.7 * x1 - 0.4 * x2 + alpha + lam + e
    keep = rng.uniform(size=n) > 0.15
    keep[:n_years] = True  # firm 0 complete, so the year axis is sorted for linearmodels too
    index = pd.MultiIndex.from_arrays([firm, year], names=["firm", "year"])
    df = pd.DataFrame({"y": y, "const": 1.0, "x1": x1, "x2": x2}, index=index)[keep]
    return df


def _assert_matches(ours, ref, rtol=1e-9):
    np.testing.assert_allclose(ours.params, ref.params.to_numpy(), rtol=rtol)
    np.testing.assert_allclose(ours.bse, ref.std_errors.to_numpy(), rtol=rtol)
    assert ours.df_resid == ref.df_resid
    assert ours.r2 == pytest.approx(ref.rsquared)


@pytest.mark.parametrize("cols", [["const", "x1", "x2"], ["x1", "x2"]])
@pytest.mark.parametrize(
    "effects",
    [
        {"entity_effects": True, "time_effects": False},
        {"entity_effects": False, "time_effects": True},
        {"entity_effects": True, "time_effects": True},
    ],
)
def test_fixed_effects_match_linearmodels(panel, cols, effects):
    y, x = panel["y"], panel[cols]
    ours = PanelOLS(y, x, **effects).fit()
    ref = LMPanelOLS(y, x, **effects).fit()
    _assert_matches(ours, ref)
    assert ours.names == cols
    assert ours.info["r2_within"] == pytest.approx(ref.rsquared_within)
    assert ours.info["r2_between"] == pytest.approx(ref.rsquared_between)
    assert ours.info["r2_overall"] == pytest.approx(ref.rsquared_overall)
    np.testing.assert_allclose(ours.effects, ref.estimated_effects.to_numpy().ravel(), atol=1e-10)
    f = ours.info["f_effects"]
    assert f.statistic == pytest.approx(ref.f_pooled.stat)
    assert f.df_num == ref.f_pooled.df and f.df_denom == ref.f_pooled.df_denom
    assert f.pvalue == pytest.approx(ref.f_pooled.pval, abs=1e-12)


@pytest.mark.parametrize(
    "effects",
    [
        {"entity_effects": True, "time_effects": False},
        {"entity_effects": True, "time_effects": True},
    ],
)
def test_robust_covariances_match_linearmodels(panel, effects):
    y, x = panel["y"], panel[["const", "x1", "x2"]]
    model = PanelOLS(y, x, **effects)
    ref_model = LMPanelOLS(y, x, **effects)
    # HC1 scales by n / (n - k - absorbed): linearmodels' debiased robust default
    _assert_matches(model.fit(cov_type="HC1"), ref_model.fit(cov_type="robust"))
    # HC0 is the bare sandwich
    ref = ref_model.fit(cov_type="robust", debiased=False, auto_df=False, count_effects=False)
    _assert_matches(model.fit(cov_type="HC0"), ref)


def test_entity_clusters_match_linearmodels(panel):
    y, x = panel["y"], panel[["const", "x1", "x2"]]
    model = PanelOLS(y, x)
    ref_model = LMPanelOLS(y, x, entity_effects=True)
    # default: Stata's G/(G-1) (n-1)/(n-k) with the nested entity effects not charged
    ours = model.fit(cov_type="cluster", groups="entity")
    ref = ref_model.fit(cov_type="clustered", cluster_entity=True, group_debias=True)
    _assert_matches(ours, ref)
    assert ours.df_inference == 39 and ours.info["n_clusters"] == 40
    # no small-sample factor at all
    ours = model.fit(cov_type="cluster", groups="entity", use_correction=False)
    ref = ref_model.fit(cov_type="clustered", cluster_entity=True, debiased=False)
    _assert_matches(ours, ref)
    # two-way clusters by entity and year, charging for the effects as linearmodels does
    ours = model.fit(cov_type="cluster", groups=("entity", "time"), count_effects=True)
    ref = ref_model.fit(
        cov_type="clustered", cluster_entity=True, cluster_time=True, group_debias=True
    )
    _assert_matches(ours, ref)
    assert ours.df_inference == 7


def test_cluster_df_charges_only_effects_not_nested_in_the_clusters(panel):
    """Two-way effects clustered by entity: the entity effects are nested in the clusters,
    so only the time effects cost degrees of freedom -- exactly what OLS on the
    entity-demeaned data with explicit year dummies charges."""
    y, x = panel["y"], panel[["x1", "x2"]]
    model = PanelOLS(y, x, entity_effects=True, time_effects=True)
    ours = model.fit(cov_type="cluster", groups="entity")
    codes, n_firms = model.entity_codes, model.n_entities
    dummies = np.eye(model.n_periods)[model.time_codes][:, 1:]
    lsdv_x = np.column_stack([_demean(model.x, codes, n_firms), _demean(dummies, codes, n_firms)])
    ref = sm.OLS(_demean(model.y, codes, n_firms), lsdv_x).fit(
        cov_type="cluster", cov_kwds={"groups": codes}
    )
    np.testing.assert_allclose(ours.params, ref.params[:2], rtol=1e-9)
    np.testing.assert_allclose(ours.bse, ref.bse[:2], rtol=1e-9)
    # charging for every effect reproduces linearmodels' default instead
    ours_all = model.fit(cov_type="cluster", groups="entity", count_effects=True)
    ref_all = LMPanelOLS(y, x, entity_effects=True, time_effects=True).fit(
        cov_type="clustered", cluster_entity=True, group_debias=True
    )
    np.testing.assert_allclose(ours_all.bse, ref_all.std_errors.to_numpy(), rtol=1e-9)
    assert ours.bse[0] < ours_all.bse[0]


def test_explicit_index_arrays_agree_with_multiindex(panel):
    flat = panel.reset_index()
    a = PanelOLS(flat["y"], flat[["x1", "x2"]], flat["firm"], flat["year"], time_effects=True)
    b = PanelOLS(panel["y"], panel[["x1", "x2"]], time_effects=True)
    ra, rb = (
        a.fit(cov_type="cluster", groups=flat["firm"]),
        b.fit(cov_type="cluster", groups="entity"),
    )
    np.testing.assert_allclose(ra.params, rb.params)
    np.testing.assert_allclose(ra.bse, rb.bse)
    assert ra.info["n_entities"] == 40 and ra.info["n_periods"] == 8


def test_within_estimator_removes_the_bias_pooled_ols_has(panel):
    y = panel["y"]
    pooled = OLS(y, panel[["const", "x1", "x2"]]).fit()
    within = PanelOLS(y, panel[["x1", "x2"]]).fit()
    assert abs(pooled.params[1] - 0.7) > 3 * pooled.bse[1]  # alpha is correlated with x1
    assert abs(within.params[0] - 0.7) < 3 * within.bse[0]
    text = within.summary()
    assert "PanelOLS" in text and "entity (40 levels)" in text and "f_effects" in text


def test_absorbed_regressors_and_unsupported_covariances_are_rejected(panel):
    firm = panel.index.get_level_values("firm").to_numpy()
    time_invariant = pd.DataFrame(
        {"x1": panel["x1"], "female": (firm % 2).astype(float)}, index=panel.index
    )
    with pytest.raises(ValueError, match="female: absorbed"):
        PanelOLS(panel["y"], time_invariant).fit()
    for cov_type in ["HC2", "HC3", "HAC"]:
        with pytest.raises(ValueError, match="not available with absorbed effects"):
            PanelOLS(panel["y"], panel[["x1", "x2"]]).fit(cov_type=cov_type)
    with pytest.raises(ValueError, match="entity_effects=True and/or time_effects=True"):
        PanelOLS(panel["y"], panel[["x1"]], entity_effects=False)
    with pytest.raises(ValueError, match="entity= is required"):
        PanelOLS(panel["y"].to_numpy(), panel[["x1"]].to_numpy())
    with pytest.raises(ValueError, match="time= is required"):
        PanelOLS(panel["y"].to_numpy(), panel[["x1"]].to_numpy(), entity=firm, time_effects=True)


def test_first_differences_match_linearmodels(panel):
    y, x = panel["y"], panel[["x1", "x2"]]
    model, ref_model = FirstDifferenceOLS(y, x), LMFirstDifferenceOLS(y, x)
    ours, ref = model.fit(), ref_model.fit()
    _assert_matches(ours, ref)
    assert ours.nobs == ref.nobs
    assert ours.model == "FirstDifferenceOLS" and ours.dep_name == "Δy"
    _assert_matches(model.fit(cov_type="HC1"), ref_model.fit(cov_type="robust"))
    ours = model.fit(cov_type="cluster", groups="entity")
    ref = ref_model.fit(cov_type="clustered", cluster_entity=True, group_debias=True)
    _assert_matches(ours, ref)
    # an explicit cluster array constant within firms is the same thing
    firm = panel.index.get_level_values("firm").to_numpy()
    same = model.fit(cov_type="cluster", groups=firm)
    np.testing.assert_allclose(same.bse, ours.bse)


def test_first_differences_skip_gaps_and_reject_constants(panel):
    y, x = panel["y"], panel[["x1", "x2"]]
    model = FirstDifferenceOLS(y, x)
    firm = panel.index.get_level_values("firm").to_numpy()
    year = panel.index.get_level_values("year").to_numpy()
    cells = set(zip(firm, year, strict=True))
    consecutive = sum((f, t - 1) in cells for f, t in cells)
    assert model.dy.shape[0] == consecutive == model.fit().nobs
    np.testing.assert_allclose(model.dy, y.to_numpy()[model.current] - y.to_numpy()[model.previous])
    with pytest.raises(ValueError, match="constant differences to zero"):
        FirstDifferenceOLS(y, panel[["const", "x1"]])
    with pytest.raises(ValueError, match="identical within each differenced pair"):
        model.fit(cov_type="cluster", groups=year)
    with pytest.raises(ValueError, match="HAC"):
        model.fit(cov_type="HAC")
