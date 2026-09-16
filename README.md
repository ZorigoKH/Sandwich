# sandwich

An econometrics engine in NumPy: OLS, sandwich (HC / cluster / HAC) standard errors,
IV/2SLS, panel fixed effects, first differences and Wald tests — every number verified
against statsmodels and linearmodels.

The name is the point. Every covariance estimator in econometrics is the same object,

```
V(β̂) = (X'X)⁻¹ · M · (X'X)⁻¹        bread · meat · bread
```

and only the **meat** `M` — the estimate of `Var(X'e)` — changes:

| `cov_type`  | meat                                                  | inference df |
|-------------|-------------------------------------------------------|--------------|
| `nonrobust` | `s² X'X`                                              | `n − k`      |
| `HC0`–`HC3` | `X' diag(e²) X`, with MacKinnon–White leverage scaling | `n − k`      |
| `cluster`   | `Σ_g (X_g'e_g)(X_g'e_g)'` · `G/(G−1) · (n−1)/(n−k)`     | `G − 1`      |
| `HAC`       | Newey–West with Bartlett weights                        | `n − k`      |

Two-way clustering (Cameron–Gelbach–Miller) is `V₁ + V₂ − V₁₂`.

Fixed effects change the bread, not the recipe: `X` becomes the within-transformed
`X̃`, and the absorbed effects are charged to the degrees of freedom — except an effect
nested inside the clusters, which the cluster sums already absorb (Stata's `xtreg, fe`,
linearmodels' `auto_df`).

## Usage

```python
import pandas as pd
from sandwich import OLS, IV2SLS, PanelOLS, FirstDifferenceOLS, add_constant

df = pd.read_csv("wages.csv")
X = add_constant(df[["educ", "exper", "female"]])

res = OLS(df["lwage"], X).fit(cov_type="cluster", groups=df["firm"])
print(res.summary())
res.f_test(["educ", "exper"])            # joint Wald test, F(2, G-1)
res.wald_test([[0, 1, -1, 0]])           # educ == exper

# returns to schooling with distance-to-college as the instrument
iv = IV2SLS(df["lwage"], add_constant(df[["exper"]]), df["educ"], df[["dist"]]).fit(cov_type="HC1")
print(iv.info["diagnostics"])            # first-stage F, Wu-Hausman, Sargan, Hansen J

# firm and year fixed effects, clustered by firm (a time-invariant column is rejected as absorbed)
fe = PanelOLS(df["lwage"], df[["exper", "tenure"]], df["firm"], df["year"],
              entity_effects=True, time_effects=True).fit(cov_type="cluster", groups="entity")
fe.info["r2_within"], fe.info["f_effects"]   # within R², F test that all effects are zero
fe.effects                                   # α̂_i + λ̂_t for every observation

fd = FirstDifferenceOLS(df["lwage"], df[["exper", "tenure"]], df["firm"], df["year"]).fit()
```

A pandas object indexed by an `(entity, time)` MultiIndex needs no `entity`/`time` arguments.

## What is implemented

- **OLS** via a thin QR decomposition (no `inv(X'X)`), rank-deficiency detection,
  `add_constant`, DataFrame-aware coefficient names.
- **Covariances**: nonrobust, HC0–HC3, one- and two-way cluster-robust, Newey–West HAC,
  each with the small-sample conventions Stata and statsmodels use.
- **Inference**: t or z tests, confidence intervals, arbitrary linear Wald tests
  `Rβ = r` reported as F or χ², a statsmodels-style `summary()`.
- **IV / 2SLS** with any of the covariances above, plus first-stage F and partial R²,
  the Wu–Hausman endogeneity test, and Sargan and (robust) Hansen J
  over-identification tests.
- **Panel fixed effects** (entity, time, or both) by the within transformation — no
  dummy per entity; two-way effects on unbalanced panels are partialled out exactly
  (Frisch–Waugh–Lovell). Nonrobust, HC0/HC1 and one- or two-way clustered covariances
  with the absorbed effects counted correctly; within / between / overall R², the
  pooled F test for the effects, the estimated effects, and a clear error for a
  regressor the effects absorb.
- **First differences** between consecutive periods only (gaps are not differenced),
  with every OLS covariance type except HAC.

Every estimator is tested against statsmodels or linearmodels on the same data to
`rtol=1e-9`; the tests are the spec.

## Roadmap

- A small formula interface, `ols("lwage ~ educ + C(region)", df)`.
- Random effects and the Hausman test.
- Driscoll–Kraay standard errors for panels.

## Development

```
pip install -e ".[dev]" linearmodels
pytest
ruff check src tests
```

MIT.
