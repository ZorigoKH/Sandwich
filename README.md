# sandwich

An econometrics engine in NumPy: OLS, sandwich (HC / cluster / HAC) standard errors,
IV/2SLS and Wald tests — every number verified against statsmodels and linearmodels.

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

## Usage

```python
import pandas as pd
from sandwich import OLS, IV2SLS, add_constant

df = pd.read_csv("wages.csv")
X = add_constant(df[["educ", "exper", "female"]])

res = OLS(df["lwage"], X).fit(cov_type="cluster", groups=df["firm"])
print(res.summary())
res.f_test(["educ", "exper"])            # joint Wald test, F(2, G-1)
res.wald_test([[0, 1, -1, 0]])           # educ == exper

# returns to schooling with distance-to-college as the instrument
iv = IV2SLS(df["lwage"], add_constant(df[["exper"]]), df["educ"], df[["dist"]]).fit(cov_type="HC1")
print(iv.info["diagnostics"])            # first-stage F, Wu-Hausman, Sargan, Hansen J
```

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

Every estimator is tested against statsmodels or linearmodels on the same data to
`rtol=1e-9`; the tests are the spec.

## Roadmap

- Panel: fixed effects (one- and two-way, within transformation) and first differences.
- A small formula interface, `ols("lwage ~ educ + C(region)", df)`.
- Random effects and the Hausman test.

## Development

```
pip install -e ".[dev]" linearmodels
pytest
ruff check src tests
```

MIT.
