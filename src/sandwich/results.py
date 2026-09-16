"""Regression results: inference, Wald tests, prediction and a text summary."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats


@dataclass
class WaldTestResult:
    """Outcome of a Wald test of ``R @ beta = r``."""

    statistic: float
    pvalue: float
    df_num: int
    df_denom: int | None
    distribution: str  # "F" or "chi2"

    def __repr__(self) -> str:
        if self.distribution == "F":
            return (
                f"<Wald test: F({self.df_num}, {self.df_denom}) = {self.statistic:.4f}, "
                f"p = {self.pvalue:.4g}>"
            )
        return f"<Wald test: chi2({self.df_num}) = {self.statistic:.4f}, p = {self.pvalue:.4g}>"


@dataclass
class RegressionResults:
    """Estimates plus everything needed for inference.

    Attributes are deliberately close to statsmodels' names (``params``, ``bse``,
    ``tvalues``, ``pvalues``, ``conf_int``) so the two can be compared side by side.
    """

    params: NDArray[np.float64]
    cov: NDArray[np.float64]
    names: list[str]
    resid: NDArray[np.float64]
    fitted: NDArray[np.float64]
    nobs: int
    df_model: int
    df_resid: int
    df_inference: int
    cov_type: str
    cov_description: str
    rss: float
    tss: float
    use_t: bool = True
    model: str = "OLS"
    dep_name: str = "y"
    has_constant: bool = True
    info: dict = field(default_factory=dict)
    effects: NDArray[np.float64] | None = None  # estimated fixed effects per observation

    # -- goodness of fit -----------------------------------------------------------------
    @property
    def r2(self) -> float:
        return 1.0 - self.rss / self.tss if self.tss > 0 else float("nan")

    @property
    def r2_adj(self) -> float:
        if self.tss <= 0 or self.df_resid <= 0:
            return float("nan")
        return 1.0 - (self.rss / self.df_resid) / (self.tss / (self.nobs - int(self.has_constant)))

    # -- coefficient-level inference -------------------------------------------------------
    @property
    def bse(self) -> NDArray[np.float64]:
        return np.sqrt(np.diag(self.cov))

    @property
    def tvalues(self) -> NDArray[np.float64]:
        return self.params / self.bse

    @property
    def pvalues(self) -> NDArray[np.float64]:
        t = np.abs(self.tvalues)
        if self.use_t:
            return 2.0 * stats.t.sf(t, self.df_inference)
        return 2.0 * stats.norm.sf(t)

    def _crit(self, alpha: float) -> float:
        if self.use_t:
            return float(stats.t.ppf(1.0 - alpha / 2.0, self.df_inference))
        return float(stats.norm.ppf(1.0 - alpha / 2.0))

    def conf_int(self, alpha: float = 0.05) -> NDArray[np.float64]:
        """``(k, 2)`` array of lower and upper confidence bounds."""
        half = self._crit(alpha) * self.bse
        return np.column_stack([self.params - half, self.params + half])

    # -- joint inference ---------------------------------------------------------------------
    def wald_test(self, r_matrix: ArrayLike, r_value: ArrayLike | None = None) -> WaldTestResult:
        """Test ``R beta = r``. Rows of ``R`` are restrictions; ``r`` defaults to zeros.

        With ``use_t`` the statistic is reported as ``F(q, df_inference) = W / q``;
        otherwise as ``chi2(q) = W``.
        """
        r_mat = np.atleast_2d(np.asarray(r_matrix, dtype=np.float64))
        q = r_mat.shape[0]
        if r_mat.shape[1] != self.params.size:
            raise ValueError(f"R must have {self.params.size} columns, got {r_mat.shape[1]}")
        r_val = np.zeros(q) if r_value is None else np.asarray(r_value, dtype=np.float64).ravel()
        if r_val.shape != (q,):
            raise ValueError(f"r must have length {q}")
        diff = r_mat @ self.params - r_val
        middle = r_mat @ self.cov @ r_mat.T
        w = float(diff @ np.linalg.solve(middle, diff))
        if self.use_t:
            f = w / q
            return WaldTestResult(
                f, float(stats.f.sf(f, q, self.df_inference)), q, self.df_inference, "F"
            )
        return WaldTestResult(w, float(stats.chi2.sf(w, q)), q, None, "chi2")

    def f_test(self, names: list[str] | None = None) -> WaldTestResult:
        """Joint test that a set of coefficients is zero (default: all but the constant)."""
        if names is None:
            names = [n for n in self.names if n != "const"]
        idx = [self.names.index(n) for n in names]
        r_mat = np.zeros((len(idx), self.params.size))
        r_mat[np.arange(len(idx)), idx] = 1.0
        return self.wald_test(r_mat)

    # -- convenience ---------------------------------------------------------------------------
    def predict(self, x: ArrayLike) -> NDArray[np.float64]:
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr[None, :]
        return x_arr @ self.params

    def to_frame(self, alpha: float = 0.05) -> pd.DataFrame:
        """Coefficient table as a DataFrame."""
        ci = self.conf_int(alpha)
        label = "t" if self.use_t else "z"
        return pd.DataFrame(
            {
                "coef": self.params,
                "std err": self.bse,
                label: self.tvalues,
                f"P>|{label}|": self.pvalues,
                f"[{alpha / 2:.3f}": ci[:, 0],
                f"{1 - alpha / 2:.3f}]": ci[:, 1],
            },
            index=self.names,
        )

    def summary(self, alpha: float = 0.05) -> str:
        """A statsmodels-style text table."""
        width = 78
        title = f"{self.model} Regression Results".center(width)
        try:
            ftest = self.f_test()
            fline = f"{ftest.statistic:.4g} (p = {ftest.pvalue:.3g})"
        except (ValueError, np.linalg.LinAlgError):
            fline = "n/a"
        left = [
            ("Dep. Variable:", self.dep_name),
            ("Model:", self.model),
            ("No. Observations:", str(self.nobs)),
            ("Df Residuals:", str(self.df_resid)),
            ("Df Model:", str(self.df_model)),
            ("Covariance Type:", self.cov_type),
        ]
        right = [
            ("R-squared:", f"{self.r2:.4f}"),
            ("Adj. R-squared:", f"{self.r2_adj:.4f}"),
            ("F-statistic:", fline),
            ("Df inference:", str(self.df_inference)),
            ("Inference:", "t / F" if self.use_t else "z / chi2"),
            ("", self.cov_description),
        ]
        lines = [title, "=" * width]
        for (lk, lv), (rk, rv) in zip(left, right, strict=True):
            lines.append(f"{lk:<18}{lv:>20}   {rk:<16}{rv:>21}")
        lines.append("=" * width)
        label = "t" if self.use_t else "z"
        lines.append(
            f"{'':<16}{'coef':>10}{'std err':>10}{label:>9}{'P>|' + label + '|':>9}"
            f"{'[' + f'{alpha / 2:.3f}':>10}{f'{1 - alpha / 2:.3f}' + ']':>10}"
        )
        lines.append("-" * width)
        ci = self.conf_int(alpha)
        for i, name in enumerate(self.names):
            lines.append(
                f"{name[:16]:<16}{self.params[i]:>10.4f}{self.bse[i]:>10.4f}{self.tvalues[i]:>9.3f}"
                f"{self.pvalues[i]:>9.3f}{ci[i, 0]:>10.3f}{ci[i, 1]:>10.3f}"
            )
        lines.append("=" * width)
        for key, value in self.info.items():
            if value is None or isinstance(value, np.ndarray):
                continue
            shown = f"{value:.4f}" if isinstance(value, float) else str(value)
            lines.append(f"{key}: {shown}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()
