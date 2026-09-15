"""Ordinary least squares."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ._linalg import as_1d, as_2d, qr_solve
from .covariance import compute_covariance
from .results import RegressionResults


def add_constant(x: ArrayLike, name: str = "const") -> np.ndarray | pd.DataFrame:
    """Prepend a column of ones (skipped if one is already there)."""
    if isinstance(x, pd.DataFrame):
        if any(np.allclose(x[c].to_numpy(dtype=float), 1.0) for c in x.columns):
            return x
        return pd.concat([pd.Series(1.0, index=x.index, name=name), x], axis=1)
    arr = as_2d(x)
    if any(np.allclose(arr[:, j], 1.0) for j in range(arr.shape[1])):
        return arr
    return np.column_stack([np.ones(arr.shape[0]), arr])


def _names_from(x: ArrayLike, names: list[str] | None, k: int, prefix: str = "x") -> list[str]:
    if names is not None:
        if len(names) != k:
            raise ValueError(f"expected {k} names, got {len(names)}")
        return [str(n) for n in names]
    if isinstance(x, pd.DataFrame):
        return [str(c) for c in x.columns]
    return [f"{prefix}{j}" for j in range(k)]


def _dep_name(y: ArrayLike) -> str:
    if isinstance(y, pd.Series) and y.name is not None:
        return str(y.name)
    if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        return str(y.columns[0])
    return "y"


class OLS:
    """``OLS(y, X).fit(cov_type=...)``.

    ``X`` must already contain a constant if you want one (use :func:`add_constant`);
    nothing is added silently. DataFrame columns become coefficient names.
    """

    def __init__(self, y: ArrayLike, x: ArrayLike, names: list[str] | None = None):
        self.y = as_1d(y)
        self.x = as_2d(x)
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(f"y has {self.y.shape[0]} rows but X has {self.x.shape[0]}")
        self.names = _names_from(x, names, self.x.shape[1])
        self.dep_name = _dep_name(y)
        self.has_constant = bool(
            any(np.allclose(self.x[:, j], 1.0) for j in range(self.x.shape[1]))
        )
        # rename a detected constant column so f_test() knows to leave it out
        if self.has_constant and names is None and not isinstance(x, pd.DataFrame):
            j = next(j for j in range(self.x.shape[1]) if np.allclose(self.x[:, j], 1.0))
            self.names[j] = "const"

    def fit(
        self,
        cov_type: str = "nonrobust",
        *,
        groups: ArrayLike | None = None,
        maxlags: int | None = None,
        use_correction: bool | None = None,
        use_t: bool = True,
    ) -> RegressionResults:
        """Estimate and attach a covariance matrix of the requested type.

        Parameters
        ----------
        cov_type : "nonrobust", "HC0"–"HC3", "cluster" or "HAC".
        groups : cluster labels (one array, or a pair for two-way clustering).
        maxlags : Newey–West bandwidth for ``cov_type="HAC"``.
        use_correction : override the estimator's default small-sample scaling.
        use_t : use ``t``/``F`` with ``df_inference`` (default) instead of ``z``/``chi2``.
        """
        n, k = self.x.shape
        beta, xtx_inv = qr_solve(self.x, self.y)
        fitted = self.x @ beta
        resid = self.y - fitted
        df_resid = n - k
        cov = compute_covariance(
            cov_type,
            self.x,
            resid,
            xtx_inv,
            df_resid,
            groups=groups,
            maxlags=maxlags,
            use_correction=use_correction,
        )
        centered = self.y - self.y.mean() if self.has_constant else self.y
        return RegressionResults(
            params=beta,
            cov=cov.cov,
            names=list(self.names),
            resid=resid,
            fitted=fitted,
            nobs=n,
            df_model=k - int(self.has_constant),
            df_resid=df_resid,
            df_inference=cov.df_inference,
            cov_type=cov.cov_type,
            cov_description=cov.description,
            rss=float(resid @ resid),
            tss=float(centered @ centered),
            use_t=use_t,
            model="OLS",
            dep_name=self.dep_name,
            has_constant=self.has_constant,
            info=dict(cov.info),
        )
