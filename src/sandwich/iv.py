"""Instrumental variables: two-stage least squares with the usual diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

from ._linalg import as_1d, as_2d, qr_solve
from .covariance import compute_covariance
from .ols import OLS, _dep_name, _names_from
from .results import RegressionResults, WaldTestResult


@dataclass
class FirstStage:
    """First-stage diagnostics for one endogenous regressor."""

    name: str
    f_stat: float
    f_pvalue: float
    r2: float
    partial_r2: float

    @property
    def weak(self) -> bool:
        """Staiger–Stock rule of thumb: an F below 10 signals weak instruments."""
        return self.f_stat < 10.0


@dataclass
class IVDiagnostics:
    first_stage: list[FirstStage]
    wu_hausman: WaldTestResult
    sargan: tuple[float, float, int] | None  # (statistic, p-value, df) if overidentified
    hansen_j: tuple[float, float, int] | None

    def __repr__(self) -> str:
        lines = ["IV diagnostics"]
        for fs in self.first_stage:
            flag = "  (weak: F < 10)" if fs.weak else ""
            lines.append(
                f"  first stage [{fs.name}]: F = {fs.f_stat:.3f} (p = {fs.f_pvalue:.3g}), "
                f"R2 = {fs.r2:.3f}, partial R2 = {fs.partial_r2:.3f}{flag}"
            )
        wh = self.wu_hausman
        lines.append(
            f"  Wu-Hausman endogeneity: {wh.distribution}({wh.df_num}"
            + (f", {wh.df_denom}" if wh.df_denom is not None else "")
            + f") = {wh.statistic:.3f} (p = {wh.pvalue:.3g})"
        )
        if self.sargan is None:
            lines.append("  overidentification: exactly identified, no test")
        else:
            s, p, df = self.sargan
            j, jp, _ = self.hansen_j
            lines.append(f"  Sargan: chi2({df}) = {s:.3f} (p = {p:.3g})")
            lines.append(f"  Hansen J (robust): chi2({df}) = {j:.3f} (p = {jp:.3g})")
        return "\n".join(lines)


class IV2SLS:
    """``IV2SLS(y, exog, endog, instruments).fit()``.

    ``exog`` are the included exogenous regressors (put the constant here), ``endog`` the
    endogenous regressors and ``instruments`` the *excluded* instruments. Any covariance
    type accepted by :class:`~sandwich.ols.OLS` works here too; it is applied to the
    projected regressors ``X̂ = P_Z X`` with the structural residuals ``y - Xβ``.
    """

    def __init__(
        self,
        y: ArrayLike,
        exog: ArrayLike,
        endog: ArrayLike,
        instruments: ArrayLike,
        names: list[str] | None = None,
    ):
        self.y = as_1d(y)
        self.exog = as_2d(exog, "exog")
        self.endog = as_2d(endog, "endog")
        self.instruments = as_2d(instruments, "instruments")
        n = self.y.shape[0]
        for arr, label in [
            (self.exog, "exog"),
            (self.endog, "endog"),
            (self.instruments, "instruments"),
        ]:
            if arr.shape[0] != n:
                raise ValueError(f"{label} has {arr.shape[0]} rows but y has {n}")
        if self.instruments.shape[1] < self.endog.shape[1]:
            raise ValueError(
                f"under-identified: {self.endog.shape[1]} endogenous regressors but only "
                f"{self.instruments.shape[1]} excluded instruments"
            )
        exog_names = _names_from(exog, None, self.exog.shape[1], prefix="x")
        endog_names = _names_from(endog, None, self.endog.shape[1], prefix="endog")
        self.instrument_names = _names_from(
            instruments, None, self.instruments.shape[1], prefix="z"
        )
        if not isinstance(exog, pd.DataFrame):
            for j in range(self.exog.shape[1]):
                if np.allclose(self.exog[:, j], 1.0):
                    exog_names[j] = "const"
        self.names = names if names is not None else exog_names + endog_names
        if len(self.names) != self.exog.shape[1] + self.endog.shape[1]:
            raise ValueError("names must cover exog followed by endog columns")
        self.endog_names = self.names[self.exog.shape[1] :]
        self.dep_name = _dep_name(y)
        self.x = np.column_stack([self.exog, self.endog])
        self.z = np.column_stack([self.exog, self.instruments])
        self.has_constant = bool(
            any(np.allclose(self.x[:, j], 1.0) for j in range(self.x.shape[1]))
        )

    def fit(
        self,
        cov_type: str = "nonrobust",
        *,
        groups: ArrayLike | None = None,
        maxlags: int | None = None,
        use_correction: bool | None = None,
        use_t: bool = True,
        debiased: bool = True,
    ) -> RegressionResults:
        """Two-stage least squares.

        ``debiased=True`` uses ``n - k`` in the residual variance (Stata's ``small``);
        ``False`` uses ``n``, the default in Stata's ``ivregress`` and linearmodels.
        """
        n, k = self.x.shape
        q_z, _ = np.linalg.qr(self.z, mode="reduced")
        x_hat = q_z @ (q_z.T @ self.x)
        beta, bread = qr_solve(x_hat, self.y)
        fitted = self.x @ beta
        resid = self.y - fitted
        df_resid = n - k if debiased else n
        cov = compute_covariance(
            cov_type,
            x_hat,
            resid,
            bread,
            df_resid,
            groups=groups,
            maxlags=maxlags,
            use_correction=use_correction,
        )
        centered = self.y - self.y.mean() if self.has_constant else self.y
        res = RegressionResults(
            params=beta,
            cov=cov.cov,
            names=list(self.names),
            resid=resid,
            fitted=fitted,
            nobs=n,
            df_model=k - int(self.has_constant),
            df_resid=n - k,
            df_inference=cov.df_inference if cov.cov_type == "cluster" else n - k,
            cov_type=cov.cov_type,
            cov_description=cov.description,
            rss=float(resid @ resid),
            tss=float(centered @ centered),
            use_t=use_t,
            model="IV-2SLS",
            dep_name=self.dep_name,
            has_constant=self.has_constant,
            info={**cov.info, "instruments": self.instrument_names},
        )
        res.info["diagnostics"] = self.diagnostics(
            res,
            cov_type=cov_type,
            groups=groups,
            maxlags=maxlags,
            use_correction=use_correction,
            use_t=use_t,
        )
        return res

    # -- diagnostics ---------------------------------------------------------------------------
    def diagnostics(self, res: RegressionResults, **fit_kwargs) -> IVDiagnostics:
        n = self.y.shape[0]
        n_exog, n_endog, n_inst = self.exog.shape[1], self.endog.shape[1], self.instruments.shape[1]
        excluded = [f"z{j}" for j in range(n_inst)]
        z_names = [f"x{j}" for j in range(n_exog)] + excluded
        first, v_hat = [], np.empty((n, n_endog))
        for j in range(n_endog):
            fs = OLS(self.endog[:, j], self.z, names=z_names).fit(**fit_kwargs)
            v_hat[:, j] = fs.resid
            f = fs.f_test(excluded)
            restricted = OLS(self.endog[:, j], self.exog).fit()
            partial_r2 = 1.0 - fs.rss / restricted.rss if restricted.rss > 0 else float("nan")
            first.append(FirstStage(self.endog_names[j], f.statistic, f.pvalue, fs.r2, partial_r2))
        # Wu–Hausman: add first-stage residuals to the structural equation and test them jointly
        aug = np.column_stack([self.x, v_hat])
        aug_names = list(self.names) + [f"v_hat_{nm}" for nm in self.endog_names]
        wh = OLS(self.y, aug, names=aug_names).fit(**fit_kwargs).f_test(aug_names[-n_endog:])
        sargan = hansen = None
        if n_inst > n_endog:
            df = n_inst - n_endog
            e = res.resid
            aux = OLS(e, self.z).fit()
            s_stat = n * aux.r2
            sargan = (s_stat, float(stats.chi2.sf(s_stat, df)), df)
            ze = self.z.T @ e
            s_hat = (self.z * (e**2)[:, None]).T @ self.z
            j_stat = float(ze @ np.linalg.solve(s_hat, ze))
            hansen = (j_stat, float(stats.chi2.sf(j_stat, df)), df)
        return IVDiagnostics(first, wh, sargan, hansen)
