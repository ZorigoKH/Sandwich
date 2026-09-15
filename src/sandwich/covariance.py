"""Covariance estimators for least-squares coefficients.

Every estimator here has the same shape: ``bread @ meat @ bread`` where the bread is
``(X'X)^{-1}`` and the meat is an estimate of ``Var(X'e)``. What changes is only the
meat — hence the package name.

===========  =====================================================  ==============
cov_type     meat                                                    df for t / F
===========  =====================================================  ==============
nonrobust    ``s² X'X``  with  ``s² = e'e / (n - k)``                 ``n - k``
HC0          ``X' diag(e²) X``                                       ``n - k``
HC1          ``n/(n-k) · HC0``                                       ``n - k``
HC2          ``X' diag(e² / (1 - h)) X``                             ``n - k``
HC3          ``X' diag(e² / (1 - h)²) X``                            ``n - k``
cluster      ``Σ_g (X_g'e_g)(X_g'e_g)'`` · small-sample factor       ``G - 1``
HAC          Newey–West: ``Γ₀ + Σ_l w_l (Γ_l + Γ_l')``                 ``n - k``
===========  =====================================================  ==============

The cluster small-sample factor is ``G/(G-1) · (n-1)/(n-k)``, the Stata / statsmodels
convention. Two-way clustering follows Cameron, Gelbach & Miller (2011):
``V = V_g1 + V_g2 - V_(g1∩g2)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._linalg import leverage, sandwich

HC_TYPES = ("HC0", "HC1", "HC2", "HC3")
COV_TYPES = ("nonrobust", *HC_TYPES, "cluster", "HAC")


@dataclass
class CovarianceResult:
    """A coefficient covariance matrix plus what inference should know about it."""

    cov: NDArray[np.float64]
    cov_type: str
    df_inference: int
    description: str
    info: dict = field(default_factory=dict)


def _check_shapes(x: NDArray[np.float64], resid: NDArray[np.float64]) -> tuple[int, int]:
    n, k = x.shape
    if resid.shape != (n,):
        raise ValueError(f"resid must have shape ({n},), got {resid.shape}")
    return n, k


def cov_nonrobust(
    x: NDArray[np.float64], resid: NDArray[np.float64], xtx_inv: NDArray[np.float64], df_resid: int
) -> CovarianceResult:
    """Homoskedastic covariance ``s² (X'X)^{-1}``."""
    _check_shapes(x, resid)
    sigma2 = float(resid @ resid) / df_resid
    return CovarianceResult(
        cov=sigma2 * xtx_inv,
        cov_type="nonrobust",
        df_inference=df_resid,
        description="homoskedastic (classical)",
        info={"sigma2": sigma2},
    )


def cov_hc(
    x: NDArray[np.float64],
    resid: NDArray[np.float64],
    xtx_inv: NDArray[np.float64],
    df_resid: int,
    kind: str = "HC1",
) -> CovarianceResult:
    """Heteroskedasticity-consistent covariance (White / MacKinnon–White HC0–HC3)."""
    n, k = _check_shapes(x, resid)
    kind = kind.upper()
    if kind not in HC_TYPES:
        raise ValueError(f"kind must be one of {HC_TYPES}, got {kind!r}")
    e2 = resid**2
    scale = 1.0
    if kind == "HC1":
        scale = n / df_resid
    elif kind == "HC2":
        e2 = e2 / (1.0 - leverage(x, xtx_inv))
    elif kind == "HC3":
        e2 = e2 / (1.0 - leverage(x, xtx_inv)) ** 2
    meat = (x * e2[:, None]).T @ x
    return CovarianceResult(
        cov=sandwich(xtx_inv, meat, scale),
        cov_type=kind,
        df_inference=df_resid,
        description=f"heteroskedasticity-robust ({kind})",
    )


def _group_index(groups: ArrayLike) -> tuple[NDArray[np.intp], int]:
    codes, uniques = _factorize(np.asarray(groups))
    return codes, len(uniques)


def _factorize(values: NDArray) -> tuple[NDArray[np.intp], NDArray]:
    uniques, codes = np.unique(values, return_inverse=True)
    return codes.astype(np.intp), uniques


def _cluster_meat(
    scores: NDArray[np.float64], codes: NDArray[np.intp], n_groups: int
) -> NDArray[np.float64]:
    """``Σ_g s_g s_g'`` where ``s_g = Σ_{i in g} x_i e_i`` (the within-cluster score sums)."""
    k = scores.shape[1]
    sums = np.zeros((n_groups, k))
    np.add.at(sums, codes, scores)
    return sums.T @ sums


def cov_cluster(
    x: NDArray[np.float64],
    resid: NDArray[np.float64],
    xtx_inv: NDArray[np.float64],
    groups: ArrayLike,
    df_resid: int,
    use_correction: bool = True,
) -> CovarianceResult:
    """Cluster-robust covariance (Liang–Zeger), one-way or two-way.

    ``groups`` is an array of cluster labels of length ``n``, or a 2-D array / tuple of
    two such arrays for two-way clustering.
    """
    n, k = _check_shapes(x, resid)
    scores = x * resid[:, None]
    g = np.asarray(groups)
    if isinstance(groups, (tuple, list)) and len(groups) == 2 and np.ndim(groups[0]) == 1:
        g = np.column_stack([np.asarray(groups[0]), np.asarray(groups[1])])
    if g.ndim == 2 and g.shape[1] == 1:
        g = g[:, 0]
    if g.shape[0] != n:
        raise ValueError(f"groups must have {n} rows, got {g.shape[0]}")

    def one_way(labels: NDArray) -> tuple[NDArray[np.float64], int]:
        codes, n_groups = _group_index(labels)
        meat = _cluster_meat(scores, codes, n_groups)
        scale = 1.0
        if use_correction:
            scale = (n_groups / (n_groups - 1.0)) * ((n - 1.0) / df_resid)
        return sandwich(xtx_inv, meat, scale), n_groups

    if g.ndim == 1:
        cov, n_groups = one_way(g)
        return CovarianceResult(
            cov=cov,
            cov_type="cluster",
            df_inference=n_groups - 1,
            description=f"cluster-robust ({n_groups} clusters)",
            info={"n_clusters": n_groups},
        )
    if g.ndim == 2 and g.shape[1] == 2:
        v1, g1 = one_way(g[:, 0])
        v2, g2 = one_way(g[:, 1])
        # the intersection cluster: a unique code per (g1, g2) pair
        pair = np.asarray([f"{a}\x1f{b}" for a, b in zip(g[:, 0], g[:, 1], strict=True)])
        v12, g12 = one_way(pair)
        cov = v1 + v2 - v12
        cov = (cov + cov.T) / 2.0
        return CovarianceResult(
            cov=cov,
            cov_type="cluster",
            df_inference=min(g1, g2) - 1,
            description=f"two-way cluster-robust ({g1} x {g2} clusters)",
            info={"n_clusters": (g1, g2), "n_intersection_clusters": g12},
        )
    raise ValueError("groups must be 1-D (one-way) or have two columns (two-way)")


def bartlett_weights(maxlags: int) -> NDArray[np.float64]:
    """Newey–West (Bartlett) kernel weights ``1 - l/(L+1)`` for ``l = 1..L``."""
    lags = np.arange(1, maxlags + 1, dtype=np.float64)
    return 1.0 - lags / (maxlags + 1.0)


def cov_hac(
    x: NDArray[np.float64],
    resid: NDArray[np.float64],
    xtx_inv: NDArray[np.float64],
    df_resid: int,
    maxlags: int,
    use_correction: bool = False,
) -> CovarianceResult:
    """Heteroskedasticity- and autocorrelation-consistent covariance (Newey–West 1987).

    Observations must be in time order. ``use_correction=True`` scales by ``n/(n-k)``
    (statsmodels' ``cov_hac_simple`` default is off when called through ``fit``).
    """
    n, k = _check_shapes(x, resid)
    if maxlags < 0:
        raise ValueError("maxlags must be non-negative")
    scores = x * resid[:, None]
    meat = scores.T @ scores
    for lag, w in enumerate(bartlett_weights(maxlags), start=1):
        gamma = scores[lag:].T @ scores[:-lag]
        meat += w * (gamma + gamma.T)
    scale = n / df_resid if use_correction else 1.0
    return CovarianceResult(
        cov=sandwich(xtx_inv, meat, scale),
        cov_type="HAC",
        df_inference=df_resid,
        description=f"HAC (Newey-West, Bartlett kernel, {maxlags} lags)",
        info={"maxlags": maxlags},
    )


def compute_covariance(
    cov_type: str,
    x: NDArray[np.float64],
    resid: NDArray[np.float64],
    xtx_inv: NDArray[np.float64],
    df_resid: int,
    *,
    groups: ArrayLike | None = None,
    maxlags: int | None = None,
    use_correction: bool | None = None,
) -> CovarianceResult:
    """Dispatch on ``cov_type`` (case-insensitive) to the estimator above."""
    ct = cov_type.upper() if cov_type.upper() in HC_TYPES else cov_type.lower()
    if ct == "nonrobust":
        return cov_nonrobust(x, resid, xtx_inv, df_resid)
    if ct in HC_TYPES:
        return cov_hc(x, resid, xtx_inv, df_resid, kind=ct)
    if ct == "cluster":
        if groups is None:
            raise ValueError("cov_type='cluster' requires groups=")
        return cov_cluster(
            x,
            resid,
            xtx_inv,
            groups,
            df_resid,
            use_correction=True if use_correction is None else use_correction,
        )
    if ct == "hac":
        if maxlags is None:
            raise ValueError("cov_type='HAC' requires maxlags=")
        return cov_hac(
            x,
            resid,
            xtx_inv,
            df_resid,
            maxlags,
            use_correction=False if use_correction is None else use_correction,
        )
    raise ValueError(f"unknown cov_type {cov_type!r}; choose from {COV_TYPES}")
