"""Numerical building blocks: least squares via QR, bread matrices, leverage."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


class RankDeficientError(ValueError):
    """Raised when the design matrix does not have full column rank."""


def as_2d(x: ArrayLike, name: str = "X") -> NDArray[np.float64]:
    """Coerce to a 2-D float array (n, k); a 1-D input becomes a single column."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got {arr.ndim}-D")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or inf; drop or impute missing values first")
    return arr


def as_1d(y: ArrayLike, name: str = "y") -> NDArray[np.float64]:
    """Coerce to a 1-D float array (n,)."""
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or inf; drop or impute missing values first")
    return arr


def qr_solve(
    x: NDArray[np.float64], y: NDArray[np.float64], rcond: float = 1e-10
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Solve the least-squares problem min ||y - Xb|| via a thin QR decomposition.

    Returns ``(beta, xtx_inv)`` where ``xtx_inv = (X'X)^{-1}`` is computed from the
    triangular factor ``R`` as ``R^{-1} R^{-T}``, which is far better conditioned than
    inverting ``X'X`` directly (the condition number is that of ``X``, not its square).

    Raises :class:`RankDeficientError` when a diagonal element of ``R`` is (relatively)
    zero, i.e. a column of ``X`` is a linear combination of the others.
    """
    n, k = x.shape
    if n < k:
        raise RankDeficientError(f"more regressors ({k}) than observations ({n})")
    q, r = np.linalg.qr(x, mode="reduced")
    diag = np.abs(np.diag(r))
    if diag.min() <= rcond * diag.max():
        bad = int(np.argmin(diag))
        raise RankDeficientError(
            f"design matrix is rank deficient (column {bad} is collinear with the others)"
        )
    beta = np.linalg.solve(r, q.T @ y)
    r_inv = np.linalg.solve(r, np.eye(k))
    xtx_inv = r_inv @ r_inv.T
    return beta, xtx_inv


def leverage(x: NDArray[np.float64], xtx_inv: NDArray[np.float64]) -> NDArray[np.float64]:
    """Diagonal of the hat matrix ``H = X (X'X)^{-1} X'`` without forming ``H``."""
    return np.einsum("ij,jk,ik->i", x, xtx_inv, x)


def sandwich(
    bread: NDArray[np.float64], meat: NDArray[np.float64], scale: float = 1.0
) -> NDArray[np.float64]:
    """The sandwich ``scale * bread @ meat @ bread``, symmetrised against round-off."""
    v = scale * (bread @ meat @ bread)
    return (v + v.T) / 2.0
