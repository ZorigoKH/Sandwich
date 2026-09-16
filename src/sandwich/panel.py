"""Panel data: fixed effects by the within transformation, and first differences.

A fixed-effects model ``y_it = x_it'β + α_i + λ_t + ε_it`` is estimated without ever
building a dummy for each entity or period. One-way effects are removed by demeaning
within entity (or within period); two-way effects by demeaning within the larger
group and then partialling out dummies for the smaller one (Frisch–Waugh–Lovell),
which is exact for unbalanced panels too. The absorbed effects still consume degrees
of freedom, and every covariance estimator here accounts for that the way Stata's
``xtreg, fe`` and linearmodels' ``PanelOLS`` do.

``FirstDifferenceOLS`` regresses ``Δy_it`` on ``Δx_it``, differencing consecutive
periods of the panel's time index only, so an entity missing period ``t-1``
contributes no observation at ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from ._linalg import as_1d, as_2d, qr_solve
from .covariance import HC_TYPES, _factorize, compute_covariance
from .ols import _dep_name, _names_from
from .results import RegressionResults, WaldTestResult

PANEL_COV_TYPES = ("nonrobust", "HC0", "HC1", "cluster")


# -- panel index helpers -----------------------------------------------------------------------
def _index_levels(obj: object) -> tuple[NDArray, NDArray] | None:
    """The two levels of a pandas (entity, time) MultiIndex, if ``obj`` carries one."""
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        idx = obj.index
        if isinstance(idx, pd.MultiIndex) and idx.nlevels == 2:
            return idx.get_level_values(0).to_numpy(), idx.get_level_values(1).to_numpy()
    return None


def _resolve_index(
    y: ArrayLike, x: ArrayLike, entity: ArrayLike | None, time: ArrayLike | None, n: int
) -> tuple[NDArray | None, NDArray | None]:
    if entity is None and time is None:
        levels = _index_levels(y) or _index_levels(x)
        if levels is not None:
            entity, time = levels
    out = []
    for arr, label in [(entity, "entity"), (time, "time")]:
        if arr is None:
            out.append(None)
            continue
        arr = np.asarray(arr)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        if arr.shape != (n,):
            raise ValueError(f"{label} must have length {n}, got shape {arr.shape}")
        out.append(arr)
    return out[0], out[1]


def _group_means(a: NDArray[np.float64], codes: NDArray[np.intp], n_groups: int) -> NDArray:
    """Per-group means of the rows of ``a`` (1-D or 2-D), as a ``(n_groups, ...)`` array."""
    counts = np.bincount(codes, minlength=n_groups).astype(np.float64)
    if a.ndim == 1:
        return np.bincount(codes, weights=a, minlength=n_groups) / counts
    sums = np.zeros((n_groups, a.shape[1]))
    np.add.at(sums, codes, a)
    return sums / counts[:, None]


def _demean(a: NDArray[np.float64], codes: NDArray[np.intp], n_groups: int) -> NDArray:
    """Subtract the group mean from every row: the within transformation."""
    return a - _group_means(a, codes, n_groups)[codes]


def _is_nested(effect: NDArray[np.intp], clusters: NDArray) -> bool:
    """True when every level of ``effect`` lies inside a single cluster."""
    cluster_codes, uniques = _factorize(np.asarray(clusters))
    pairs = effect.astype(np.int64) * len(uniques) + cluster_codes
    return len(np.unique(pairs)) == int(effect.max()) + 1


# -- fixed effects -----------------------------------------------------------------------------
class PanelOLS:
    """``PanelOLS(y, X, entity, time, entity_effects=True, time_effects=False).fit()``.

    ``entity`` and ``time`` are arrays of labels, one per row; they can be left out when
    ``y`` (or ``X``) is a pandas object indexed by an ``(entity, time)`` MultiIndex.
    ``time`` is only needed for time effects or time clusters. A constant column in
    ``X`` is kept and reported the way ``xtreg, fe`` reports ``_cons``: the grand mean
    of ``y`` net of the slopes. Columns that do not vary within the absorbed groups
    are rejected rather than silently dropped.

    ``fit`` accepts ``cov_type`` "nonrobust", "HC0", "HC1" or "cluster"; ``groups`` may
    be an array of labels, ``"entity"``, ``"time"``, or a pair of those for two-way
    clustering. HC2/HC3 and HAC are not offered because their leverage and time
    ordering are not defined for absorbed effects.
    """

    def __init__(
        self,
        y: ArrayLike,
        x: ArrayLike,
        entity: ArrayLike | None = None,
        time: ArrayLike | None = None,
        *,
        entity_effects: bool = True,
        time_effects: bool = False,
        names: list[str] | None = None,
    ):
        self.y = as_1d(y)
        self.x = as_2d(x)
        n = self.y.shape[0]
        if self.x.shape[0] != n:
            raise ValueError(f"y has {n} rows but X has {self.x.shape[0]}")
        if not (entity_effects or time_effects):
            raise ValueError(
                "choose entity_effects=True and/or time_effects=True; "
                "for pooled OLS use sandwich.OLS"
            )
        entity_arr, time_arr = _resolve_index(y, x, entity, time, n)
        if entity_arr is None:
            raise ValueError("entity= is required (or index y by an (entity, time) MultiIndex)")
        if time_arr is None and time_effects:
            raise ValueError("time= is required for time_effects=True")
        self.entity_codes, self.entities = _factorize(entity_arr)
        self.n_entities = len(self.entities)
        if time_arr is not None:
            self.time_codes, self.periods = _factorize(time_arr)
            self.n_periods: int | None = len(self.periods)
        else:
            self.time_codes, self.periods, self.n_periods = None, None, None
        self.entity_effects = bool(entity_effects)
        self.time_effects = bool(time_effects)
        self.names = _names_from(x, names, self.x.shape[1])
        self.dep_name = _dep_name(y)
        const_cols = [j for j in range(self.x.shape[1]) if np.allclose(self.x[:, j], 1.0)]
        self.has_constant = bool(const_cols)
        if self.has_constant and names is None and not isinstance(x, pd.DataFrame):
            self.names[const_cols[0]] = "const"
        self._const_cols = const_cols

    # -- the within transformation ---------------------------------------------------------------
    def _effect_groups(self) -> list[tuple[str, NDArray[np.intp], int]]:
        groups = []
        if self.entity_effects:
            groups.append(("entity", self.entity_codes, self.n_entities))
        if self.time_effects:
            assert self.time_codes is not None and self.n_periods is not None
            groups.append(("time", self.time_codes, self.n_periods))
        return groups

    def _transform(self, *arrays: NDArray[np.float64]) -> list[NDArray[np.float64]]:
        """Remove the fixed effects from each array (1-D or 2-D) by the same projection."""
        groups = self._effect_groups()
        if len(groups) == 1:
            _, codes, n_groups = groups[0]
            return [_demean(a, codes, n_groups) for a in arrays]
        # two-way: demean within the larger group, partial out dummies for the smaller one
        (_, big_codes, n_big), (_, small_codes, n_small) = sorted(groups, key=lambda g: -g[2])
        dummies = np.eye(n_small)[small_codes][:, 1:]  # drop the first level
        d_tilde = _demean(dummies, big_codes, n_big)
        out = []
        for a in arrays:
            a_tilde = _demean(a, big_codes, n_big)
            coef = np.linalg.lstsq(d_tilde, a_tilde, rcond=None)[0]
            out.append(a_tilde - d_tilde @ coef)
        return out

    def _absorbed_df(self) -> dict[str, int]:
        """Degrees of freedom consumed by each set of effects (one level is redundant
        with a constant, and the second set of effects is redundant with the first)."""
        out = {}
        drop_first = self.has_constant
        for name, _, n_groups in self._effect_groups():
            out[name] = n_groups - int(drop_first)
            drop_first = True
        return out

    def _resolve_groups(self, groups: ArrayLike | str | None) -> NDArray | None:
        if groups is None:
            return None
        if isinstance(groups, (tuple, list)) and len(groups) == 2 and np.ndim(groups[0]) <= 1:
            cols = [self._resolve_groups(g) for g in groups]
            if all(np.ndim(c) == 1 for c in cols):
                return np.column_stack(cols)
        if isinstance(groups, str):
            if groups == "entity":
                return self.entity_codes
            if groups == "time":
                if self.time_codes is None:
                    raise ValueError("groups='time' needs time= at construction")
                return self.time_codes
            raise ValueError("groups must be 'entity', 'time', a pair of those, or an array")
        g = np.asarray(groups)
        if g.shape[0] != self.y.shape[0]:
            raise ValueError(f"groups must have {self.y.shape[0]} rows, got {g.shape[0]}")
        return g

    def _extra_df(self, cov_type: str, groups: NDArray | None, count_effects: bool | None) -> int:
        """Absorbed degrees of freedom the covariance should charge for.

        By default every effect is counted, except one nested inside the clusters of a
        cluster-robust estimator: the cluster sums already absorb it, so charging for it
        again would double count (linearmodels' ``auto_df``, Stata's ``xtreg, fe``).
        """
        absorbed = self._absorbed_df()
        if count_effects is True or (count_effects is None and cov_type != "cluster"):
            return sum(absorbed.values())
        if count_effects is False:
            return 0
        if groups is None:
            return sum(absorbed.values())
        cols = [groups] if groups.ndim == 1 else [groups[:, j] for j in range(groups.shape[1])]
        total = 0
        for name, codes, _ in self._effect_groups():
            if not any(_is_nested(codes, c) for c in cols):
                total += absorbed[name]
        return total

    # -- estimation ---------------------------------------------------------------------------
    def fit(
        self,
        cov_type: str = "nonrobust",
        *,
        groups: ArrayLike | str | None = None,
        use_correction: bool | None = None,
        use_t: bool = True,
        count_effects: bool | None = None,
    ) -> RegressionResults:
        """Estimate by the within transformation and attach a covariance matrix.

        Parameters
        ----------
        cov_type : "nonrobust", "HC0", "HC1" or "cluster".
        groups : cluster labels, ``"entity"``, ``"time"`` or a pair for two-way clusters.
        use_correction : override the cluster estimator's small-sample scaling
            (default ``G/(G-1) · (n-1)/df``, Stata's; ``False`` for none).
        use_t : ``t``/``F`` inference with ``df_inference`` (default) or ``z``/``chi2``.
        count_effects : whether the covariance's degrees of freedom charge for the
            absorbed effects. ``None`` charges for every effect not nested in the clusters.
        """
        ct = cov_type.upper() if cov_type.upper() in HC_TYPES else cov_type.lower()
        if ct not in PANEL_COV_TYPES:
            raise ValueError(
                f"cov_type {cov_type!r} is not available with absorbed effects; "
                f"choose from {PANEL_COV_TYPES}"
            )
        n, k = self.x.shape
        y_t, x_t = self._transform(self.y, self.x)
        self._check_absorbed(x_t)
        if self.has_constant:  # add the grand means back so the constant is identified
            y_t = y_t + self.y.mean()
            x_t = x_t + self.x.mean(axis=0)
        beta, xtx_inv = qr_solve(x_t, y_t)
        resid = y_t - x_t @ beta
        absorbed = self._absorbed_df()
        n_absorbed = sum(absorbed.values())
        df_resid = n - k - n_absorbed
        if df_resid <= 0:
            raise ValueError(f"no residual degrees of freedom: n={n}, k={k}, absorbed={n_absorbed}")
        group_arr = self._resolve_groups(groups)
        extra_df = self._extra_df(ct, group_arr, count_effects)
        cov = compute_covariance(
            ct,
            x_t,
            resid,
            xtx_inv,
            n - k - extra_df,
            groups=group_arr,
            use_correction=use_correction,
        )
        fitted = self.x @ beta
        effects = (self.y - fitted) - resid
        rss = float(resid @ resid)
        y_centered = y_t - self.y.mean() if self.has_constant else y_t
        tss = float(y_centered @ y_centered)
        r2_within, r2_between, r2_overall = self._r2_variants(beta)
        info = {
            "effects": " + ".join(
                f"{name} ({n_groups} levels)" for name, _, n_groups in self._effect_groups()
            ),
            "absorbed_df": n_absorbed,
            "n_entities": self.n_entities,
            "n_periods": self.n_periods,
            "r2_within": r2_within,
            "r2_between": r2_between,
            "r2_overall": r2_overall,
            "f_effects": self._f_effects(rss, n_absorbed, df_resid),
            **cov.info,
        }
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
            rss=rss,
            tss=tss,
            use_t=use_t,
            model="PanelOLS",
            dep_name=self.dep_name,
            has_constant=self.has_constant,
            info=info,
            effects=effects,
        )

    def _check_absorbed(self, x_t: NDArray[np.float64]) -> None:
        scale = np.maximum(np.linalg.norm(self.x, axis=0), 1.0)
        gone = np.linalg.norm(x_t, axis=0) <= 1e-12 * scale
        gone[self._const_cols] = False
        if gone.any():
            cols = ", ".join(self.names[j] for j in np.flatnonzero(gone))
            raise ValueError(
                f"{cols}: absorbed by the fixed effects (no variation within groups); "
                "drop the column, or use fewer effects"
            )

    # -- fit statistics ----------------------------------------------------------------------
    def _r2_variants(self, beta: NDArray[np.float64]) -> tuple[float, float, float]:
        """Within, between and overall R² as linearmodels and Stata define them."""

        def r2(y: NDArray, x: NDArray, center: bool) -> float:
            e = y - x @ beta
            dev = y - y.mean() if center else y
            tss = float(dev @ dev)
            return 1.0 - float(e @ e) / tss if tss > 0 else 0.0

        if self.has_constant and self.x.shape[1] == 1:
            return 0.0, 0.0, 0.0
        y_bar = _group_means(self.y, self.entity_codes, self.n_entities)
        x_bar = _group_means(self.x, self.entity_codes, self.n_entities)
        between = r2(y_bar, x_bar, self.has_constant)
        overall = r2(self.y, self.x, self.has_constant)
        within = r2(
            _demean(self.y, self.entity_codes, self.n_entities),
            _demean(self.x, self.entity_codes, self.n_entities),
            False,
        )
        return within, between, overall

    def _f_effects(self, rss: float, n_absorbed: int, df_resid: int) -> WaldTestResult:
        """F test that all absorbed effects are zero, against pooled OLS."""
        y, x = self.y, self.x
        df_num = n_absorbed
        if not self.has_constant:  # pooled OLS gets a constant the effects otherwise supply
            y, x = y - y.mean(), x - x.mean(axis=0)
            df_num -= 1
        coef = np.linalg.lstsq(x, y, rcond=None)[0]
        e = y - x @ coef
        rss_pooled = float(e @ e)
        stat = ((rss_pooled - rss) / df_num) / (rss / df_resid)
        return WaldTestResult(
            stat, float(stats.f.sf(stat, df_num, df_resid)), df_num, df_resid, "F"
        )


# -- first differences -------------------------------------------------------------------------
class FirstDifferenceOLS:
    """``FirstDifferenceOLS(y, X, entity, time).fit()``: OLS of ``Δy`` on ``ΔX``.

    Differences are taken between consecutive periods of the panel's time index within
    each entity; gaps produce no observation. A constant in ``X`` is rejected (it would
    difference to zero — put a time trend in ``X`` if you want a drift term). All of the
    OLS covariance types are available except HAC.
    """

    def __init__(
        self,
        y: ArrayLike,
        x: ArrayLike,
        entity: ArrayLike | None = None,
        time: ArrayLike | None = None,
        names: list[str] | None = None,
    ):
        self.y = as_1d(y)
        self.x = as_2d(x)
        n = self.y.shape[0]
        if self.x.shape[0] != n:
            raise ValueError(f"y has {n} rows but X has {self.x.shape[0]}")
        entity_arr, time_arr = _resolve_index(y, x, entity, time, n)
        if entity_arr is None or time_arr is None:
            raise ValueError(
                "entity= and time= are required (or index y by an (entity, time) MultiIndex)"
            )
        if any(np.allclose(self.x[:, j], 1.0) for j in range(self.x.shape[1])):
            raise ValueError("a constant differences to zero; drop it (or add a time trend)")
        self.entity_codes, self.entities = _factorize(entity_arr)
        self.time_codes, self.periods = _factorize(time_arr)
        self.n_entities, self.n_periods = len(self.entities), len(self.periods)
        if self.n_periods < 2:
            raise ValueError("first differences need at least two time periods")
        self.names = _names_from(x, names, self.x.shape[1])
        self.dep_name = _dep_name(y)
        order = np.lexsort((self.time_codes, self.entity_codes))
        ent, tim = self.entity_codes[order], self.time_codes[order]
        adjacent = (ent[1:] == ent[:-1]) & (tim[1:] == tim[:-1] + 1)
        self.current = order[1:][adjacent]
        self.previous = order[:-1][adjacent]
        if self.current.size == 0:
            raise ValueError("no entity is observed in two consecutive periods")
        self.dy = self.y[self.current] - self.y[self.previous]
        self.dx = self.x[self.current] - self.x[self.previous]

    def _resolve_groups(self, groups: ArrayLike | str | None) -> NDArray | None:
        if groups is None:
            return None
        if isinstance(groups, str):
            if groups == "entity":
                return self.entity_codes[self.current]
            raise ValueError("groups must be 'entity' or an array constant within each entity")
        g = np.asarray(groups)
        if g.shape[0] != self.y.shape[0]:
            raise ValueError(f"groups must have {self.y.shape[0]} rows, got {g.shape[0]}")
        if not np.array_equal(g[self.current], g[self.previous]):
            raise ValueError("cluster labels must be identical within each differenced pair")
        return g[self.current]

    def fit(
        self,
        cov_type: str = "nonrobust",
        *,
        groups: ArrayLike | str | None = None,
        use_correction: bool | None = None,
        use_t: bool = True,
    ) -> RegressionResults:
        if cov_type.lower() == "hac":
            raise ValueError("HAC is not defined for stacked panel differences")
        n_d, k = self.dx.shape
        beta, xtx_inv = qr_solve(self.dx, self.dy)
        resid = self.dy - self.dx @ beta
        df_resid = n_d - k
        cov = compute_covariance(
            cov_type,
            self.dx,
            resid,
            xtx_inv,
            df_resid,
            groups=self._resolve_groups(groups),
            use_correction=use_correction,
        )
        return RegressionResults(
            params=beta,
            cov=cov.cov,
            names=list(self.names),
            resid=resid,
            fitted=self.dx @ beta,
            nobs=n_d,
            df_model=k,
            df_resid=df_resid,
            df_inference=cov.df_inference,
            cov_type=cov.cov_type,
            cov_description=cov.description,
            rss=float(resid @ resid),
            tss=float(self.dy @ self.dy),
            use_t=use_t,
            model="FirstDifferenceOLS",
            dep_name=f"Δ{self.dep_name}",
            has_constant=False,
            info={
                "n_entities": self.n_entities,
                "n_periods": self.n_periods,
                "n_levels_obs": int(self.y.shape[0]),
                **cov.info,
            },
        )
