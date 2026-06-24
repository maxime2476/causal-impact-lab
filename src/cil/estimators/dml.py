"""Double/debiased ML heterogeneity on the panel (EconML).

Estimates the average and exposure-heterogeneous effect of the exposure-shock
interaction on employment growth, using EconML's LinearDML and CausalForestDML
with **purged time-blocked cross-fitting** (:class:`PurgedTimeBlockedCV`). A
random K-fold here would leak serial correlation across folds and is a
correctness bug; the time-blocked splitter is mandatory. A placebo refutation
(treatment permuted across time) checks the estimate collapses toward zero.

References
----------
Chernozhukov et al. (2018), Econometrics Journal 21(1); EconML.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import polars as pl
from econml.dml import CausalForestDML, LinearDML
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor

from cil.estimators.time_blocked_cv import PurgedTimeBlockedCV

FloatArray = npt.NDArray[np.float64]


class DMLResult(BaseModel):
    """Heterogeneity estimates at one horizon.

    Parameters
    ----------
    horizon
        Response horizon.
    n_obs
        Estimation sample size.
    linear_ate
        LinearDML average treatment effect.
    linear_ate_ci_low, linear_ate_ci_high
        LinearDML ATE confidence interval.
    linear_cate_slope
        LinearDML CATE slope in exposure (effect heterogeneity).
    forest_ate
        CausalForestDML average treatment effect.
    placebo_ate
        ATE under a time-permuted (placebo) treatment; should be near zero.
    """

    horizon: int
    n_obs: int
    linear_ate: float
    linear_ate_ci_low: float
    linear_ate_ci_high: float
    linear_cate_slope: float
    forest_ate: float
    placebo_ate: float


def _nuisance() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=100, min_samples_leaf=20, random_state=0, n_jobs=-1
    )


def build_sample(
    panel: pl.DataFrame,
    exposure: pl.DataFrame,
    shock: pl.DataFrame,
    *,
    shock_col: str,
    horizon: int,
    n_lags: int = 6,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, npt.NDArray[np.int_]]:
    """Build the DML arrays ``(Y, T, X, W, time_codes)`` for one horizon.

    Parameters
    ----------
    panel
        Cell panel (``unit_id``, ``supersector_code``, ``date``,
        ``log_employment``).
    exposure
        Cell exposure (``supersector_code``, ``exposure``).
    shock
        Shock frame (``date``, shock column).
    shock_col
        Name of the shock column.
    horizon
        Response horizon ``h``.
    n_lags
        Number of lagged employment-change controls.

    Returns
    -------
    Y, T, X, W : numpy.ndarray
        Outcome, treatment, effect modifier, and controls.
    time_codes : numpy.ndarray
        Integer period code per row for the splitter.
    """
    base = (
        panel.join(exposure, on="supersector_code", how="inner")
        .join(shock.select("date", shock_col), on="date", how="inner")
        .sort(["unit_id", "date"])
        .with_columns(treatment=pl.col("exposure") * pl.col(shock_col))
    )
    lag_exprs = [
        (pl.col("log_employment").shift(lag) - pl.col("log_employment").shift(lag + 1))
        .over("unit_id")
        .alias(f"dy_l{lag}")
        for lag in range(1, n_lags + 1)
    ]
    controls = [f"dy_l{lag}" for lag in range(1, n_lags + 1)]
    frame = (
        base.with_columns(lag_exprs)
        .with_columns(
            target=(
                pl.col("log_employment").shift(-horizon)
                - pl.col("log_employment").shift(1)
            ).over("unit_id"),
            period=pl.col("date").rank("dense"),
        )
        .select(["target", "treatment", "exposure", "period", *controls])
        .drop_nulls()
    )
    y = frame["target"].to_numpy().astype(np.float64)
    t = frame["treatment"].to_numpy().astype(np.float64)
    x = frame["exposure"].to_numpy().astype(np.float64).reshape(-1, 1)
    w = frame.select(controls).to_numpy().astype(np.float64)
    time_codes = frame["period"].to_numpy().astype(int)
    return y, t, x, w, time_codes


def estimate_heterogeneity(
    panel: pl.DataFrame,
    exposure: pl.DataFrame,
    shock: pl.DataFrame,
    *,
    shock_col: str,
    horizon: int,
    n_lags: int = 6,
    n_splits: int = 5,
    embargo: int = 6,
    seed: int = 20260101,
) -> DMLResult:
    """Estimate LinearDML and CausalForestDML CATEs with a placebo check.

    Parameters
    ----------
    panel, exposure, shock
        Inputs as in :func:`build_sample`.
    shock_col
        Name of the shock column.
    horizon
        Response horizon.
    n_lags
        Number of control lags.
    n_splits, embargo
        Time-blocked cross-fitting parameters.
    seed
        Random seed (used for the placebo permutation).

    Returns
    -------
    DMLResult
        Average and heterogeneous effects with a placebo ATE.
    """
    y, t, x, w, time_codes = build_sample(
        panel, exposure, shock, shock_col=shock_col, horizon=horizon, n_lags=n_lags
    )
    cv = PurgedTimeBlockedCV(time_codes, n_splits=n_splits, embargo=embargo)
    linear = LinearDML(
        model_y=_nuisance(), model_t=_nuisance(), cv=cv, random_state=seed
    )
    linear.fit(y, t, X=x, W=w)
    ate = float(linear.ate(x))
    ate_ci = linear.ate_interval(x)
    slope = float(np.asarray(linear.coef_).ravel()[0])

    cv_forest = PurgedTimeBlockedCV(time_codes, n_splits=n_splits, embargo=embargo)
    forest = CausalForestDML(
        model_y=_nuisance(), model_t=_nuisance(), cv=cv_forest, random_state=seed
    )
    forest.fit(y, t, X=x, W=w)
    forest_ate = float(forest.ate(x))

    rng = np.random.default_rng(seed)
    t_placebo = rng.permutation(t)
    cv_placebo = PurgedTimeBlockedCV(time_codes, n_splits=n_splits, embargo=embargo)
    placebo = LinearDML(
        model_y=_nuisance(), model_t=_nuisance(), cv=cv_placebo, random_state=seed
    )
    placebo.fit(y, t_placebo, X=x, W=w)
    placebo_ate = float(placebo.ate(x))

    return DMLResult(
        horizon=horizon,
        n_obs=int(y.size),
        linear_ate=ate,
        linear_ate_ci_low=float(ate_ci[0]),
        linear_ate_ci_high=float(ate_ci[1]),
        linear_cate_slope=slope,
        forest_ate=forest_ate,
        placebo_ate=placebo_ate,
    )
