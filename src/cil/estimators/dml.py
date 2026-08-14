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


class CateDriverResult(BaseModel):
    """CausalForest heterogeneity-driver decomposition at one horizon.

    The driver strength is the **Best Linear Predictor (BLP)** of the estimated
    CATE on the *standardised* effect modifiers, not the forest's impurity
    ``feature_importances_``. Impurity importances are biased toward
    high-cardinality features (continuous covariates offer more split points than
    a sector shifter with a handful of values), so they are unreliable as driver
    rankings; the standardised BLP coefficients are comparable across features and
    sign-informative.

    Parameters
    ----------
    horizon
        Response horizon.
    n_obs
        Estimation sample size.
    features
        Effect-modifier names (columns of ``X``).
    blp_coef
        BLP coefficient of the CATE on each standardised feature (signed; the
        CATE change per one-SD move in the feature), aligned to ``features``.
    importances
        Normalised ``|blp_coef|`` (sums to 1) -- relative driver strength.
    forest_ate
        CausalForestDML average treatment effect on this multi-feature design.
    cate_top_exposure_tercile, cate_bottom_exposure_tercile
        Mean CATE in the top vs bottom exposure tercile (which cells respond
        most).
    """

    horizon: int
    n_obs: int
    features: list[str]
    blp_coef: list[float]
    importances: list[float]
    forest_ate: float
    cate_top_exposure_tercile: float
    cate_bottom_exposure_tercile: float


def _cell_base_features(panel: pl.DataFrame, *, base_months: int = 12) -> pl.DataFrame:
    """Predetermined per-cell effect modifiers from the base period.

    Returns ``unit_id``, ``base_share`` (cell employment share within its state)
    and ``log_base_emp`` (cell size), averaged over the earliest *base_months*.
    """
    base_dates = sorted(panel["date"].unique().to_list())[:base_months]
    base = (
        panel.filter(pl.col("date").is_in(base_dates))
        .group_by(["unit_id", "state_fips"])
        .agg(base_emp=pl.col("employment").mean())
        .filter(pl.col("base_emp") > 0)
    )
    return base.with_columns(
        base_share=pl.col("base_emp") / pl.col("base_emp").sum().over("state_fips"),
        log_base_emp=pl.col("base_emp").log(),
    ).select("unit_id", "base_share", "log_base_emp")


def build_driver_sample(
    panel: pl.DataFrame,
    exposure: pl.DataFrame,
    shock: pl.DataFrame,
    *,
    shock_col: str,
    horizon: int,
    n_lags: int = 6,
) -> tuple[
    FloatArray, FloatArray, FloatArray, FloatArray, npt.NDArray[np.int_], list[str]
]:
    """Build ``(Y, T, X, W, time_codes, feature_names)`` with multi-feature ``X``.

    The effect-modifier matrix ``X`` carries several predetermined cell
    characteristics -- ``exposure`` (sector sensitivity), ``base_share`` (within-
    state employment share) and ``log_base_emp`` (cell size) -- so the causal
    forest can attribute heterogeneity across them.
    """
    features = ["exposure", "base_share", "log_base_emp"]
    base = (
        panel.join(exposure, on="supersector_code", how="inner")
        .join(_cell_base_features(panel), on="unit_id", how="inner")
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
        .select(["target", "treatment", *features, "period", *controls])
        .drop_nulls()
    )
    y = frame["target"].to_numpy().astype(np.float64)
    t = frame["treatment"].to_numpy().astype(np.float64)
    x = frame.select(features).to_numpy().astype(np.float64)
    w = frame.select(controls).to_numpy().astype(np.float64)
    time_codes = frame["period"].to_numpy().astype(int)
    return y, t, x, w, time_codes, features


def estimate_cate_drivers(
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
) -> CateDriverResult:
    """Fit a multi-feature CausalForestDML and extract heterogeneity drivers.

    Returns which predetermined cell characteristics drive the treatment-effect
    heterogeneity (normalised forest feature importances) and how the CATE differs
    between the top and bottom exposure tercile.
    """
    y, t, x, w, time_codes, features = build_driver_sample(
        panel, exposure, shock, shock_col=shock_col, horizon=horizon, n_lags=n_lags
    )
    cv = PurgedTimeBlockedCV(time_codes, n_splits=n_splits, embargo=embargo)
    forest = CausalForestDML(
        model_y=_nuisance(), model_t=_nuisance(), cv=cv, random_state=seed
    )
    forest.fit(y, t, X=x, W=w)

    # Best Linear Predictor of the CATE on the standardised effect modifiers
    # (impurity importances are cardinality-biased and are not used).
    cate = np.asarray(forest.const_marginal_effect(x), dtype=np.float64).ravel()
    std = x.std(axis=0, ddof=0)
    std[std == 0.0] = 1.0
    x_std = (x - x.mean(axis=0)) / std
    design = np.column_stack([np.ones(x_std.shape[0]), x_std])
    coef, *_ = np.linalg.lstsq(design, cate, rcond=None)
    blp = coef[1:]
    abs_blp = np.abs(blp)
    denom = float(abs_blp.sum())
    importances = abs_blp / denom if denom > 0 else abs_blp

    exposure_col = x[:, features.index("exposure")]
    lo, hi = np.quantile(exposure_col, [1 / 3, 2 / 3])
    bottom = cate[exposure_col <= lo]
    top = cate[exposure_col >= hi]
    return CateDriverResult(
        horizon=horizon,
        n_obs=int(y.size),
        features=features,
        blp_coef=[float(v) for v in blp],
        importances=[float(v) for v in importances],
        forest_ate=float(forest.ate(x)),
        cate_top_exposure_tercile=float(top.mean()) if top.size else float("nan"),
        cate_bottom_exposure_tercile=(
            float(bottom.mean()) if bottom.size else float("nan")
        ),
    )


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
