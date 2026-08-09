"""Interacted panel local projection -- the headline estimator.

For each horizon ``h`` estimates

    y_{i,t+h} - y_{i,t-1} = beta_h (E_i * s_t) + gamma_i + delta_t
                            + sum_l phi_{h,l} dy_{i,t-l} + eps,

with unit and time fixed effects. Time effects absorb the aggregate component,
so ``beta_h`` is the relative (cross-sectional) semi-elasticity per unit
exposure per unit shock (see ``docs/methods.md``). Inference is Driscoll-Kraay
(cross-sectional + serial robust); response-horizon p-values are BH-FDR adjusted.

References
----------
Jorda (2005), AER 95(1).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from linearmodels.panel import PanelOLS
from pydantic import BaseModel
from scipy.stats import norm

from cil.inference.bh_fdr import bh_adjust

TREATMENT = "treatment"


class PanelLPConfig(BaseModel):
    """Configuration for the panel local projection.

    Parameters
    ----------
    horizons
        Horizons to estimate (negative are pre-trend leads).
    n_control_lags
        Number of lagged monthly employment changes included as controls.
    confidence_level
        Two-sided confidence level for intervals.
    """

    horizons: tuple[int, ...]
    n_control_lags: int = 6
    confidence_level: float = 0.95


def _prepare(
    panel: pl.DataFrame,
    exposure: pl.DataFrame,
    shock: pl.DataFrame,
    shock_col: str,
    n_control_lags: int,
) -> pl.DataFrame:
    """Attach exposure, shock, the treatment interaction, and control lags."""
    base = (
        panel.join(exposure, on="supersector_code", how="inner")
        .join(shock.select("date", shock_col), on="date", how="inner")
        .sort(["unit_id", "date"])
        .with_columns(treatment=pl.col("exposure") * pl.col(shock_col))
    )
    control_exprs = [
        (pl.col("log_employment").shift(lag) - pl.col("log_employment").shift(lag + 1))
        .over("unit_id")
        .alias(f"dy_l{lag}")
        for lag in range(1, n_control_lags + 1)
    ]
    return base.with_columns(control_exprs)


def _fit_horizon(
    prepared: pl.DataFrame,
    horizon: int,
    controls: list[str],
    confidence_level: float,
    *,
    cluster_col: str | None = None,
) -> dict[str, float]:
    """Estimate one horizon and return its coefficient row.

    With ``cluster_col=None`` the covariance is Driscoll-Kraay (kernel), robust to
    cross-sectional and serial correlation. With ``cluster_col`` set, the
    covariance is **two-way clustered on that exposure dimension (supersector) and
    on time** -- the exposure-robust choice for this shift-share design. The
    sector dimension follows Borusyak-Hull-Jaravel (2022) (the shifter is common
    to all units sharing a supersector); the *time* dimension is essential because
    the aggregate shock ``s_t`` is common to every cell at ``t``, so naive one-way
    sector clustering would ignore that within-time correlation and badly
    understate the standard errors (see ADR-0017).
    """
    outcome = (
        pl.col("log_employment").shift(-horizon) - pl.col("log_employment").shift(1)
    ).over("unit_id")
    keep = ["unit_id", "date", "outcome", TREATMENT, *controls]
    if cluster_col is not None and cluster_col not in keep:
        keep.append(cluster_col)
    frame = prepared.with_columns(outcome=outcome).select(keep).drop_nulls()
    pdf = frame.to_pandas().set_index(["unit_id", "date"])
    exog = pdf[[TREATMENT, *controls]]
    model = PanelOLS(pdf["outcome"], exog, entity_effects=True, time_effects=True)
    if cluster_col is None:
        result = model.fit(cov_type="kernel", kernel="bartlett")
    else:
        clusters = pdf[[cluster_col]].copy()
        clusters["_time"] = pdf.index.get_level_values("date")
        result = model.fit(cov_type="clustered", clusters=clusters)
    beta = float(result.params[TREATMENT])
    se = float(result.std_errors[TREATMENT])
    z = float(norm.ppf(0.5 + confidence_level / 2.0))
    p_value = float(2.0 * (1.0 - norm.cdf(abs(beta / se)))) if se > 0 else float("nan")
    return {
        "horizon": float(horizon),
        "beta": beta,
        "se": se,
        "t_stat": beta / se if se > 0 else float("nan"),
        "p_value": p_value,
        "ci_low": beta - z * se,
        "ci_high": beta + z * se,
        "n_obs": float(result.nobs),
    }


def run_panel_lp(
    panel: pl.DataFrame,
    exposure: pl.DataFrame,
    shock: pl.DataFrame,
    config: PanelLPConfig,
    *,
    shock_col: str,
) -> pl.DataFrame:
    """Estimate the interacted panel LP across horizons.

    Parameters
    ----------
    panel
        Analysis-ready cell panel (``unit_id``, ``supersector_code``, ``date``,
        ``log_employment``).
    exposure
        Cell exposure shifter (``supersector_code``, ``exposure``).
    shock
        Shock frame (``date`` and the shock column).
    config
        Estimation configuration.
    shock_col
        Name of the shock column in *shock*.

    Returns
    -------
    polars.DataFrame
        One row per horizon: ``beta``, ``se``, ``t_stat``, ``p_value``,
        ``ci_low``, ``ci_high``, ``n_obs``, and ``p_value_bh`` (BH-FDR adjusted
        across response horizons ``h >= 0``).
    """
    controls = [f"dy_l{lag}" for lag in range(1, config.n_control_lags + 1)]
    prepared = _prepare(panel, exposure, shock, shock_col, config.n_control_lags)
    rows = [
        # Horizon -1 is the event-study reference (outcome y_{t-1}-y_{t-1} == 0)
        # and is omitted. Pre-trend leads use fixed effects only: a lead outcome
        # is mechanically collinear with the lagged-difference controls, so
        # including them is degenerate. Response horizons keep the controls.
        _fit_horizon(prepared, h, controls if h >= 0 else [], config.confidence_level)
        for h in config.horizons
        if h != -1
    ]
    result = pl.DataFrame(rows).sort("horizon")
    response = result.filter(pl.col("horizon") >= 0)
    adjusted = bh_adjust(response["p_value"].to_numpy())
    bh_map = dict(zip(response["horizon"].to_list(), adjusted.tolist(), strict=True))
    return result.with_columns(
        p_value_bh=pl.col("horizon").map_elements(
            lambda h: bh_map.get(h, float("nan")), return_dtype=pl.Float64
        )
    )


def run_panel_lp_exposure_robust(
    panel: pl.DataFrame,
    exposure: pl.DataFrame,
    shock: pl.DataFrame,
    config: PanelLPConfig,
    *,
    shock_col: str,
    cluster_col: str = "supersector_code",
) -> pl.DataFrame:
    """Estimate the panel LP with **exposure-robust** (BHJ) standard errors.

    Identical point estimates to :func:`run_panel_lp`, but the covariance is
    two-way clustered on the exposure dimension (*cluster_col*, the supersector)
    **and on time**. The sector dimension is the Borusyak-Hull-Jaravel (2022)
    exposure-robust cluster (the shifter is common to units sharing a supersector;
    Adao-Kolesar-Morales 2019 give an asymptotically equivalent variance); the
    time dimension captures the common aggregate shock. In this design the two-way
    SE lands very close to Driscoll-Kraay, whereas naive one-way sector clustering
    would understate it ~3x (ADR-0017). Response horizons carry a BH-FDR column.

    Returns
    -------
    polars.DataFrame
        One row per horizon with the same columns as :func:`run_panel_lp`; the
        ``se``/``t_stat``/``p_value``/CI reflect the two-way clustered covariance.
    """
    controls = [f"dy_l{lag}" for lag in range(1, config.n_control_lags + 1)]
    prepared = _prepare(panel, exposure, shock, shock_col, config.n_control_lags)
    rows = [
        _fit_horizon(
            prepared,
            h,
            controls if h >= 0 else [],
            config.confidence_level,
            cluster_col=cluster_col,
        )
        for h in config.horizons
        if h != -1
    ]
    result = pl.DataFrame(rows).sort("horizon")
    response = result.filter(pl.col("horizon") >= 0)
    adjusted = bh_adjust(response["p_value"].to_numpy())
    bh_map = dict(zip(response["horizon"].to_list(), adjusted.tolist(), strict=True))
    return result.with_columns(
        p_value_bh=pl.col("horizon").map_elements(
            lambda h: bh_map.get(h, float("nan")), return_dtype=pl.Float64
        )
    )


def leads_summary(result: pl.DataFrame, fdr_alpha: float = 0.10) -> dict[str, float]:
    """Summarize the pre-trend leads (horizons ``h < 0``).

    Parameters
    ----------
    result
        Output of :func:`run_panel_lp`.
    fdr_alpha
        FDR level for flagging any significant lead.

    Returns
    -------
    dict of str to float
        ``n_leads``, ``max_abs_t``, ``min_p``, and ``any_significant`` (1.0 if a
        lead is significant after BH adjustment among the leads).
    """
    leads = result.filter(pl.col("horizon") < 0)
    if leads.height == 0:
        return {
            "n_leads": 0.0,
            "max_abs_t": float("nan"),
            "min_p": float("nan"),
            "any_significant": 0.0,
        }
    adj = bh_adjust(leads["p_value"].to_numpy())
    return {
        "n_leads": float(leads.height),
        "max_abs_t": float(np.nanmax(np.abs(leads["t_stat"].to_numpy()))),
        "min_p": float(np.nanmin(leads["p_value"].to_numpy())),
        "any_significant": float(bool(np.any(adj <= fdr_alpha))),
    }
