"""LP-DiD: a local-projections difference-in-differences estimator.

Implements the Dube-Girardi-Jorda-Taylor (2025) LP-DiD estimator with the
clean-control condition: at each horizon the long-difference outcome is
regressed on the treatment-change indicator, excluding already-treated units
from the control pool so the two-way-FE negative-weighting problem
(Goodman-Bacon 2021) does not arise.

There is no maintained Python implementation (the reference is Stata:
github.com/danielegirardi/lpdid). This port is validated by analytic /
two-way-FE equivalence on toy panels; a slot for Stata golden fixtures is left
in ``tests/golden`` to drop in later.

References
----------
Dube, Girardi, Jorda & Taylor (2025), *A Local Projections Approach to
Difference-in-Differences*, Journal of Applied Econometrics.
"""

from __future__ import annotations

import polars as pl
from linearmodels.panel import PanelOLS
from pydantic import BaseModel
from scipy.stats import norm

D_TREAT = "d_treat"


class LPDiDConfig(BaseModel):
    """Configuration for the LP-DiD estimator.

    Parameters
    ----------
    horizons
        Non-negative response horizons (and optional negative leads).
    confidence_level
        Two-sided confidence level for intervals.
    """

    horizons: tuple[int, ...]
    confidence_level: float = 0.95


def _prepare(
    data: pl.DataFrame, unit_col: str, time_col: str, treat_col: str, outcome_col: str
) -> pl.DataFrame:
    """Add the treatment-change indicator and the already-treated flag."""
    return data.sort([unit_col, time_col]).with_columns(
        d_treat=(pl.col(treat_col) - pl.col(treat_col).shift(1)).over(unit_col),
        already_treated=(pl.col(treat_col).shift(1) == 1).over(unit_col),
    )


def _fit_horizon(
    prepared: pl.DataFrame,
    horizon: int,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    confidence_level: float,
) -> dict[str, float]:
    """Estimate the LP-DiD ATT at one horizon with clean controls."""
    outcome = (pl.col(outcome_col).shift(-horizon) - pl.col(outcome_col).shift(1)).over(
        unit_col
    )
    frame = (
        prepared.with_columns(outcome=outcome)
        # Clean-control condition: drop already-treated control rows.
        .filter(~pl.col("already_treated").fill_null(value=False))
        .select([unit_col, time_col, "outcome", D_TREAT])
        .drop_nulls()
    )
    pdf = frame.to_pandas().set_index([unit_col, time_col])
    result = PanelOLS(
        pdf["outcome"], pdf[[D_TREAT]], entity_effects=True, time_effects=True
    ).fit(cov_type="clustered", cluster_entity=True)
    beta = float(result.params[D_TREAT])
    se = float(result.std_errors[D_TREAT])
    z = float(norm.ppf(0.5 + confidence_level / 2.0))
    return {
        "horizon": float(horizon),
        "att": beta,
        "se": se,
        "ci_low": beta - z * se,
        "ci_high": beta + z * se,
        "n_obs": float(result.nobs),
    }


def lp_did(
    data: pl.DataFrame,
    config: LPDiDConfig,
    *,
    unit_col: str = "unit_id",
    time_col: str = "date",
    outcome_col: str = "log_employment",
    treat_col: str = "treated",
) -> pl.DataFrame:
    """Estimate dynamic ATTs by LP-DiD with clean controls.

    Parameters
    ----------
    data
        Long panel with unit, time, outcome, and a binary treatment column.
    config
        Estimation configuration.
    unit_col, time_col, outcome_col, treat_col
        Column names.

    Returns
    -------
    polars.DataFrame
        One row per horizon: ``att``, ``se``, ``ci_low``, ``ci_high``,
        ``n_obs``.
    """
    prepared = _prepare(data, unit_col, time_col, treat_col, outcome_col)
    rows = [
        _fit_horizon(
            prepared, h, unit_col, time_col, outcome_col, config.confidence_level
        )
        for h in config.horizons
    ]
    return pl.DataFrame(rows).sort("horizon")
