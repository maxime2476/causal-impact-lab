"""LP-DiD: a local-projections difference-in-differences estimator.

Implements the Dube-Girardi-Jorda-Taylor (2025) LP-DiD estimator with the
**horizon-dependent** clean-control condition: at each horizon the long-difference
outcome is regressed on the treatment-change indicator, with the control pool
restricted to units that stay untreated through the outcome window (``t+h``).
Excluding both already-treated units *and* not-yet-treated controls that adopt
inside the window avoids the two-way-FE negative-weighting problem
(Goodman-Bacon 2021) and the horizon-dependent contamination bias it would
otherwise induce under staggered adoption.

There is no maintained Python implementation (the reference is Stata:
github.com/danielegirardi/lpdid). This port is validated by two-way-FE
equivalence on a toy panel and by a **known-DGP staggered golden** that recovers a
prescribed dynamic effect path to numerical precision (``tests/golden``,
ADR-0018); a Stata cross-validation can be dropped in later.

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
    """Add the treatment-change indicator (0->1 switch = newly treated)."""
    return data.sort([unit_col, time_col]).with_columns(
        d_treat=(pl.col(treat_col) - pl.col(treat_col).shift(1)).over(unit_col),
    )


def _fit_horizon(
    prepared: pl.DataFrame,
    horizon: int,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    treat_col: str,
    confidence_level: float,
) -> dict[str, float]:
    """Estimate the LP-DiD ATT at one horizon with horizon-dependent clean controls.

    The clean-control condition is horizon-specific (Dube-Girardi-Jorda-Taylor):
    the treated group is the units newly treated at ``t`` (``d_treat == 1``); a
    clean control is a unit untreated at ``t`` that stays untreated through the
    outcome window (``t+h`` for a response horizon). Controls that switch on
    *inside* the window would contaminate the long difference, so they are
    excluded here rather than by a horizon-independent already-treated filter.
    (Absorbing / staggered adoption is assumed.)
    """
    outcome = (pl.col(outcome_col).shift(-horizon) - pl.col(outcome_col).shift(1)).over(
        unit_col
    )
    # Treatment status at the end of the outcome window (t+h). For leads (h < 0)
    # the window is pre-event, so cleanliness only requires being untreated at t.
    lead_treated = pl.col(treat_col).shift(-horizon).over(unit_col)
    newly_treated = pl.col(D_TREAT) == 1
    if horizon >= 0:
        clean_control = (pl.col(treat_col) == 0) & (lead_treated == 0)
    else:
        clean_control = pl.col(treat_col) == 0
    frame = (
        prepared.with_columns(outcome=outcome)
        .filter(newly_treated | clean_control)
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
            prepared,
            h,
            unit_col,
            time_col,
            outcome_col,
            treat_col,
            config.confidence_level,
        )
        for h in config.horizons
    ]
    return pl.DataFrame(rows).sort("horizon")
