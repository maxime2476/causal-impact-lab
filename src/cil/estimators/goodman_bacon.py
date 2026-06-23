"""Goodman-Bacon diagnostics for staggered/non-absorbing treatment.

A light diagnostic, not a full Bacon decomposition: it contrasts the static
two-way fixed-effects (TWFE) estimate with the clean-control LP-DiD estimate and
reports the share of already-treated observations that TWFE uses as controls
("forbidden comparisons"). A large gap or a large forbidden share warns that the
TWFE estimate is contaminated by negative weighting.

References
----------
Goodman-Bacon (2021), *Difference-in-differences with variation in treatment
timing*, Journal of Econometrics 225(2).
"""

from __future__ import annotations

import polars as pl
from linearmodels.panel import PanelOLS
from pydantic import BaseModel

from cil.estimators.lp_did import LPDiDConfig, lp_did


class BaconDiagnostic(BaseModel):
    """Comparison of TWFE and clean-control estimates.

    Parameters
    ----------
    twfe_estimate
        Static TWFE coefficient on the treatment level.
    clean_estimate
        Clean-control LP-DiD estimate at horizon 0.
    difference
        ``twfe_estimate - clean_estimate``; large magnitude flags negative
        weighting.
    forbidden_share
        Share of observations that are already-treated (used as controls by
        TWFE but excluded by the clean-control condition).
    """

    twfe_estimate: float
    clean_estimate: float
    difference: float
    forbidden_share: float


def twfe_estimate(
    data: pl.DataFrame,
    *,
    unit_col: str = "unit_id",
    time_col: str = "date",
    outcome_col: str = "log_employment",
    treat_col: str = "treated",
) -> float:
    """Return the static TWFE coefficient on the treatment level."""
    pdf = data.select([unit_col, time_col, outcome_col, treat_col]).to_pandas()
    pdf = pdf.set_index([unit_col, time_col])
    result = PanelOLS(
        pdf[outcome_col], pdf[[treat_col]], entity_effects=True, time_effects=True
    ).fit(cov_type="clustered", cluster_entity=True)
    return float(result.params[treat_col])


def forbidden_share(
    data: pl.DataFrame, *, unit_col: str = "unit_id", treat_col: str = "treated"
) -> float:
    """Return the share of already-treated observations (forbidden controls)."""
    flagged = data.sort([unit_col, "date"]).with_columns(
        already=(pl.col(treat_col).shift(1) == 1).over(unit_col)
    )
    flags = flagged["already"].fill_null(value=False).to_numpy().astype(float)
    return float(flags.mean())


def bacon_diagnostic(
    data: pl.DataFrame,
    *,
    unit_col: str = "unit_id",
    time_col: str = "date",
    outcome_col: str = "log_employment",
    treat_col: str = "treated",
) -> BaconDiagnostic:
    """Compute the TWFE-vs-clean diagnostic and the forbidden-comparison share.

    Parameters
    ----------
    data
        Long panel with unit, time, outcome, and a binary treatment column.
    unit_col, time_col, outcome_col, treat_col
        Column names.

    Returns
    -------
    BaconDiagnostic
        The comparison summary.
    """
    twfe = twfe_estimate(
        data,
        unit_col=unit_col,
        time_col=time_col,
        outcome_col=outcome_col,
        treat_col=treat_col,
    )
    clean_series = lp_did(
        data,
        LPDiDConfig(horizons=(0,)),
        unit_col=unit_col,
        time_col=time_col,
        outcome_col=outcome_col,
        treat_col=treat_col,
    )["att"]
    clean = float(clean_series.to_numpy()[0])
    share = forbidden_share(data, unit_col=unit_col, treat_col=treat_col)
    return BaconDiagnostic(
        twfe_estimate=twfe,
        clean_estimate=clean,
        difference=twfe - clean,
        forbidden_share=share,
    )
