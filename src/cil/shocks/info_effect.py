"""Information-effect test (Jarocinski-Karadi), monthly proxy and high-frequency.

Classifies each monetary surprise as a standard policy shock or a central-bank
information shock by the co-movement of the surprise with equity returns: a
tightening surprise that *raises* equities (same sign) signals the Fed revealed
positive economic news -- an information shock; opposite-sign co-movement is a
conventional policy shock. The share of information-classified observations
quantifies contamination.

Two variants are provided:

- :func:`classify` -- the **monthly proxy**: monthly equity returns stand in for
  the announcement window (a coarse approximation).
- :func:`classify_high_frequency` -- the true Jarocinski-Karadi "poor man's sign
  restriction": both the rate surprise and the equity move are measured in the
  same *high-frequency announcement window* (from ``mps_fomc``: ``mps`` and the
  same-window ``sp500``). The monthly proxy overstates contamination because
  month-long equity moves are dominated by non-FOMC news.

References
----------
Jarocinski & Karadi (2020), *Deconstructing Monetary Policy Surprises*,
AEJ:Macro 12(2).
"""

from __future__ import annotations

import polars as pl
from pydantic import BaseModel


class InfoEffectSummary(BaseModel):
    """Summary of the information-effect classification.

    Parameters
    ----------
    n_months
        Number of classified observations (nonzero surprises).
    n_information
        Observations classified as information shocks (surprise and equity
        co-move).
    contamination_share
        Fraction of classified observations that are information shocks.
    """

    n_months: int
    n_information: int
    contamination_share: float


def _classify_core(
    merged: pl.DataFrame, shock_col: str, return_col: str
) -> tuple[pl.DataFrame, InfoEffectSummary]:
    """Apply the sign co-movement rule to a merged frame; summarise it."""
    classified = merged.with_columns(
        is_information=(pl.col(shock_col).sign() == pl.col(return_col).sign())
    ).with_columns(
        monetary_component=pl.when(pl.col("is_information"))
        .then(0.0)
        .otherwise(pl.col(shock_col))
    )
    n_months = classified.height
    n_information = int(classified["is_information"].sum())
    summary = InfoEffectSummary(
        n_months=n_months,
        n_information=n_information,
        contamination_share=(n_information / n_months) if n_months else 0.0,
    )
    return classified, summary


def classify(
    shock: pl.DataFrame,
    equity_returns: pl.DataFrame,
    *,
    shock_col: str,
    return_col: str = "equity_return",
) -> tuple[pl.DataFrame, InfoEffectSummary]:
    """Classify shocks as policy vs information (monthly proxy).

    Parameters
    ----------
    shock
        Frame with ``date`` and the shock column.
    equity_returns
        Frame with ``date`` and the equity-return column.
    shock_col
        Name of the shock column.
    return_col
        Name of the equity-return column.

    Returns
    -------
    classified : polars.DataFrame
        Columns ``date``, the shock, the return, ``is_information`` (bool), and
        ``monetary_component`` (the shock on policy months, else 0).
    summary : InfoEffectSummary
        Aggregate contamination statistics.
    """
    merged = (
        shock.select("date", shock_col)
        .join(equity_returns.select("date", return_col), on="date", how="inner")
        .drop_nulls()
        .filter(pl.col(shock_col) != 0.0)
        .sort("date")
    )
    return _classify_core(merged, shock_col, return_col)


def classify_high_frequency(
    fomc: pl.DataFrame,
    *,
    shock_col: str = "mps",
    equity_col: str = "sp500",
) -> tuple[pl.DataFrame, InfoEffectSummary]:
    """Classify per-FOMC surprises using the same-window equity move.

    This is the true high-frequency test: both the rate surprise and the equity
    move come from the same announcement window (no monthly proxy), so it is not
    contaminated by non-FOMC equity news.

    Parameters
    ----------
    fomc
        Per-FOMC frame with ``date``, the rate-surprise column, and the
        same-window equity column (e.g. the ``mps_fomc`` table).
    shock_col
        Name of the rate-surprise column.
    equity_col
        Name of the same-window equity-move column.

    Returns
    -------
    classified : polars.DataFrame
        Per-FOMC classification with ``is_information`` and
        ``monetary_component`` (the surprise on policy events, else 0).
    summary : InfoEffectSummary
        Aggregate contamination statistics.
    """
    merged = (
        fomc.select("date", shock_col, equity_col)
        .drop_nulls()
        .filter(pl.col(shock_col) != 0.0)
        .sort("date")
    )
    return _classify_core(merged, shock_col, equity_col)
