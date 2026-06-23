"""Information-effect test (Jarocinski-Karadi), monthly proxy.

Classifies each monetary surprise as a standard policy shock or a central-bank
information shock by the co-movement of the surprise with broad equity returns:
a tightening surprise that *raises* equities (same sign) signals the Fed revealed
positive economic news -- an information shock; opposite-sign co-movement is a
conventional policy shock. The share of information-classified months quantifies
contamination.

This is a monthly proxy of the Jarocinski-Karadi high-frequency "poor man's sign
restrictions"; intraday data is not freely reproducible, so monthly equity
returns stand in for the high-frequency window.

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
        Number of classified months (nonzero shocks).
    n_information
        Months classified as information shocks (shock and equity co-move).
    contamination_share
        Fraction of classified months that are information shocks.
    """

    n_months: int
    n_information: int
    contamination_share: float


def classify(
    shock: pl.DataFrame,
    equity_returns: pl.DataFrame,
    *,
    shock_col: str,
    return_col: str = "equity_return",
) -> tuple[pl.DataFrame, InfoEffectSummary]:
    """Classify shocks as policy vs information and quantify contamination.

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
