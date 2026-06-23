"""Proxy-SVAR / external-instrument identification of the policy shock.

Estimates a reduced-form VAR on a small macro block, then instruments the
policy-equation reduced-form residual with an external instrument (a published
high-frequency-identified shock; the instrument is borrowed, the SVAR is ours).
The first-stage strength is reported with a heteroskedasticity-robust F and a
weak-instrument flag (Montiel Olea-Pflueger guidance). Impulse responses are
deferred to the aggregate-IRF phase.

References
----------
Stock & Watson (2018), Economic Journal 128; Mertens & Ravn (2013), AER 103(4);
Gertler & Karadi (2015), AEJ:Macro 7(1); Montiel Olea & Pflueger (2013), JBES.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import statsmodels.api as sm
from pydantic import BaseModel
from statsmodels.tsa.api import VAR


class FirstStageDiagnostics(BaseModel):
    """First-stage strength of the external instrument.

    Parameters
    ----------
    n_obs
        Observations in the first-stage regression.
    coefficient
        First-stage coefficient on the instrument.
    robust_f
        Heteroskedasticity-robust first-stage F (= robust t-squared for a single
        instrument); used as the Montiel Olea-Pflueger effective F.
    partial_r_squared
        First-stage R-squared on the instrument.
    weak
        Whether ``robust_f`` is below the configured threshold.
    """

    n_obs: int
    coefficient: float
    robust_f: float
    partial_r_squared: float
    weak: bool


def build_macro_block(
    policy_rate: pl.DataFrame, macro_current: pl.DataFrame
) -> pl.DataFrame:
    """Assemble the reduced-form VAR block: rate, activity, prices, slack.

    Parameters
    ----------
    policy_rate
        Spliced policy-rate frame (``date``, ``policy_rate``).
    macro_current
        Latest-vintage macro frame
        (:data:`cil.data.schemas.MACRO_CURRENT_SCHEMA`).

    Returns
    -------
    polars.DataFrame
        Columns ``date``, ``policy_rate``, ``log_ip``, ``log_cpi``,
        ``unemployment``.
    """
    wide = macro_current.pivot(
        values="value", index="reference_date", on="series_id"
    ).rename({"reference_date": "date"})
    block = (
        policy_rate.select("date", "policy_rate")
        .join(wide, on="date", how="inner")
        .select(
            "date",
            "policy_rate",
            log_ip=100.0 * pl.col("INDPRO").log(),
            log_cpi=100.0 * pl.col("CPIAUCSL").log(),
            unemployment=pl.col("UNRATE"),
        )
        .drop_nulls()
        .sort("date")
    )
    return block


def first_stage(
    macro_block: pl.DataFrame,
    instrument: pl.DataFrame,
    *,
    instrument_col: str,
    n_lags: int = 12,
    weak_f_threshold: float = 10.0,
) -> tuple[pl.DataFrame, FirstStageDiagnostics]:
    """Run the proxy-SVAR first stage and return the identified shock proxy.

    Parameters
    ----------
    macro_block
        The VAR block from :func:`build_macro_block`.
    instrument
        Frame with ``date`` and the instrument column.
    instrument_col
        Name of the instrument column.
    n_lags
        Reduced-form VAR lag order.
    weak_f_threshold
        First-stage F threshold below which the instrument is flagged weak.

    Returns
    -------
    shock : polars.DataFrame
        Columns ``date`` and ``svar_shock`` (instrument-projected policy
        residual, standardized).
    diagnostics : FirstStageDiagnostics
        First-stage strength.
    """
    endog = macro_block.drop("date").to_pandas()
    var_result = VAR(endog).fit(n_lags)
    resid = var_result.resid["policy_rate"].to_numpy()
    resid_dates = macro_block["date"].to_list()[n_lags:]
    resid_frame = pl.DataFrame({"date": resid_dates, "policy_resid": resid})

    merged = resid_frame.join(
        instrument.select("date", instrument_col), on="date", how="inner"
    ).drop_nulls()
    y = merged["policy_resid"].to_numpy()
    z = merged[instrument_col].to_numpy()
    x = sm.add_constant(z)
    fit = sm.OLS(y, x).fit(cov_type="HC1")
    t_stat = float(fit.tvalues[1])
    robust_f = t_stat**2
    projected = fit.fittedvalues
    std = float(np.std(projected))
    shock = pl.DataFrame(
        {
            "date": merged["date"],
            "svar_shock": projected / std if std > 0 else projected,
        }
    )
    diagnostics = FirstStageDiagnostics(
        n_obs=int(fit.nobs),
        coefficient=float(fit.params[1]),
        robust_f=robust_f,
        partial_r_squared=float(fit.rsquared),
        weak=robust_f < weak_f_threshold,
    )
    return shock, diagnostics
