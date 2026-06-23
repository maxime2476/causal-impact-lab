"""Predictability test for monetary shocks (Bauer-Swanson style).

A clean monetary shock should be unpredictable from information available before
the policy action. This regresses the shock on lagged macro predictors and its
own lag, reporting the R-squared and the joint F-test p-value. A large R-squared
or a small p-value indicates predictability -- a sign the shock is contaminated
by the systematic policy response.

References
----------
Bauer & Swanson (2023), *A Reassessment of Monetary Policy Surprises and
High-Frequency Identification*, NBER Macroeconomics Annual 37.
"""

from __future__ import annotations

import polars as pl
import statsmodels.api as sm
from pydantic import BaseModel

from cil.shocks.features import add_lags


class PredictabilityResult(BaseModel):
    """Result of the predictability regression.

    Parameters
    ----------
    n_obs
        Observations in the regression.
    r_squared
        Share of shock variance explained by the lagged predictors.
    f_pvalue
        Joint F-test p-value (H0: shock is unpredictable). Large is good.
    predictable
        Whether the joint test rejects unpredictability at 5%.
    """

    n_obs: int
    r_squared: float
    f_pvalue: float
    predictable: bool


def predictability_test(
    shock: pl.DataFrame,
    predictors: pl.DataFrame,
    *,
    shock_col: str,
    predictor_cols: list[str],
    n_lags: int = 6,
    hac_maxlags: int = 6,
) -> PredictabilityResult:
    """Regress the shock on lagged predictors and report predictability.

    Parameters
    ----------
    shock
        Frame with ``date`` and the shock column.
    predictors
        Frame with ``date`` and the predictor columns.
    shock_col
        Name of the shock column.
    predictor_cols
        Predictor column names (lagged internally).
    n_lags
        Number of monthly lags of each predictor and of the shock itself.
    hac_maxlags
        Newey-West truncation lag.

    Returns
    -------
    PredictabilityResult
        Predictability diagnostics.
    """
    merged = shock.select("date", shock_col).join(predictors, on="date", how="inner")
    merged = add_lags(merged, [*predictor_cols, shock_col], n_lags).drop_nulls()
    lag_cols = [
        f"{c}_l{lag}"
        for c in [*predictor_cols, shock_col]
        for lag in range(1, n_lags + 1)
    ]
    pdf = merged.select([shock_col, *lag_cols]).to_pandas()
    y = pdf[shock_col]
    x = sm.add_constant(pdf[lag_cols])
    fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": hac_maxlags})
    f_pvalue = float(fit.f_pvalue)
    return PredictabilityResult(
        n_obs=int(fit.nobs),
        r_squared=float(fit.rsquared),
        f_pvalue=f_pvalue,
        predictable=f_pvalue < 0.05,
    )
