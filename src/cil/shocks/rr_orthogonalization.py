"""Romer-Romer-style orthogonalization of the policy rate.

Regresses the monthly change in the policy rate on the Fed's real-time
information set (real-time inflation, industrial-production growth, the
unemployment level and its change, with lags, plus the lagged rate level). The
residuals are the monetary shock: the component of the policy action not
explained by the systematic response to observed conditions. A positive
residual is a tighter-than-predicted move (contractionary, ``s_t > 0``).

This is a reproducible, public-data proxy for Romer-Romer (2004), which used
Greenbook forecasts; here the real-time vintages stand in for the forecasts.

References
----------
Romer & Romer (2004), *A New Measure of Monetary Policy Shocks*, AER 94(4).
"""

from __future__ import annotations

import polars as pl
import statsmodels.api as sm
from pydantic import BaseModel

from cil.shocks.features import add_lags, realtime_macro_features

_BASE_FEATURES = ["inflation", "ip_growth", "unemployment", "d_unemployment"]


class OrthogonalizationDiagnostics(BaseModel):
    """Fit diagnostics for the orthogonalization regression.

    Parameters
    ----------
    n_obs
        Number of observations in the regression.
    r_squared
        Coefficient of determination (systematic predictability of the action).
    hac_maxlags
        Newey-West lag truncation used for the HAC covariance.
    """

    n_obs: int
    r_squared: float
    hac_maxlags: int


def romer_romer_shock(
    macro_pit: pl.DataFrame,
    policy_rate: pl.DataFrame,
    *,
    n_lags: int = 2,
    hac_maxlags: int = 6,
) -> tuple[pl.DataFrame, OrthogonalizationDiagnostics]:
    """Compute the Romer-Romer-style shock series.

    Parameters
    ----------
    macro_pit
        Point-in-time macro frame for the real-time information set.
    policy_rate
        Spliced policy-rate frame with columns ``date`` and ``policy_rate``.
    n_lags
        Monthly lags of the information set to include.
    hac_maxlags
        Newey-West truncation lag for HAC standard errors.

    Returns
    -------
    shock : polars.DataFrame
        Columns ``date`` and ``rr_shock`` (the regression residual).
    diagnostics : OrthogonalizationDiagnostics
        Fit diagnostics.
    """
    features = realtime_macro_features(macro_pit)
    features = add_lags(features, _BASE_FEATURES, n_lags)
    policy = (
        policy_rate.select("date", "policy_rate")
        .sort("date")
        .with_columns(
            d_policy=pl.col("policy_rate") - pl.col("policy_rate").shift(1),
            policy_lag=pl.col("policy_rate").shift(1),
        )
    )
    design = features.join(policy, on="date", how="inner").drop_nulls()
    predictors = [
        *_BASE_FEATURES,
        *[f"{c}_l{lag}" for c in _BASE_FEATURES for lag in range(1, n_lags + 1)],
        "policy_lag",
    ]
    pdf = design.select(["date", "d_policy", *predictors]).to_pandas()
    y = pdf["d_policy"]
    x = sm.add_constant(pdf[predictors])
    model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": hac_maxlags})
    shock = pl.DataFrame({"date": design["date"], "rr_shock": model.resid.to_numpy()})
    diagnostics = OrthogonalizationDiagnostics(
        n_obs=int(model.nobs),
        r_squared=float(model.rsquared),
        hac_maxlags=hac_maxlags,
    )
    return shock, diagnostics
