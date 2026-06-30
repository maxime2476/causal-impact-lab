"""Shift-share (Bartik) interest-rate exposure construction.

Cell exposure is the predetermined interest-rate sensitivity of the cell's
supersector. Two shifters are provided (per ADR-0005): an *estimated*
semi-elasticity of national supersector employment to the policy/shadow rate
(headline), and a *documented* duration/credit-dependence proxy (robustness).
The state-level exposure ``E_s = sum_k omega_{s,k} sigma_k`` uses predetermined
base-period employment shares.

Identification leans on the exogenous-shocks justification (Borusyak-Hull-Jaravel
2022): the national shock is plausibly exogenous and the shares are
predetermined. Estimating the shifter on *national* aggregates (separate
variation from the state-by-supersector panel) and from the *rate* rather than
the shock limits mechanical circularity.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import statsmodels.api as sm

#: Documented ordinal interest-sensitivity proxy by QCEW supersector code.
#: Higher = more interest-sensitive (durable/credit/rate-duration intensive),
#: from the monetary-transmission literature (construction, durables
#: manufacturing, finance most sensitive; health/education, government least).
DURATION_PROXY: dict[str, float] = {
    "1011": 0.6,  # Natural Resources and Mining
    "1012": 1.0,  # Construction (most rate-sensitive)
    "1013": 0.9,  # Manufacturing (durables)
    "1021": 0.6,  # Trade, Transportation, and Utilities
    "1022": 0.5,  # Information
    "1023": 0.8,  # Financial Activities
    "1024": 0.5,  # Professional and Business Services
    "1025": 0.2,  # Education and Health Services (least cyclical)
    "1026": 0.5,  # Leisure and Hospitality
    "1027": 0.4,  # Other Services
    "1028": 0.1,  # Public Administration (least rate-sensitive)
}


def national_sector_log_employment(qcew_cells: pl.DataFrame) -> pl.DataFrame:
    """Aggregate QCEW cells to national supersector log employment.

    Parameters
    ----------
    qcew_cells
        Cell frame with ``state_fips``, ``supersector_code``, ``date``,
        ``employment``.

    Returns
    -------
    polars.DataFrame
        Columns ``supersector_code``, ``date``, ``log_employment``.
    """
    return (
        qcew_cells.group_by(["supersector_code", "date"])
        .agg(employment=pl.col("employment").sum())
        .filter(pl.col("employment") > 0)
        .with_columns(log_employment=pl.col("employment").log())
        .sort(["supersector_code", "date"])
    )


def estimate_sigma_semielasticity(
    national: pl.DataFrame,
    policy_rate: pl.DataFrame,
    *,
    horizon: int = 12,
) -> pl.DataFrame:
    """Estimate the rate semi-elasticity of employment per supersector.

    For each supersector, regress the ``horizon``-month employment growth on the
    ``horizon``-month change in the policy rate. The (negative) coefficient is
    the semi-elasticity; sensitivity is its magnitude.

    Parameters
    ----------
    national
        National supersector log-employment frame.
    policy_rate
        Policy-rate frame (``date``, ``policy_rate``).
    horizon
        Months over which growth and the rate change are measured.

    Returns
    -------
    polars.DataFrame
        Columns ``supersector_code``, ``sigma`` (semi-elasticity),
        ``sensitivity`` (``-sigma``).
    """
    rate = (
        policy_rate.select("date", "policy_rate")
        .sort("date")
        .with_columns(
            d_rate=pl.col("policy_rate") - pl.col("policy_rate").shift(horizon)
        )
    )
    rows: list[dict[str, float | str]] = []
    for code, group in national.sort("date").group_by("supersector_code"):
        sector_code = code[0] if isinstance(code, tuple) else code
        merged = (
            group.with_columns(
                growth=pl.col("log_employment")
                - pl.col("log_employment").shift(horizon)
            )
            .join(rate, on="date", how="inner")
            .drop_nulls(["growth", "d_rate"])
        )
        if merged.height < 3 * horizon:
            continue
        pdf = merged.select(["growth", "d_rate"]).to_pandas()
        fit = sm.OLS(pdf["growth"], sm.add_constant(pdf["d_rate"])).fit()
        sigma = float(fit.params["d_rate"])
        rows.append(
            {
                "supersector_code": str(sector_code),
                "sigma": sigma,
                "sensitivity": -sigma,
            }
        )
    return pl.DataFrame(rows).sort("supersector_code")


#: NAICS 2-digit sector prefix -> QCEW supersector code (duration proxy at
#: finer NAICS levels).
NAICS2_TO_SUPERSECTOR: dict[str, str] = {
    "11": "1011",
    "21": "1011",
    "23": "1012",
    "31": "1013",
    "32": "1013",
    "33": "1013",
    "22": "1021",
    "42": "1021",
    "44": "1021",
    "45": "1021",
    "48": "1021",
    "49": "1021",
    "51": "1022",
    "52": "1023",
    "53": "1023",
    "54": "1024",
    "55": "1024",
    "56": "1024",
    "61": "1025",
    "62": "1025",
    "71": "1026",
    "72": "1026",
    "81": "1027",
    "92": "1028",
}


def duration_proxy_sigma() -> pl.DataFrame:
    """Return the documented duration/credit sensitivity proxy per supersector.

    Returns
    -------
    polars.DataFrame
        Columns ``supersector_code`` and ``sensitivity``.
    """
    return pl.DataFrame(
        {
            "supersector_code": list(DURATION_PROXY.keys()),
            "sensitivity": list(DURATION_PROXY.values()),
        }
    ).sort("supersector_code")


def duration_proxy_for_codes(codes: list[str]) -> pl.DataFrame:
    """Map sector codes (supersector or finer NAICS) to the duration proxy.

    A supersector code (e.g. ``"1013"``) maps directly; a finer NAICS code
    (e.g. 3-digit ``"331"``) maps via its 2-digit prefix to a supersector, then
    to that supersector's proxy value.

    Parameters
    ----------
    codes
        Distinct sector codes present in the panel.

    Returns
    -------
    polars.DataFrame
        Columns ``supersector_code`` (the input code) and ``sensitivity``;
        codes with no mapping are omitted.
    """
    rows: list[dict[str, float | str]] = []
    for code in sorted(set(codes)):
        if code in DURATION_PROXY:
            rows.append({"supersector_code": code, "sensitivity": DURATION_PROXY[code]})
        elif code[:2] in NAICS2_TO_SUPERSECTOR:
            rows.append(
                {
                    "supersector_code": code,
                    "sensitivity": DURATION_PROXY[NAICS2_TO_SUPERSECTOR[code[:2]]],
                }
            )
    return pl.DataFrame(rows)


def _standardize(frame: pl.DataFrame, col: str) -> pl.DataFrame:
    arr = frame[col].to_numpy().astype(np.float64)
    center = float(np.nanmean(arr))
    std = float(np.nanstd(arr, ddof=1))
    scale = std if std > 0 else 1.0
    return frame.with_columns(exposure=(pl.col(col).cast(pl.Float64) - center) / scale)


def cell_exposure(sensitivity: pl.DataFrame) -> pl.DataFrame:
    """Standardize sector sensitivity into the cell exposure shifter.

    Parameters
    ----------
    sensitivity
        Frame with ``supersector_code`` and ``sensitivity``.

    Returns
    -------
    polars.DataFrame
        Columns ``supersector_code`` and ``exposure`` (standardized sensitivity,
        mean 0 / unit sd across supersectors).
    """
    return _standardize(sensitivity, "sensitivity").select(
        "supersector_code", "exposure"
    )


def base_period_shares(qcew_cells: pl.DataFrame, base_months: int = 12) -> pl.DataFrame:
    """Predetermined base-period employment shares ``omega_{s,k}``.

    Parameters
    ----------
    qcew_cells
        Cell frame with ``state_fips``, ``supersector_code``, ``date``,
        ``employment``.
    base_months
        Number of earliest months averaged to form the base shares.

    Returns
    -------
    polars.DataFrame
        Columns ``state_fips``, ``supersector_code``, ``share``.
    """
    base_dates = sorted(qcew_cells["date"].unique().to_list())[:base_months]
    base = qcew_cells.filter(pl.col("date").is_in(base_dates))
    totals = base.group_by(["state_fips", "supersector_code"]).agg(
        emp=pl.col("employment").mean()
    )
    return (
        totals.with_columns(
            share=pl.col("emp") / pl.col("emp").sum().over("state_fips")
        )
        .select("state_fips", "supersector_code", "share")
        .sort(["state_fips", "supersector_code"])
    )


def state_exposure(shares: pl.DataFrame, sensitivity: pl.DataFrame) -> pl.DataFrame:
    """Construct state-level shift-share exposure ``E_s``.

    Parameters
    ----------
    shares
        Base-period shares from :func:`base_period_shares`.
    sensitivity
        Sector sensitivity (``supersector_code``, ``sensitivity``).

    Returns
    -------
    polars.DataFrame
        Columns ``state_fips`` and ``exposure`` (standardized across states).
    """
    joined = shares.join(sensitivity, on="supersector_code", how="inner")
    e_s = joined.group_by("state_fips").agg(
        raw=(pl.col("share") * pl.col("sensitivity")).sum()
    )
    return _standardize(e_s, "raw").select("state_fips", "exposure")
