"""QCEW revision-bound robustness check.

BLS does not archive state-by-industry QCEW vintages, so true real-time revisions
are unavailable (confirmed against the BLS QCEW data-availability documentation;
see ADR-0022). As a documented bound we perturb log employment by a
revision-scale shock and re-estimate the headline coefficient.

Two models are provided:

- :func:`revision_bound` -- **iid** per-observation noise. Kept for reference, but
  its bound is misleadingly tight: independent noise averages out over the ~1.5M
  cell-months, so the coefficient barely moves regardless of magnitude.
- :func:`correlated_revision_bound` -- a **benchmark-step** model that matches how
  QCEW actually revises: a persistent per-cell, per-year level step (the annual
  benchmark), plus small idiosyncratic noise. Persistent steps do *not* average
  out and are not absorbed by the cell fixed effect (they vary across years within
  a cell), so this is the honest robustness check. Its magnitude is calibrated to
  the empirical QCEW-vs-CES growth discrepancy (:func:`growth_discrepancy_sd`), a
  conservative proxy that also bakes in definitional differences.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from pydantic import BaseModel

from cil.estimators.panel_lp import PanelLPConfig, run_panel_lp


class RevisionBound(BaseModel):
    """Spread of the headline coefficient under simulated QCEW revisions.

    Parameters
    ----------
    horizon
        Response horizon.
    actual_beta
        Coefficient on the unperturbed (final) data.
    beta_min, beta_max
        Range of the coefficient across simulated-revision draws.
    revision_sd
        Standard deviation of the simulated log-employment revision.
    n_draws
        Number of perturbation draws.
    """

    horizon: int
    actual_beta: float
    beta_min: float
    beta_max: float
    revision_sd: float
    n_draws: int


def revision_bound(
    panel: pl.DataFrame,
    exposure: pl.DataFrame,
    shock: pl.DataFrame,
    *,
    shock_col: str,
    horizon: int,
    revision_sd: float = 0.003,
    n_draws: int = 50,
    seed: int = 20260101,
) -> RevisionBound:
    """Bound the headline coefficient under simulated QCEW revisions.

    Parameters
    ----------
    panel, exposure, shock
        Headline inputs.
    shock_col
        Name of the shock column.
    horizon
        Response horizon.
    revision_sd
        Standard deviation of the simulated log-employment revision (documented
        QCEW magnitude; ~0.3%).
    n_draws
        Number of perturbation draws.
    seed
        Random seed.

    Returns
    -------
    RevisionBound
        The coefficient range across simulated revisions.
    """
    rng = np.random.default_rng(seed)
    config = PanelLPConfig(horizons=(horizon,))
    actual = float(
        run_panel_lp(panel, exposure, shock, config, shock_col=shock_col)["beta"][0]
    )
    betas: list[float] = []
    log_emp = panel["log_employment"].to_numpy()
    for _ in range(n_draws):
        perturbed = panel.with_columns(
            log_employment=pl.Series(
                log_emp + rng.normal(0.0, revision_sd, size=log_emp.size)
            )
        )
        betas.append(
            float(
                run_panel_lp(perturbed, exposure, shock, config, shock_col=shock_col)[
                    "beta"
                ][0]
            )
        )
    draws = np.asarray(betas)
    return RevisionBound(
        horizon=horizon,
        actual_beta=actual,
        beta_min=float(draws.min()),
        beta_max=float(draws.max()),
        revision_sd=revision_sd,
        n_draws=n_draws,
    )


def growth_discrepancy_sd(qcew_supersector: pl.DataFrame, ces: pl.DataFrame) -> float:
    """Std of the QCEW-vs-CES 12-month log-growth discrepancy (revision scale).

    A data-grounded, conservative calibration for the revision magnitude: it is
    the dispersion of the difference between QCEW and CES supersector employment
    growth, which exceeds true QCEW revisions (it also contains definitional
    differences between the two sources).

    Parameters
    ----------
    qcew_supersector, ces
        QCEW and CES supersector panels (``state_fips``, ``supersector_code``,
        ``date``, ``employment``).

    Returns
    -------
    float
        Standard deviation of ``qcew_growth - ces_growth``.
    """
    joined = (
        ces.select("state_fips", "supersector_code", "date", ces_emp="employment")
        .join(
            qcew_supersector.select(
                "state_fips", "supersector_code", "date", qcew_emp="employment"
            ),
            on=["state_fips", "supersector_code", "date"],
            how="inner",
        )
        .filter((pl.col("ces_emp") > 0) & (pl.col("qcew_emp") > 0))
        .sort(["state_fips", "supersector_code", "date"])
        .with_columns(
            ces_g=(pl.col("ces_emp").log() - pl.col("ces_emp").log().shift(12)).over(
                ["state_fips", "supersector_code"]
            ),
            qcew_g=(pl.col("qcew_emp").log() - pl.col("qcew_emp").log().shift(12)).over(
                ["state_fips", "supersector_code"]
            ),
        )
        .drop_nulls(["ces_g", "qcew_g"])
    )
    diff = (joined["qcew_g"] - joined["ces_g"]).to_numpy()
    return float(np.std(diff))


class CorrelatedRevisionBound(BaseModel):
    """Headline-coefficient spread under a correlated (benchmark-step) revision.

    Parameters
    ----------
    horizon
        Response horizon.
    actual_beta
        Coefficient on the unperturbed (final) data.
    beta_min, beta_max, beta_sd
        Range and standard deviation of the coefficient across revision draws.
    sigma_bench
        Per-cell, per-year benchmark level-step standard deviation (calibrated).
    sigma_idio
        Idiosyncratic monthly noise standard deviation.
    n_draws
        Number of revision draws.
    """

    horizon: int
    actual_beta: float
    beta_min: float
    beta_max: float
    beta_sd: float
    sigma_bench: float
    sigma_idio: float
    n_draws: int


def correlated_revision_bound(
    panel: pl.DataFrame,
    exposure: pl.DataFrame,
    shock: pl.DataFrame,
    *,
    shock_col: str,
    horizon: int,
    sigma_bench: float,
    sigma_idio: float = 0.002,
    n_draws: int = 40,
    seed: int = 20260101,
) -> CorrelatedRevisionBound:
    """Bound the headline coefficient under benchmark-step QCEW revisions.

    Each draw adds a revision ``r_{i,t} = b_{i, year(t)} + eta_{i,t}`` to log
    employment, where ``b`` is a per-cell, per-year benchmark level step
    (``N(0, sigma_bench)``) and ``eta`` is idiosyncratic monthly noise
    (``N(0, sigma_idio)``), then re-estimates the headline coefficient.

    Parameters
    ----------
    panel, exposure, shock
        Headline inputs.
    shock_col
        Name of the shock column.
    horizon
        Response horizon.
    sigma_bench
        Benchmark level-step SD (e.g. ``growth_discrepancy_sd`` / sqrt(2)).
    sigma_idio
        Idiosyncratic monthly-noise SD.
    n_draws
        Number of revision draws.
    seed
        Random seed.

    Returns
    -------
    CorrelatedRevisionBound
        The coefficient spread across benchmark-step revisions.
    """
    rng = np.random.default_rng(seed)
    config = PanelLPConfig(horizons=(horizon,))
    actual = float(
        run_panel_lp(panel, exposure, shock, config, shock_col=shock_col)["beta"][0]
    )
    log_emp = panel["log_employment"].to_numpy()
    _, unit_idx = np.unique(panel["unit_id"].to_numpy(), return_inverse=True)
    years = panel["date"].dt.year().to_numpy()
    _, year_idx = np.unique(years, return_inverse=True)
    unit_idx = unit_idx.ravel().astype(int)
    year_idx = year_idx.ravel().astype(int)
    n_cells = int(unit_idx.max()) + 1
    n_years = int(year_idx.max()) + 1
    betas: list[float] = []
    for _ in range(n_draws):
        bench = rng.normal(0.0, sigma_bench, size=(n_cells, n_years))
        revision = bench[unit_idx, year_idx] + rng.normal(
            0.0, sigma_idio, size=log_emp.size
        )
        perturbed = panel.with_columns(log_employment=pl.Series(log_emp + revision))
        betas.append(
            float(
                run_panel_lp(perturbed, exposure, shock, config, shock_col=shock_col)[
                    "beta"
                ][0]
            )
        )
    draws = np.asarray(betas)
    return CorrelatedRevisionBound(
        horizon=horizon,
        actual_beta=actual,
        beta_min=float(draws.min()),
        beta_max=float(draws.max()),
        beta_sd=float(draws.std()),
        sigma_bench=sigma_bench,
        sigma_idio=sigma_idio,
        n_draws=n_draws,
    )
