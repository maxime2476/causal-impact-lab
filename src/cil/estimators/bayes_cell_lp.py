"""Cell-level Bayesian hierarchical local projection (two-stage / nested).

The pooled hierarchical LP (:mod:`cil.estimators.bayes_lp`) partial-pools across
supersectors. This module goes one level finer -- to the **state x supersector
cell** -- via a tractable two-stage (meta-analytic) hierarchy:

1. **Stage 1 (per cell).** For each cell estimate its own time-series slope of
   ``h``-horizon log-employment growth on the shock, ``beta_hat_i`` with standard
   error ``se_i`` (a closed-form OLS, computed vectorized over all cells at once).
   A fully-joint cell-level design would need a dense two-way demean of a
   ~4,500-column matrix (tens of GB); the per-cell slopes avoid that entirely.
2. **Stage 2 (nested pooling).** A Bayesian normal-normal hierarchy with cells
   nested in supersectors::

       beta_hat_i ~ Normal(beta_i, se_i)          # stage-1 measurement model
       beta_i     ~ Normal(mu_sector[k(i)], tau_within)
       mu_sector  ~ Normal(mu0, tau_between)

   ``mu0`` is the grand-mean response (approximately the aggregate effect);
   ``tau_between`` / ``tau_within`` decompose the response heterogeneity into a
   **between-supersector** and a **within-supersector (across-state)** component,
   the new object this estimator delivers.

Caveat (stated, not hidden): the stage-1 slopes share the common aggregate shock,
so treating them as independent given ``beta_i`` understates the uncertainty of
``mu0``; the variance *decomposition* (``tau`` components) is the robust output
and the triangulation target, not a sharper aggregate point estimate.

References
----------
Gelman & Hill (2007), hierarchical / measurement-error models; Jorda (2005).
"""

from __future__ import annotations

import arviz as az
import numpy as np
import numpy.typing as npt
import polars as pl
import pymc as pm
from pydantic import BaseModel

FloatArray = npt.NDArray[np.float64]


def per_cell_slopes(
    panel: pl.DataFrame,
    shock: pl.DataFrame,
    *,
    shock_col: str,
    horizon: int,
    min_obs: int = 30,
) -> pl.DataFrame:
    """Stage 1: per-cell OLS slope of horizon-``h`` growth on the shock.

    Parameters
    ----------
    panel
        Cell panel (``unit_id``, ``supersector_code``, ``date``,
        ``log_employment``).
    shock
        Shock frame (``date`` and the shock column).
    shock_col
        Name of the shock column.
    horizon
        Response horizon ``h``.
    min_obs
        Minimum per-cell observations to keep a cell.

    Returns
    -------
    polars.DataFrame
        One row per cell: ``unit_id``, ``supersector_code``, ``beta_hat``,
        ``se``, ``n_obs`` -- restricted to cells with finite, positive ``se``.
    """
    base = (
        panel.join(shock.select("date", shock_col), on="date", how="inner")
        .sort(["unit_id", "date"])
        .with_columns(
            y=(
                pl.col("log_employment").shift(-horizon)
                - pl.col("log_employment").shift(1)
            ).over("unit_id"),
        )
        .select(["unit_id", "supersector_code", shock_col, "y"])
        .drop_nulls()
    )
    s = pl.col(shock_col)
    grouped = (
        base.group_by("unit_id")
        .agg(
            supersector_code=pl.col("supersector_code").first(),
            n_obs=pl.len(),
            sxx=((s - s.mean()) ** 2).sum(),
            sxy=((s - s.mean()) * (pl.col("y") - pl.col("y").mean())).sum(),
            syy=((pl.col("y") - pl.col("y").mean()) ** 2).sum(),
        )
        .filter((pl.col("n_obs") >= min_obs) & (pl.col("sxx") > 0.0))
        .with_columns(beta_hat=pl.col("sxy") / pl.col("sxx"))
        .with_columns(
            rss=(pl.col("syy") - pl.col("beta_hat") * pl.col("sxy")).clip(lower_bound=0)
        )
        .with_columns(
            se=((pl.col("rss") / (pl.col("n_obs") - 2)) / pl.col("sxx")).sqrt()
        )
        .filter(
            pl.col("se").is_finite()
            & (pl.col("se") > 0.0)
            & pl.col("beta_hat").is_finite()
        )
    )
    return grouped.select(
        "unit_id", "supersector_code", "beta_hat", "se", "n_obs"
    ).sort("unit_id")


def fit_cell_hierarchy(
    beta_hat: FloatArray,
    se: FloatArray,
    sector_idx: npt.NDArray[np.int_],
    n_sectors: int,
    *,
    prior_sd: float = 1.0,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    seed: int = 20260101,
    target_accept: float = 0.95,
) -> az.InferenceData:
    """Stage 2: sample the nested normal-normal hierarchy (non-centred).

    Parameters
    ----------
    beta_hat, se
        Stage-1 per-cell slopes and their standard errors ``(n_cells,)``.
    sector_idx
        Zero-based supersector index of each cell ``(n_cells,)``.
    n_sectors
        Number of supersectors.
    prior_sd
        Prior SD on ``mu0`` and the ``tau`` scales (prior-sensitivity lever).
    draws, tune, chains, seed, target_accept
        NUTS controls.

    Returns
    -------
    arviz.InferenceData
        Posterior with ``mu0``, ``tau_between``, ``tau_within`` and the
        supersector means ``mu_sector``.
    """
    n_cells = beta_hat.shape[0]
    with pm.Model():
        mu0 = pm.Normal("mu0", 0.0, prior_sd)
        tau_between = pm.HalfNormal("tau_between", prior_sd)
        tau_within = pm.HalfNormal("tau_within", prior_sd)
        z_sector = pm.Normal("z_sector", 0.0, 1.0, shape=n_sectors)
        mu_sector = pm.Deterministic("mu_sector", mu0 + tau_between * z_sector)
        z_cell = pm.Normal("z_cell", 0.0, 1.0, shape=n_cells)
        beta_cell = mu_sector[sector_idx] + tau_within * z_cell
        pm.Normal("obs", beta_cell, se, observed=beta_hat)
        return pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,
            target_accept=target_accept,
            random_seed=seed,
            progressbar=False,
        )


class CellHierarchySummary(BaseModel):
    """Posterior summary of the nested cell-level hierarchy for one horizon.

    Parameters
    ----------
    horizon
        Response horizon.
    n_cells, n_sectors
        Cells and supersectors in the fit.
    mu0_mean, mu0_hdi_low, mu0_hdi_high
        Grand-mean response ``mu0`` (posterior mean and 94% HDI).
    tau_between, tau_within
        Posterior-mean between-supersector and within-supersector SDs.
    between_share
        ``tau_between^2 / (tau_between^2 + tau_within^2)`` -- share of response
        heterogeneity that is between-supersector rather than across-state.
    max_rhat, min_ess
        Convergence diagnostics.
    """

    horizon: int
    n_cells: int
    n_sectors: int
    mu0_mean: float
    mu0_hdi_low: float
    mu0_hdi_high: float
    tau_between: float
    tau_within: float
    between_share: float
    max_rhat: float
    min_ess: float


def summarize_cell(
    idata: az.InferenceData, horizon: int, n_cells: int, n_sectors: int
) -> CellHierarchySummary:
    """Summarise ``mu0``, the variance decomposition, and convergence."""
    summary = az.summary(
        idata, var_names=["mu0", "tau_between", "tau_within", "mu_sector"]
    )
    hdi = az.hdi(idata, var_names=["mu0"], prob=0.94)["mu0"].values
    post = idata.posterior
    tau_b = float(post["tau_between"].mean())
    tau_w = float(post["tau_within"].mean())
    denom = tau_b**2 + tau_w**2
    return CellHierarchySummary(
        horizon=horizon,
        n_cells=n_cells,
        n_sectors=n_sectors,
        mu0_mean=float(post["mu0"].mean()),
        mu0_hdi_low=float(hdi[0]),
        mu0_hdi_high=float(hdi[1]),
        tau_between=tau_b,
        tau_within=tau_w,
        between_share=float(tau_b**2 / denom) if denom > 0 else float("nan"),
        max_rhat=float(summary["r_hat"].max()),
        min_ess=float(summary["ess_bulk"].min()),
    )


def sector_means(idata: az.InferenceData, sectors: list[str]) -> pl.DataFrame:
    """Posterior supersector mean responses (``mu_sector``) with 94% HDIs."""
    post = idata.posterior["mu_sector"]
    means = post.mean(dim=("chain", "draw")).values
    hdi = az.hdi(idata, var_names=["mu_sector"], prob=0.94)["mu_sector"].values
    return pl.DataFrame(
        {
            "supersector_code": sectors,
            "mu_sector_mean": means.astype(float),
            "hdi_low": hdi[:, 0].astype(float),
            "hdi_high": hdi[:, 1].astype(float),
        }
    ).sort("supersector_code")
