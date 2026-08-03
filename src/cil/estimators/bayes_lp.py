"""Bayesian hierarchical local projection (PyMC).

Partial-pooling LP across supersectors: each supersector's response to the shock
is drawn from a population distribution, shrinking noisy sectors toward the
pooled mean. Unit and time fixed effects are absorbed by a two-way within
transform before sampling, so the model carries only the hierarchical slopes and
variance components. Delivers posterior IRFs (the population mean ``mu_beta``),
sector posteriors, convergence diagnostics (ArviZ), prior sensitivity, and a
posterior predictive check, for a frequentist-vs-Bayesian triangulation.

References
----------
Gelman & Hill (2007), hierarchical models; ArviZ; Jorda (2005).
"""

from __future__ import annotations

import arviz as az
import numpy as np
import numpy.typing as npt
import polars as pl
import pymc as pm
from pydantic import BaseModel

FloatArray = npt.NDArray[np.float64]


class HierarchicalLPData(BaseModel):
    """Sample sizes for a prepared hierarchical-LP design.

    Parameters
    ----------
    horizon
        Response horizon.
    n_obs
        Number of observations after differencing and dropping nulls.
    n_sectors
        Number of supersector groups pooled.
    """

    horizon: int
    n_obs: int
    n_sectors: int


def _demean_by_group(out: FloatArray, idx: npt.NDArray[np.int_], n_groups: int) -> None:
    """Subtract group means from every column of *out* in place (vectorized)."""
    counts = np.bincount(idx, minlength=n_groups).astype(np.float64)
    counts[counts == 0.0] = 1.0
    for col in range(out.shape[1]):
        sums = np.bincount(idx, weights=out[:, col], minlength=n_groups)
        out[:, col] -= (sums / counts)[idx]


def _two_way_demean(
    matrix: FloatArray,
    cell_codes: npt.NDArray[np.int_],
    time_codes: npt.NDArray[np.int_],
    *,
    iters: int = 4,
) -> FloatArray:
    """Absorb cell and time fixed effects by iterative within demeaning.

    Vectorized with ``bincount`` group sums (the pandas ``groupby.transform``
    does not scale to the 3-digit panel's ~1.5M rows x ~100 columns).
    """
    out = np.asarray(matrix, dtype=np.float64).copy()
    _, cell_idx = np.unique(cell_codes, return_inverse=True)
    _, time_idx = np.unique(time_codes, return_inverse=True)
    cell_idx = cell_idx.ravel()
    time_idx = time_idx.ravel()
    for _ in range(iters):
        _demean_by_group(out, cell_idx, int(cell_idx.max()) + 1)
        _demean_by_group(out, time_idx, int(time_idx.max()) + 1)
    return out


def prepare_design(
    panel: pl.DataFrame,
    shock: pl.DataFrame,
    *,
    shock_col: str,
    horizon: int,
) -> tuple[FloatArray, FloatArray, list[str]]:
    """Build the two-way-demeaned outcome and per-sector shock design.

    Parameters
    ----------
    panel
        Cell panel (``unit_id``, ``supersector_code``, ``date``,
        ``log_employment``).
    shock
        Shock frame (``date``, shock column).
    shock_col
        Name of the shock column.
    horizon
        Response horizon ``h``.

    Returns
    -------
    y : numpy.ndarray
        Two-way-demeaned outcome ``(n,)``.
    x : numpy.ndarray
        Two-way-demeaned per-sector shock design ``(n, n_sectors)``.
    sectors : list of str
        Supersector codes indexing the columns of ``x``.
    """
    base = (
        panel.join(shock.select("date", shock_col), on="date", how="inner")
        .sort(["unit_id", "date"])
        .with_columns(
            target=(
                pl.col("log_employment").shift(-horizon)
                - pl.col("log_employment").shift(1)
            ).over("unit_id"),
            cell_code=pl.col("unit_id").rank("dense"),
            time_code=pl.col("date").rank("dense"),
        )
        .select(["target", shock_col, "supersector_code", "cell_code", "time_code"])
        .drop_nulls()
    )
    sectors = sorted(base["supersector_code"].unique().to_list())
    shock_vals = base[shock_col].to_numpy().astype(np.float64)
    sector_arr = base["supersector_code"].to_numpy()
    design = np.zeros((base.height, len(sectors)), dtype=np.float64)
    for j, code in enumerate(sectors):
        design[:, j] = np.where(sector_arr == code, shock_vals, 0.0)
    cell_codes = base["cell_code"].to_numpy().astype(int)
    time_codes = base["time_code"].to_numpy().astype(int)
    y = _two_way_demean(
        base["target"].to_numpy().astype(np.float64).reshape(-1, 1),
        cell_codes,
        time_codes,
    ).ravel()
    x = _two_way_demean(design, cell_codes, time_codes)
    return y, x, sectors


def fit_hierarchical_lp(
    y: FloatArray,
    x: FloatArray,
    *,
    prior_sd: float = 1.0,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    seed: int = 20260101,
) -> az.InferenceData:
    """Sample the partial-pooling hierarchical LP for one horizon.

    Parameters
    ----------
    y
        Two-way-demeaned outcome.
    x
        Two-way-demeaned per-sector shock design ``(n, n_sectors)``.
    prior_sd
        Standard deviation of the priors on the population mean and the scale
        (used for prior-sensitivity analysis).
    draws, tune, chains
        NUTS sampling controls.
    seed
        Random seed.

    Returns
    -------
    arviz.InferenceData
        Posterior with ``mu_beta`` (population IRF), ``beta`` (sector
        responses), ``tau_beta``, and ``sigma``.

    Notes
    -----
    The Gaussian likelihood is expressed through the **sufficient statistics**
    ``G = X'X``, ``b = X'y`` and ``y'y`` (computed once), so each NUTS gradient
    costs O(n_sectors^2) rather than O(n * n_sectors). The posterior is exactly
    the full model's; this is what makes the 3-digit panel (~1.5M rows) tractable.
    """
    n_sectors = x.shape[1]
    gram = x.T @ x
    xty = x.T @ y
    yty = float(y @ y)
    n_obs = float(y.shape[0])
    with pm.Model():
        mu_beta = pm.Normal("mu_beta", 0.0, prior_sd)
        tau_beta = pm.HalfNormal("tau_beta", prior_sd)
        z = pm.Normal("z", 0.0, 1.0, shape=n_sectors)
        beta = pm.Deterministic("beta", mu_beta + tau_beta * z)
        sigma = pm.HalfNormal("sigma", 1.0)
        # Residual sum of squares via sufficient statistics:
        # (y - Xb)'(y - Xb) = y'y - 2 b'(X'y) + b'(X'X) b.
        rss = (
            yty
            - 2.0 * pm.math.dot(beta, xty)
            + pm.math.dot(beta, pm.math.dot(gram, beta))
        )
        log_lik = (
            -0.5 * n_obs * pm.math.log(2.0 * np.pi * sigma**2) - 0.5 * rss / sigma**2
        )
        pm.Potential("likelihood", log_lik)
        return pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,
            target_accept=0.9,
            random_seed=seed,
            progressbar=False,
        )


class PosteriorSummary(BaseModel):
    """Posterior summary and convergence diagnostics for one horizon.

    Parameters
    ----------
    horizon
        Response horizon.
    mu_mean, mu_hdi_low, mu_hdi_high
        Posterior mean and 94% HDI of the population IRF ``mu_beta``.
    tau_mean
        Posterior mean of the between-sector standard deviation.
    max_rhat
        Maximum R-hat across parameters (convergence; should be < 1.01).
    min_ess
        Minimum bulk effective sample size.
    """

    horizon: int
    mu_mean: float
    mu_hdi_low: float
    mu_hdi_high: float
    tau_mean: float
    max_rhat: float
    min_ess: float


def summarize(idata: az.InferenceData, horizon: int) -> PosteriorSummary:
    """Summarize the population IRF and convergence for one horizon."""
    summary = az.summary(idata, var_names=["mu_beta", "tau_beta", "beta", "sigma"])
    hdi = az.hdi(idata, var_names=["mu_beta"], prob=0.94)["mu_beta"].values
    post = idata.posterior
    return PosteriorSummary(
        horizon=horizon,
        mu_mean=float(post["mu_beta"].mean()),
        mu_hdi_low=float(hdi[0]),
        mu_hdi_high=float(hdi[1]),
        tau_mean=float(post["tau_beta"].mean()),
        max_rhat=float(summary["r_hat"].max()),
        min_ess=float(summary["ess_bulk"].min()),
    )


def posterior_predictive_check(
    idata: az.InferenceData, y: FloatArray
) -> dict[str, float]:
    """Compare the posterior mean of the outcome to observed moments.

    Returns
    -------
    dict of str to float
        Observed and posterior-predictive standard deviations and their ratio.
    """
    post = idata.posterior
    sigma = float(post["sigma"].mean())
    obs_sd = float(np.std(y))
    return {
        "observed_sd": obs_sd,
        "model_sigma": sigma,
        "sigma_ratio": sigma / obs_sd if obs_sd > 0 else float("nan"),
    }
