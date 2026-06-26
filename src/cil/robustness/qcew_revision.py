"""QCEW revision-bound robustness check.

BLS does not archive state-by-industry QCEW vintages, so true real-time revisions
are unavailable. As a documented bound, we perturb log employment by the
documented (small, near-census) QCEW revision magnitude and re-estimate the
headline coefficient, reporting the resulting spread. This is a *simulated* bound,
not real vintages -- stated as such (see ``docs/data.md`` and ADR-0002).
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
