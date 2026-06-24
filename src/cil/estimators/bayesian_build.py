"""Build the Bayesian hierarchical-LP results on real data and store them.

Fits the partial-pooling LP at the primary horizons, runs a prior-sensitivity
check at the primary decision horizon, performs posterior predictive checks, and
compares the posterior population IRF to the frequentist panel-LP estimate. See
``docs/results.md`` for the triangulation.

Run as a module::

    uv run python -m cil.estimators.bayesian_build
"""

from __future__ import annotations

import polars as pl

from cil.config import Settings, get_settings
from cil.data.store import Store
from cil.estimators.bayes_lp import (
    fit_hierarchical_lp,
    posterior_predictive_check,
    prepare_design,
    summarize,
)

_SHOCK_COL = "shock"
_HORIZONS = (0, 12, 24)
_PRIMARY = 12
_PRIOR_SDS = (0.5, 1.0, 2.0)
_DRAWS = 800
_TUNE = 800
_CHAINS = 2


def build_bayesian(settings: Settings | None = None) -> dict[str, float]:
    """Fit, check, and store the Bayesian hierarchical LP. Return a summary.

    Returns
    -------
    dict of str to float
        Posterior population IRF mean at each horizon, the max R-hat, and the
        prior-sensitivity spread of ``mu_beta`` at the primary horizon.
    """
    settings = settings or get_settings()
    with Store(settings.paths.store_path) as store:
        panel = store.read_table("panel_cell")
        brw = store.read_table("brw_shocks")
        freq = (
            store.read_table("panel_lp_results")
            if store.table_exists("panel_lp_results")
            else None
        )
        shock = brw.rename({"brw_monthly": _SHOCK_COL})

        summaries = []
        ppc_rows = []
        for h in _HORIZONS:
            y, x, _ = prepare_design(panel, shock, shock_col=_SHOCK_COL, horizon=h)
            idata = fit_hierarchical_lp(
                y,
                x,
                draws=_DRAWS,
                tune=_TUNE,
                chains=_CHAINS,
                seed=settings.inference.seed,
            )
            summaries.append(summarize(idata, h))
            ppc = posterior_predictive_check(idata, y)
            ppc_rows.append({"horizon": h, **ppc})

        summary_frame = pl.DataFrame([s.model_dump() for s in summaries])
        store.write_table("bayes_lp_summary", summary_frame)
        store.write_table("bayes_ppc", pl.DataFrame(ppc_rows))

        # Prior sensitivity at the primary decision horizon.
        y, x, _ = prepare_design(panel, shock, shock_col=_SHOCK_COL, horizon=_PRIMARY)
        prior_rows = []
        for prior_sd in _PRIOR_SDS:
            idata = fit_hierarchical_lp(
                y,
                x,
                prior_sd=prior_sd,
                draws=_DRAWS,
                tune=_TUNE,
                chains=_CHAINS,
                seed=settings.inference.seed,
            )
            s = summarize(idata, _PRIMARY)
            prior_rows.append(
                {"prior_sd": prior_sd, "mu_mean": s.mu_mean, "max_rhat": s.max_rhat}
            )
        prior_frame = pl.DataFrame(prior_rows)
        store.write_table("bayes_prior_sensitivity", prior_frame)

        # Frequentist-vs-Bayesian comparison.
        bayes_mu = summary_frame.select("horizon", bayes_mu=pl.col("mu_mean"))
        if freq is not None:
            comparison = bayes_mu.join(
                freq.select(pl.col("horizon").cast(pl.Int64), freq_beta=pl.col("beta")),
                on="horizon",
                how="left",
            )
            store.write_table("bayes_vs_freq", comparison)

        mu_vals = prior_frame["mu_mean"].to_numpy()
        mu_spread = float(mu_vals.max() - mu_vals.min())
        result = {f"mu_beta_h{s.horizon}": s.mu_mean for s in summaries}
        result["max_rhat"] = max(s.max_rhat for s in summaries)
        result["prior_mu_spread_h12"] = mu_spread
        return result


def main() -> None:
    """Build the Bayesian results and print a summary (entry point)."""
    for key, value in build_bayesian().items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
