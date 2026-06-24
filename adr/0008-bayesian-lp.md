# ADR-0008: Bayesian hierarchical local projection

- Status: Accepted
- Date: 2026-06-24

## Context

Phase 7 adds the Bayesian pillar: a partial-pooling local projection that
shrinks noisy sector responses toward a population mean and delivers full
posterior uncertainty, for a frequentist-vs-Bayesian triangulation of the
headline relative effect.

## Decision

- **Model** (`cil.estimators.bayes_lp`): per horizon, the supersector responses
  to the shock are partially pooled, ``beta_k ~ Normal(mu_beta, tau_beta)`` with
  a non-centred parameterization. Unit and time fixed effects are absorbed by a
  two-way within transform *before* sampling, so the model carries only the
  hierarchical slopes and variance components -- keeping NUTS tractable on the
  full panel.
- **Pooling level:** supersector (the level at which exposure and the treatment
  vary). ``mu_beta`` is the population IRF; ``beta_k`` the shrunk sector
  responses.
- **Horizons:** the full Bayesian fit is run at h = 0, 12, 24 (impact plus the
  two primary decision horizons).
- **Diagnostics:** ArviZ R-hat and effective sample size; ``target_accept = 0.9``
  to limit divergences.
- **Prior sensitivity:** the population/scale prior standard deviation is varied
  over {0.5, 1.0, 2.0} at the primary horizon; the spread of the posterior
  ``mu_beta`` is reported.
- **Posterior predictive check:** the model error scale is compared to the
  observed outcome dispersion.
- **Triangulation:** the posterior ``mu_beta`` is compared to the frequentist
  panel-LP ``beta_h``.

## Consequences

- The Bayesian and frequentist estimators are compared on the same design; a
  documented disagreement (or agreement) is part of the results.
- Absorbing the fixed effects before sampling trades a small approximation in the
  two-way within transform for a model that samples in seconds-to-minutes rather
  than being intractable with hundreds of unit and time parameters.
- Partial pooling stabilizes noisy sector estimates but inherits the sample's
  overall power limits; the posterior intervals are honest about this.
