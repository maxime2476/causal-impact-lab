# Results

This page reports results honestly, including nulls and imprecise estimates, as
prominently as positive findings. It grows phase by phase; numbers are
regenerable with `uv run python -m cil.estimators.build`.

## Headline relative effect (Phase 4, preliminary)

The interacted panel local projection (unit + time fixed effects, BRW headline
shock, estimated shift-share exposure) on the 2014-2020 state-by-supersector
panel does **not** support the headline claim on the currently available sample.

| Horizon | beta_h | sign expected | significant (BH-FDR) |
|---|---|---|---|
| h = 0 | -0.11 | negative | no (p ~ 0.41) |
| h = 12 | +0.19 | negative | no (wrong sign) |
| h = 24 | +0.10 | negative | no (wrong sign) |

- **No response horizon survives BH-FDR** (adjusted p ~ 0.996 across `h = 0..24`).
- A **marginal pre-trend at `h = -2`** (t ~ -2.2) flags the
  conditional-parallel-trends assumption; the lead test is not clean.

Per the frozen `analysis_plan.md`, this is a **null / non-supportive** result.
The claim, sign, horizons, and falsification conditions were fixed before
estimation and are not revised to fit this outcome.

### Interpreting the null (caveats, not excuses)

- The QCEW open-data API limits the cross-sectional panel to **2014-2020**;
  pre-2014 history needs the bulk-file extension (a documented follow-up).
- Exposure varies only across **11 supersectors**, limiting cross-sectional
  power.
- **COVID months remain in the baseline** here; they are excluded only in the
  Phase 8 robustness, and the 2020 shock dominates the short window.

These are addressed by later phases (aggregate IRF, DML heterogeneity, Bayesian
partial pooling, and the COVID/break robustness). The headline number stands as
reported until then.

## Aggregate dynamic effect (Phase 5, complement)

The aggregate IRF is the **assumption-dependent complement** to the headline
relative effect: it requires the shock to be exogenous to the aggregate state of
the economy. It is reported separately and never as "the" answer.

**Time-series LP** (national log employment on the BRW shock, full 1994-2020
sample, Newey-West SEs): the response is positive on impact and turns negative
over the medium run, consistent with a contractionary shock reducing employment.

| Horizon | theta (employment, % per unit shock) | p-value |
|---|---|---|
| h = 0 | +2.5 | 0.12 |
| h = 12 | -6.6 | 0.10 |
| h = 24 | -6.5 | 0.12 |

The medium-run effects have the expected sign but are only marginally
significant.

**LP-IV proxy-SVAR** (employment response to a +1pp policy-rate increase,
instrumented by BRW): point estimates are negative at the medium run
(theta_12 ~ -4), but the **first stage is weak** (robust F ~ 4-5, below any
threshold), so the Anderson-Rubin weak-instrument-robust intervals are very wide
(e.g. h = 12 ~ [-30, +1]). The rate-scaled IRF is therefore **not reliably
identified** on this sample; this is reported, not hidden, and follows directly
from the weak BRW first stage documented in ADR-0004.

**Bottom line:** the aggregate complement is *suggestive* of a contractionary
employment decline at the medium run but is imprecise, and the structural
rate-response is weakly identified. It does not overturn or substitute for the
headline relative design.

## Heterogeneity via DML (Phase 6)

Double/debiased ML (EconML) with **purged time-blocked cross-fitting** estimates
the average and exposure-heterogeneous effect of the interaction. The placebo
(treatment permuted across time) collapses to ~0 at every horizon, so the
estimates are not spurious -- but the effect itself is **not robust**:

| Horizon | LinearDML ATE (95% CI) | CausalForest ATE | placebo |
|---|---|---|---|
| h = 0 | -0.15 ([-0.25, -0.06]) | -1.40 | +0.03 |
| h = 12 | +0.07 ([+0.02, +0.13]) | -0.61 | -0.00 |
| h = 24 | +0.17 ([+0.09, +0.26]) | +0.09 | -0.02 |

- The LinearDML effect is significantly negative on impact (the expected sign)
  but **flips to significantly positive** at h = 12 and h = 24.
- LinearDML and CausalForestDML **disagree sharply** (e.g. -0.15 vs -1.40 at
  h = 0), indicating instability / unmodelled nonlinearity on the short sample.
- Placebos are ~0 throughout: the refutation passes (no spurious effect), but the
  sign instability and estimator disagreement mean the heterogeneity evidence
  **does not support** the headline claim either.

This corroborates the pre-registered null: across the panel-LP and DML estimators
the relative effect is not robustly negative on the 2014-2020 sample.

## Bayesian hierarchical LP (Phase 7)

A PyMC partial-pooling LP (supersector responses shrunk toward a population mean,
two-way FE absorbed) provides a Bayesian read with full posterior uncertainty.
All fits converged (max R-hat = 1.00); the posterior predictive matches the
outcome dispersion (sigma ratio ~ 1.0); and the population IRF is robust to the
prior (mu_beta spread ~ 0.06 across prior SDs {0.5, 1, 2}).

| Horizon | Bayesian mu_beta (94% HDI) | Frequentist beta_h |
|---|---|---|
| h = 0 | -0.01 ([-1.80, +1.86]) | -0.11 |
| h = 12 | +0.03 ([-2.03, +1.63]) | +0.19 |
| h = 24 | +0.01 ([-1.99, +1.74]) | +0.10 |

- The Bayesian population IRF is **centred near zero with very wide credible
  intervals** at every horizon -- no robust relative effect.
- The posterior means are **shrunk versions of the frequentist estimates** (same
  signs, smaller magnitudes); between-sector heterogeneity ``tau`` is modest
  (0.13-0.35).

**Triangulation:** the frequentist panel-LP, the DML estimators, and the Bayesian
hierarchical LP **agree** -- the relative effect is not robustly identified on the
2014-2020 sample. The agreement across very different inference paradigms
strengthens the pre-registered null.

## Method notes

- Inference: Driscoll-Kraay standard errors (cross-sectional + serial robust),
  with a wild-cluster-bootstrap cross-check; BH-FDR across horizons.
- LP-DiD is implemented and validated on synthetic staggered panels but is not
  the headline for a single national shock (no staggered timing); see ADR-0005.
