# ADR-0005: Headline estimator, exposure, and inference

- Status: Accepted
- Date: 2026-06-23

## Context

Phase 4 implements the headline relative-effect estimator and its inference, the
shift-share exposure, the LP-DiD port, and a Goodman-Bacon diagnostic. Several
choices were not fixed by the contract and are recorded here, along with an
honest account of what the estimator finds on the available sample.

## Decision

### Exposure shifter sigma_k (two variants)

- **Headline:** the *estimated* semi-elasticity of national supersector
  employment to the policy/shadow rate. Estimating it on national aggregates
  (separate variation from the state-by-supersector panel) and on the *rate*
  (not the shock) limits mechanical circularity. Cell exposure is the
  standardized sensitivity.
- **Robustness:** a *documented* duration/credit-dependence proxy
  (`DURATION_PROXY`), entering the specification curve.

### Headline estimator

Interacted panel local projection with unit and time fixed effects (per ADR-0003
and the frozen analysis plan). Horizon -1 is the event-study reference (its
outcome is identically zero) and is omitted; pre-trend leads use fixed effects
only because a lead long-difference is mechanically collinear with the
lagged-difference controls. Inference is Driscoll-Kraay (`cov_type="kernel"`),
with a wild-cluster-bootstrap cross-check available; response-horizon p-values
are BH-FDR adjusted.

### LP-DiD

The clean-control LP-DiD is implemented as a reusable, citable port and
validated by analytic / two-way-FE equivalence on synthetic staggered panels (a
drop-in slot for Stata golden fixtures is left in `tests/golden`). **It is not
applied as the headline on the monetary panel:** a single national shock has no
staggered cross-sectional treatment *timing*, so a binarised episode treatment is
rank-deficient (the adoption indicator is collinear with the time effects). The
continuous interacted panel LP is the appropriate headline; LP-DiD remains for
genuinely staggered designs and for the documented Goodman-Bacon comparison.

## Result on the available sample (honest, pre-registered)

On the 2014-2020 panel with the BRW headline shock and estimated exposure, the
relative effect is **not supported**: at the primary decision horizons
`beta_12 = +0.19` and `beta_24 = +0.10` are wrong-signed, and **no response
horizon survives BH-FDR** (adjusted p ~ 0.996); `beta_0 = -0.11` has the
expected sign but is insignificant. A marginal pre-trend appears at `h = -2`
(t ~ -2.2), flagging the conditional-parallel-trends assumption. Per the frozen
analysis plan this is a null / non-supportive result, reported as prominently as
a positive one. Likely contributors (not excuses): the QCEW-API-limited
2014-2020 window, exposure variation only across 11 supersectors, and COVID
months still in the baseline (excluded only in the Phase 8 robustness).

## Consequences

- The headline is reported as a null with a pre-trend caveat; the analysis plan
  is not edited to fit it.
- LP-DiD is shipped and tested but scoped to staggered designs, with its
  inapplicability to the simultaneous national shock documented rather than
  forced.
- Power limitations motivate the Phase 5-8 complements (aggregate IRF, DML,
  Bayesian pooling, COVID handling, and the pre-2014 QCEW bulk extension).
