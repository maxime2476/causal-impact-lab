# ADR-0020: Tier 2.4 — CausalForest heterogeneity drivers

- Status: Accepted
- Date: 2026-08-13

## Context

The DML step fit a CausalForestDML with a **single** effect modifier
(`exposure`), so it could not say *which* predetermined cell characteristics
drive the treatment-effect heterogeneity. Tier 2.4 adds a multi-feature driver
analysis.

## Decision

- `build_driver_sample` builds a multi-column effect-modifier matrix `X` of
  predetermined cell characteristics: `exposure` (sector rate-sensitivity),
  `base_share` (within-state employment share) and `log_base_emp` (cell size).
- `estimate_cate_drivers` fits a CausalForestDML on that `X` (purged time-blocked
  cross-fitting, as elsewhere) and reports driver strength.

**Driver measure — BLP, not impurity importances.** Tree `feature_importances_`
are biased toward high-cardinality features (a continuous covariate offers far
more split points than a sector shifter with ~100 values), so they mis-rank
drivers — verified on a synthetic design where they crowned a pure-noise
continuous feature over the true driver. Instead we report the **Best Linear
Predictor** of the estimated CATE on the *standardised* modifiers: a coefficient
per one-SD move, comparable across features and sign-informative. A normalised
`|BLP|` gives a relative driver share. Stored in `cate_drivers`.

## Result (honest)

At h = 12 on the real panel, the CausalForest finds **weak** CATE heterogeneity
(forest ATE ~ 0.026; mean CATE 0.009 in the top exposure tercile vs 0.024 in the
bottom). The standardised BLP driver strengths:

| Feature | BLP coef (per 1 SD) | driver share |
|---|---|---|
| `log_base_emp` (cell size) | +0.030 | 0.43 |
| `base_share` (within-state share) | +0.025 | 0.35 |
| `exposure` (sector sensitivity) | −0.016 | 0.23 |

Every coefficient is tiny relative to the ATE, so there is **no strong,
cleanly-attributable driver**. Cell size and within-state share are marginally
the stronger *linear* predictors; exposure is the weakest of the three — expected,
because exposure is already embedded in the treatment `T = exposure * shock`, so
conditional on `T` it adds little. This is consistent with the cell-level Bayesian
decomposition (ADR-0019): the systematic response lives in the sector-exposure
design itself (already in `T`), leaving only weak residual heterogeneity for the
other characteristics to explain.

## Consequences

- The heterogeneity is attributed to interpretable, predetermined characteristics
  with a cardinality-unbiased measure, consistent with the cell-level Bayesian
  finding (ADR-0019) that the response is an industry-composition effect.
- The impurity-importance pitfall is documented so it is not reintroduced.
