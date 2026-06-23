# ADR-0004: Headline monetary shock choice

- Status: Accepted
- Date: 2026-06-23

## Context

Phase 3 builds three monetary-shock series and the diagnostics to compare them:
an in-house Romer-Romer-style orthogonalization (real-time ALFRED vintages
standing in for Greenbook forecasts), an in-house proxy-SVAR with an external
instrument, and the published Bu-Rogers-Wu (BRW) benchmark. The headline
estimator needs one series as primary, with the others as robustness.

Diagnostics on the real sample (`shock_diagnostics`, `shock_xcorr`):

- RR orthogonalization R-squared ≈ 0.02; RR–BRW correlation ≈ 0.09 (the monthly
  in-house proxy captures different variation than BRW).
- Proxy-SVAR first-stage robust F ≈ 0.7 with BRW instrumenting the monthly
  effective-funds-rate VAR residual — **weak** by any threshold; the
  single-instrument projected shock is mechanically collinear with BRW.
- BRW predictability from the real-time information set: p ≈ 0.02, R² ≈ 0.08 —
  modest but significant, consistent with Bauer-Swanson (2023).
- Information-effect contamination (monthly proxy): ≈ 0.45 of months co-move
  with equities; the monthly proxy overstates contamination versus the
  high-frequency Jarocinski-Karadi test.

## Decision

- **Headline shock: BRW.** It spans pre/post-2008, is the most validated
  benchmark, embeds high-frequency identification, and is available across the
  cross-sectional panel window (2014–2020). It is the primary `s_t`.
- **In-house RR series: reported as robustness / cross-check.** Its low
  correlation with BRW is documented, not smoothed over.
- **Proxy-SVAR: reserved for the aggregate IRF (Phase 5),** carrying the weak
  first-stage as an explicit caveat and a weak-instrument-robust check. It is
  not used as a distinct headline shock (its single-instrument shock is collinear
  with BRW).
- **Robustness:** all three series enter the specification curve; predictability
  and information-effect contamination are reported for each. Where predictability
  is a concern, an orthogonalized variant of the shock is supplied.

## Consequences

- The headline uses a transparent, well-documented benchmark while the in-house
  series guards against benchmark-specific artifacts.
- The weak proxy-SVAR first stage is a reported finding that tempers the
  aggregate-IRF claims in Phase 5 (assumption-dependent complement, as framed).
- BRW's modest predictability is disclosed; it motivates the orthogonalized
  robustness variant rather than being suppressed.
