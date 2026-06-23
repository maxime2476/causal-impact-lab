# ADR-0006: Aggregate impulse responses (the complement)

- Status: Accepted
- Date: 2026-06-23

## Context

The headline estimand is the cleanly-identified relative effect; the contract
also requires an aggregate dynamic effect as an explicit, assumption-dependent
*complement*. Phase 5 implements it and frames its fragility honestly.

## Decision

- **Time-series local projection** (`cil.estimators.ts_lp`): the IRF of national
  log employment to the identified shock, with Newey-West (HAC) standard errors,
  estimated on the full sample (the shock and PAYEMS span 1994-2020, unlike the
  QCEW-limited cross-sectional panel).
- **Proxy-SVAR via LP-IV** (`cil.estimators.proxy_svar`): the employment response
  to a policy-rate increase, instrumenting the rate change with the borrowed BRW
  high-frequency shock. Reported with the first-stage F and an **Anderson-Rubin
  weak-instrument-robust** interval, which stays valid under weak identification.
- **Headline shock = BRW** (ADR-0004); the in-house RR series is available as a
  robustness shifter.

## Result and framing

- The time-series LP shows a positive impact response turning negative over the
  medium run (theta_12 ~ -6.6, p ~ 0.10; theta_24 ~ -6.5), the expected sign but
  only marginally significant.
- The LP-IV first stage is **weak** (robust F ~ 4-5); AR intervals are wide
  (h = 12 ~ [-30, +1]), so the rate-scaled IRF is not reliably identified. This
  is a direct, disclosed consequence of the weak BRW first stage.
- The aggregate effect is reported **separately and as a complement**, never as
  the project's answer. Its identifying assumption (shock exogeneity to the
  aggregate state) is stated and is probed by the shock diagnostics (Phase 3).

## Consequences

- The aggregate IRF adds suggestive, honestly-caveated evidence without
  contaminating the headline relative design.
- The weak-instrument finding tempers any structural rate-elasticity claim and
  motivates reporting AR intervals rather than Wald intervals for the LP-IV.
- A stronger external instrument (a dedicated high-frequency surprise series)
  would tighten the LP-IV; this is noted as a possible future enhancement.
