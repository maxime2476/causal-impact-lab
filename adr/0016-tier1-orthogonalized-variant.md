# ADR-0016: Tier 1.4 — orthogonalized shock variant (MPS_ORTH)

- Status: Accepted
- Date: 2026-08-05

## Context

The Bauer-Swanson workbook ships both the raw surprise (`MPS`) and an
**orthogonalized** surprise (`MPS_ORTH`) purged of variation predictable from a
pre-FOMC information set. Sections 1.1 and 1.3 showed the raw MPS is a strong but
information-contaminated instrument (price-puzzle-like aggregate IRF). Tier 1.4
uses `MPS_ORTH` as the predictability-robust variant and reports the tradeoff.

## Decision

- Add an LP-IV variant instrumented by `MPS_ORTH` (`lpiv_irf_orth` in
  `cil.estimators.aggregate`), alongside the raw-MPS headline and BRW robustness.
- Add a predictability comparison in `cil.shocks.build`: run the Bauer-Swanson
  predictability test on both `MPS` and `MPS_ORTH` against the real-time ALFRED
  macro lags, storing R-squared and the joint p-value for each.

## Result (honest)

**Aggregate LP-IV.** Orthogonalization flips the aggregate IRF to the
theory-consistent sign at the cost of instrument strength:

| Instrument | First-stage F | theta_12 | Sign |
|---|---|---|---|
| Raw MPS (1.1) | 13-15 (strong) | +4.03 | wrong (price puzzle) |
| **MPS_ORTH** | **1.6-2.8 (weak)** | **-0.63** | **correct** |

Every horizon of `lpiv_irf_orth` is negative (h0 -1.9, h12 -0.6, h24 -11.8), but
the first stage is below the weak-instrument threshold throughout. Removing the
predictable/information component — the same component flagged in ADR-0015 — is
what corrects the sign, consistent with the price puzzle being driven by
information contamination. The gain in plausibility costs identification.

**Predictability.** Against our real-time macro lags, `MPS_ORTH` is only modestly
less predictable than `MPS` (R-squared 0.058 vs 0.065); both remain jointly
significant. This is expected: `MPS_ORTH` is orthogonalized to Bauer-Swanson's
*own* predictor set, not our ALFRED macro lags, so it stays partly predictable
against ours. Reported plainly, not overstated.

## Consequences

- We now bracket the aggregate rate response: strong-but-contaminated (raw MPS,
  wrong-signed) and clean-but-weak (MPS_ORTH, right-signed, imprecise). Neither is
  a reliable point estimate — the aggregate stays an assumption-dependent
  complement, and the relative design remains the headline.
- Closes Tier 1 (identification): HF instrument (1.1), narrative shock (1.2), HF
  information effect (1.3), orthogonalized variant (1.4).
