# ADR-0015: Tier 1.3 — high-frequency information-effect test

- Status: Accepted
- Date: 2026-08-04

## Context

Phase 3 shipped a Jarocinski-Karadi information-effect test using a **monthly
proxy** (month-long equity returns standing in for the announcement window),
noting it likely *overstates* contamination. With the Bauer-Swanson `mps_fomc`
table (ADR-0013) we now have the rate surprise and the S&P 500 move measured in
the **same high-frequency window**, so the true "poor man's sign restriction" is
finally reproducible on real data.

## Decision

- Add `info_effect.classify_high_frequency`, applying the sign co-movement rule
  to `mps_fomc` (rate surprise `mps` vs same-window `sp500`). An event is an
  *information* shock when the two co-move (a tightening that lifts equities).
- Run, in `cil.shocks.build`, both the HF test and a monthly-proxy test of the
  **same** MPS series, so the only difference is the window — isolating the
  proxy's overstatement.
- Emit a decontaminated instrument `mps_clean` (monthly sum of the
  policy-component surprises, information events zeroed) for downstream use.

## Result (honest)

On the real sample:

| Test | Window | Contamination share |
|---|---|---|
| MPS monthly proxy | month-long equity | 52.4% |
| BRW monthly proxy | month-long equity | 44.9% |
| **MPS high-frequency** | intraday announcement | **32.5%** (105 / 323 events) |

The monthly proxy overstates contamination by ~20 pp: month-long equity moves are
dominated by non-FOMC news, so many surprises are mislabelled as information. The
HF share (~ one third) matches the Jarocinski-Karadi finding that a sizeable
minority of FOMC surprises carry a central-bank information component.

## Consequences

- The confounding "information effect" is real but smaller than the monthly proxy
  suggested; `mps_clean` gives a decontaminated instrument for robustness.
- This is a diagnostic on the *aggregate* identification; it does not change the
  headline relative design, whose identification does not rest on the HF window.
