# ADR-0009: Robustness suite

- Status: Accepted
- Date: 2026-06-26

## Context

Phase 8 stress-tests the headline relative effect: a specification curve, placebo
/ permutation tests, structural-break detection, COVID handling, and a QCEW
revision bound. The aim is to show the full distribution of estimates and the
fragility of any conclusion, not to find a favourable specification.

## Decision

- **Specification curve** (`cil.robustness.specification_curve`): the headline
  panel LP is re-estimated across the pre-registered grid -- shock series
  (BRW, in-house RR), exposure shifter (estimated semi-elasticity, documented
  duration proxy), control-lag depth (3, 6, 12), and COVID-sample handling (full,
  Mar-Dec 2020 excluded) -- at the primary decision horizons, with BH-FDR control
  across specifications. The whole curve is reported.
- **Placebo / permutation** (`cil.robustness.placebo`): the shock is permuted
  across time and exposure across supersectors; the permutation p-value is the
  share of placebo coefficients at least as large in magnitude as the actual.
- **Structural breaks** (`cil.robustness.breaks`): Bai-Perron multiple breaks via
  the `ruptures` PELT algorithm with a BIC-style penalty.
- **COVID** (`cil.robustness.covid`): the baseline excludes Mar-Dec 2020 (with
  time fixed effects a COVID dummy is absorbed, so sample exclusion is the
  meaningful lever); an Auerbach-Gorodnichenko state-dependent aggregate LP
  separates expansion and recession responses.
- **QCEW revision bound** (`cil.robustness.qcew_revision`): since BLS does not
  archive state-by-industry vintages, the headline is re-estimated under
  simulated log-employment revisions of the documented (small) magnitude; the
  coefficient spread is reported as a **simulated** bound, not real vintages.

## Consequences

- Conclusions are stated against the full specification curve and the placebo
  null, not a single estimate; this is the project's guard against
  specification search.
- `ruptures` is added as a dependency (pinned in the lockfile) for Bai-Perron.
- The QCEW bound is a documented simulation, consistent with the relaxed-PIT
  compromise in ADR-0002; real state-vintage data would supersede it.
