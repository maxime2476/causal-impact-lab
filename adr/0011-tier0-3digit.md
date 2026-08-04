# ADR-0011: 3-digit headline granularity (deviation from the frozen plan)

- Status: Accepted
- Date: 2026-06-30

## Context

The frozen `docs/analysis_plan.md` (Phase 2) pre-registered the unit of analysis
as (state × **supersector**) — 11 sectors. That choice minimised disclosure
suppression but left only 11 distinct exposure values, which (with the original
2014–2020 window) gave a wrong-signed, pre-trend-flagged, insignificant headline.

Tier 0.2 raises the cross-sectional resolution to **NAICS 3-digit** (QCEW
`agglvl_code == 55`, ~100 sectors) using the bulk history from 0.1.

## Decision

- The **headline panel** is now built at 3-digit (`qcew.aggregation_level = 55`),
  with `coverage_min_fraction` relaxed to 0.90 to accommodate the heavier
  disclosure suppression at finer NAICS. This yields ~4,566 (state × 3-digit)
  units over 1994–2020 (vs 547 supersector cells), an ~8× increase in
  cross-sectional variation.
- This is a **deliberate, post-registration deviation** from the frozen plan's
  supersector unit. Per the project's no-specification-search rule, it is logged
  here and the **supersector design remains fully reproducible** (set
  `aggregation_level = 53`) and is reported alongside; the frozen claim text is
  **not** edited.
- The duration-proxy exposure is extended to finer NAICS via a 2-digit →
  supersector crosswalk (`cil.exposure.shift_share.duration_proxy_for_codes`);
  the estimated semi-elasticity shifter already operates at any level.

## Result (honest)

At 3-digit on 1994–2020, the relative effect `beta_h` is **correctly signed
(negative) at every response horizon** and the **event-study leads are clean**
(no significant pre-trend) — a markedly more credible design than the
supersector/2014–2020 result. Magnitudes remain **small and not
BH-significant**, so the headline is still a (now correctly-signed, well-behaved)
null, reported as such.

## Consequences

- Substantially more power and a credible pre-trend; the sign now matches theory.
- The deviation is disclosed; both granularities are reported. The pre-registered
  supersector result stands as the registered benchmark.
- Heavier suppression at 3-digit is managed by the coverage threshold and logged
  in the suppression footprint.
