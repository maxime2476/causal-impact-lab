# ADR-0014: Tier 1.2 — narrative (Romer-Romer) shock complement

- Status: Accepted
- Date: 2026-08-04

## Context

The aggregate identification so far rested on high-frequency surprises (BRW,
then the Bauer-Swanson MPS of ADR-0013). Both are market-window instruments. A
narrative / forecast-based shock provides an *independent* identification whose
assumptions do not overlap with the 30-minute window — a genuine robustness axis
for the aggregate complement.

## Decision

- Ingest the maintained, directly-downloadable **Breitenlechner (2018) update of
  Romer & Romer (2004)** narrative shocks (`cil.data.rr`, Stata `.dta`):
  intended-funds-rate changes purged of the Fed's Greenbook forecasts. Quarterly,
  1969Q1-2012Q4, three vintages — original method (`rr_org`), extended to 2008
  (`rr08`), extended to 2012 (`rr12`).
- Run a **quarterly** aggregate time-series LP (national employment aggregated to
  quarterly mean, logged) against `rr12` (the longest vintage) over 0-16 quarters
  (`build_narrative_shock_irf`, table `rr_lp_irf`). A positive shock is a
  contractionary surprise.

## Result (honest)

- The response is **correctly signed at the medium/long run**: employment turns
  negative from ~ h = 7 quarters, troughing near **-0.43% around 3-4 years**
  (163-171 quarterly obs).
- It is **imprecise** — every horizon's HAC interval comfortably includes zero
  (p > 0.5 beyond the first year). A credible, correctly-signed null.

## Consequences

- We now have two aggregate identifications that disagree on sign: the HF LP-IV
  is wrong-signed (price-puzzle-like, ADR-0013) while the narrative LP is
  right-signed but imprecise. Their disagreement is itself the finding — the
  **aggregate effect is assumption-dependent**, reinforcing that the
  cleanly-identified relative design remains the headline, not the aggregate.
- Frequency/coverage caveat: the series is quarterly and ends 2012Q4, so this
  complement does not cover the 2013-2020 tail of the panel; it is reported as a
  robustness axis, not a headline estimate.
