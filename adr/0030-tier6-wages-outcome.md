# ADR-0030: Tier 6.1 — wages as a second outcome

- Status: Accepted
- Date: 2026-09-02

## Context

The study estimated the relative effect of monetary shocks on **employment**. The
same clean cross-sectional identification can be pointed at a second labour-market
outcome — **wages** — at no data cost, since QCEW already reports quarterly total
wages and employment in the bulk zips we ingest.

## Decision

- `qcew_bulk.parse_year_wages` computes the industry-level **average weekly wage**
  across ownership as `sum(total_qtrly_wages) / (sum(mean quarterly employment) *
  13)` — the employment-weighted aggregate. Wages are quarterly (unlike the
  monthly employment series).
- `pipeline.ingest_qcew_wages` stores `qcew_wages` from the **same cached zips**
  (no new download), wired into `run`.
- `run_panel_lp` is generalised with an `outcome_col` argument (default
  `log_employment`, so existing behaviour is unchanged), letting the headline
  estimator run on any outcome.
- `cil.estimators.wages` builds a quarterly wage panel (90%-coverage cells),
  aggregates the BRW shock to quarterly, and runs the interacted panel LP on log
  average weekly wage over −4..+8 quarters (`wage_panel_lp_results`).

## Result (honest)

On the quarterly wage panel (501,348 wage rows; 4,454 cells after coverage
filtering), the relative wage effect **mirrors employment**: negative at **all 9
response horizons** (β from −0.024 on impact to −0.007 at 8 quarters, range
[−0.034, −0.007]), event-study leads clean (max |t| = 0.49), and **0 horizons
BH-significant**. A second, independent outcome gives the same **correctly-signed,
not-significant null** — strengthening the read that contractionary shocks depress
exposed-industry labour outcomes, but imprecisely.

## Consequences

- A second honest estimand with the same identification, reusing ingested data.
- The `outcome_col` generalisation makes the panel LP reusable for any future
  outcome; the wage build is a thin wrapper.
