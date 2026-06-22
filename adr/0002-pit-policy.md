# ADR-0002: Point-in-time policy and the data layer

- Status: Accepted
- Date: 2026-06-22

## Context

The analysis must avoid look-ahead bias: an observation may use only data that
was available at its reference date. Full real-time (point-in-time, PIT)
treatment is feasible where vintage archives exist and revisions matter, but not
everywhere. We must state, per source, exactly what PIT treatment applies and
why, so the compromise is auditable rather than hidden.

## Decision

### Scoped PIT, not uniform

- **Strict PIT** for the national outcome and materially-revised macro
  confounders (`PAYEMS`, `CPIAUCSL`, `PCEPI`, `INDPRO`, `UNRATE`), sourced from
  ALFRED. The full revision history is stored as
  `(series_id, reference_date, vintage_date, value)`; `as_of(V)` reconstructs
  the series as known on `V` using only vintages on or before `V`. The
  no-look-ahead invariant is enforced by a unit test and a Hypothesis property
  test.
- **Relaxed PIT** for the QCEW state × supersector panel: no state×industry
  vintage archive exists, and QCEW revisions are minor (near-census). Final
  values are used, documented as a compromise, with a bounding revision check
  planned for the robustness phase.
- **As-published** for the oil price, the policy rate / shadow rate, the CES-SAE
  cross-check, and the BRW benchmark.

### Source access and ownership summation

- FRED/ALFRED via the documented REST API (key from the environment, never
  committed and never written to provenance). QCEW via the open-data CSV API.
  Wu-Xia from the Atlanta Fed workbook; BRW from the authors' CSV; CES-SAE state
  totals from FRED's mirror (`{ABBR}NA`).
- QCEW supersectors are published by ownership; total employment per
  (state, supersector, month) is the sum across ownership components, with a
  suppression flag when any component is withheld.

### QCEW coverage window

The QCEW open-data API serves only 2014 onward. The cross-sectional panel is
built for 2014–2020 (within the configured sample), which still spans the
2015–2018 tightening cycle. Pre-2014 extension via bulk flat files is recorded
as a follow-up; the aggregate and shock series are unaffected.

### Seasonality

QCEW NAICS data is carried not-seasonally-adjusted; the seasonal-handling choice
is made explicitly at the estimation stage, never mixed silently.

### Contracts, storage, provenance

Every boundary frame is validated against a `pandera` (polars) schema; a
violation is a hard error. Analysis-ready tables and a provenance ledger live in
DuckDB; raw payloads are cached on disk with a content hash.

## Consequences

- Results that depend on the macro series are robust to revision by
  construction; the QCEW relaxation is bounded and disclosed, not silent.
- The headline cross-sectional sample is shorter (2014–2020) than the aggregate
  sample until the bulk-file extension is implemented; this is a data-access
  limitation, stated plainly, not a methodological choice.
- Strict typing plus schema contracts make ingestion failures loud and local.
