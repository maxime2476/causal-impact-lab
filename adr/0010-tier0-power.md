# ADR-0010: Tier 0 — restoring statistical power

- Status: Accepted (in progress, one section per commit)
- Date: 2026-06-30

## Context

The v0.1.0 headline is a precisely-stated null, but it is **power-limited**: the
QCEW open-data API restricts the cross-sectional panel to 2014–2020, and the
supersector level gives only 11 distinct exposure values. Tier 0 addresses the
root cause before any other refinement.

## Decisions

### 0.1 — Pre-2014 history via QCEW bulk flat files (this section)

- Ingest the annual `by_area` bulk zips (`cil.data.qcew_bulk`) instead of the
  2014+ API for the cell panel. One ~400 MB download per year (cached), reaching
  back to **1990** (NAICS reconstructed); the effective panel is bounded below by
  the shock series (BRW from 1994), giving **1994–2020**.
- The bulk path is aggregation-level configurable, so the same cached zips serve
  both the supersector panel and the 3-digit panel (section 0.2) without
  re-downloading.
- Ownership components are summed and disclosure suppression flagged, identical
  to the API path. NAICS revisions (2002/2007/2012/2017) are a documented caveat;
  3-digit codes are largely stable across them.

### 0.2 — NAICS 3-digit headline granularity (planned)

Switch `qcew.aggregation_level` to 55 (~90 sectors) to multiply cross-sectional
exposure variation; manage the heavier disclosure suppression with the coverage
threshold.

### 0.3 — CES-SAE supersector reconciliation (planned)

A longer, seasonally-adjusted alternative panel to triangulate QCEW.

## Consequences

- The cell panel gains ~20 years and the 1990s/2000s tightening cycles — the
  main lever on the headline's power.
- Storage/bandwidth: ~10 GB of cached bulk zips under the gitignored data root;
  re-runs are incremental.
- The frozen analysis plan and its claim are unchanged; only the sample coverage
  improves. Any change to the headline is a power effect, reported as such.
