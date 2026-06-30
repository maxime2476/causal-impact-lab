# ADR-0012: CES-SAE supersector reconciliation (Tier 0.3)

- Status: Accepted
- Date: 2026-06-30

## Context

The headline panel is QCEW (administrative, near-census, NSA). An independent
cross-check against the CES State-and-Area survey (sample-based, seasonally
adjusted) validates the QCEW state-by-supersector employment used for exposure
and the panel.

## Decision

- Ingest **state × supersector seasonally-adjusted All-Employees** from the BLS
  `sm` flat files. State×supersector CES is access-constrained (FRED carries it
  inconsistently; the BLS API caps unregistered use at 25 queries/day), so the
  **per-state `sm.data.*` files** are used: the target series are identified from
  the `sm.series` catalog (statewide, data type 01, seasonal S, supersector
  aggregate) and matched in each per-state data file by `series_id`.
- Map the **11 CES supersectors to QCEW supersectors**
  (`CES_TO_QCEW_SUPERSECTOR`); Mining & Logging ↔ Natural Resources & Mining and
  Government ↔ Public Administration are the looser correspondences.
- Reconcile by aggregating the 3-digit QCEW panel up to supersectors (NAICS
  2-digit crosswalk) and computing, per (state, supersector), the correlation of
  log levels and of year-on-year growth between CES and QCEW.

## Consequences

- A documented validation of the QCEW panel; high correlations support the
  administrative source, divergences are flagged honestly.
- CES values (thousands) are converted to persons; CES is SA while QCEW is NSA,
  so the **growth** correlation is the more comparable metric (levels differ in
  seasonality and in survey vs. census scope).
- Adds the `ces_supersector` and `ces_qcew_reconciliation` tables; per-state
  `sm.data` files are cached under the gitignored data root.
