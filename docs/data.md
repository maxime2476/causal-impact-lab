# Data layer

This page documents every data source, the point-in-time (PIT) policy, the
zero-lower-bound (ZLB) splice, and disclosure suppression. All figures in the
project are computed on real data; synthetic data appears only in tests.

## Sources

| Series | Source | Access | Frequency | PIT treatment |
|---|---|---|---|---|
| National employment (`PAYEMS`) | FRED/ALFRED | API (key) | monthly | **strict PIT** (vintages) |
| Macro confounders (`CPIAUCSL`, `PCEPI`, `INDPRO`, `UNRATE`) | FRED/ALFRED | API (key) | monthly | **strict PIT** |
| Oil price (`MCOILWTICO`) | FRED | API (key) | monthly | as-published |
| Policy rate (`FEDFUNDS`) + Wu-Xia shadow rate | FRED + Atlanta Fed | API + `.xlsx` | monthly | as-published; spliced |
| State × supersector employment | BLS QCEW | open-data API (CSV) | monthly | **revised (final)**, documented |
| State employment cross-check | BLS CES-SAE via FRED (`{ABBR}NA`) | API (key) | monthly (SA) | as-published |
| Published shock benchmark | Bu-Rogers-Wu | author CSV | monthly / per-FOMC | as-published |

Exact identifiers and URLs live in `cil.config` (`DataConfig`). Each raw pull is
cached under `data/raw/<source>/` with a provenance record (source, URL,
retrieval timestamp, vintage date, SHA-256, byte count) in the DuckDB
`provenance` table.

## Point-in-time policy

The policy is **scoped, not uniform** (and is stated as such — see ADR-0002).

- **Strict PIT** on national employment and the macro confounders that are
  materially revised (`PAYEMS`, `CPIAUCSL`, `PCEPI`, `INDPRO`, `UNRATE`).
  ALFRED returns the full revision history; each observation's `realtime_start`
  is treated as a *vintage date*, giving a long
  `(series_id, reference_date, vintage_date, value)` frame. From it,
  `cil.data.alfred.as_of(pit, V)` reconstructs the series as known on date `V`
  using only vintages on or before `V` — no look-ahead. The magnitude of
  revisions is real and large: e.g. `PAYEMS` for 2008-01 was first released as
  138,102k and later revised to 138,391k, a ~289k difference. A property test
  and a unit test enforce the no-look-ahead invariant.
- **Relaxed PIT** on the QCEW state × supersector panel. BLS publishes no
  historical vintage archive at state×industry granularity, and QCEW
  preliminary→final revisions are minor (near-census coverage). We use the
  final/revised values, documented here as a deliberate compromise — **not**
  full PIT. A bounding robustness check on the period where state-level QCEW
  vintages exist is planned for the robustness phase.

## QCEW details

- **Granularity.** State × NAICS *supersector* (`agglvl_code == 53`, ~11
  supersectors) minimises disclosure suppression relative to finer NAICS levels.
  The eleven supersectors span Natural Resources & Mining through Public
  Administration.
- **Ownership.** QCEW breaks supersectors out by ownership (federal, state,
  local, private); there is no Total-Covered row at supersector level. We sum
  the ownership components into a total per (state, supersector, month).
- **Suppression.** If any ownership component is withheld under disclosure rules,
  the summed total is a lower bound and the cell-month is flagged `suppressed`.
  Partial suppression of small government components is common and rarely affects
  the dominant private component; the panel's drop decision therefore uses
  positive-employment coverage (a cell needs employment > 0 in at least
  `coverage_min_fraction` of months), and the per-cell suppression footprint is
  logged to `qcew_suppression`.
- **Seasonality.** QCEW NAICS files are **not seasonally adjusted**. The panel
  carries NSA employment; seasonal handling (controls vs. adjustment) is an
  explicit estimation-stage decision and is not mixed in silently here.
- **Coverage window.** The QCEW open-data **API serves only 2014 onward**;
  earlier NAICS history exists solely in bulk flat-file archives (100–250 MB per
  year). The cross-sectional panel is therefore built for 2014–2020 (within the
  configured sample), spanning the 2015–2018 tightening cycle and the 2020
  shock. Extending pre-2014 via the bulk files is a documented follow-up. The
  aggregate/shock series are unaffected and retain their full history.

## ZLB splice

The monthly policy rate equals the effective federal funds rate (EFFR) away from
the ZLB and the **Wu-Xia shadow rate** within the configured ZLB windows
(2008-12–2015-12 and 2020-03–2022-02), gated behind `use_shadow_rate`. Where a
shadow value is unavailable it falls back to the EFFR. The Atlanta Fed suspended
shadow-rate updates in March 2022; coverage to early 2022 spans the ZLB windows
after which the EFFR applies.

## CES-SAE scope

The CES-SAE cross-check uses state **total-nonfarm** SA employment from FRED's
mirror (`{ABBR}NA`), converted from thousands to persons to match QCEW. It
cross-checks state aggregates; supersector-level CES↔QCEW reconciliation is
deferred to the cross-check robustness work.

## Reproducing the panels

```bash
uv run python -m cil.data.pipeline
```

This builds, on real data, the analysis-ready tables in `data/cil.duckdb`:
`macro_pit`, `macro_current`, `policy_rate`, `brw_shocks`, `ces_sae`,
`qcew_cells`, `qcew_suppression`, `panel_cell`, `panel_dropped_cells`, and
`provenance`. Cached raw payloads make re-runs incremental.
