# ADR-0027: Tier 4.2 — refresh path (and the DVC decision)

- Status: Accepted
- Date: 2026-08-26

## Context

`cil.data.pipeline` implemented only the *backfill* path: each raw payload is
fetched once and cached, and re-runs reuse the cache. There was no way to pull
newly released data — the refresh path was explicitly left unimplemented. The
Tier 4 plan also listed "DVC" for data versioning.

## Decision — refresh path

Add `force_refresh` through the pipeline:

- `_cache_or_fetch(..., force_refresh=False)` re-fetches and overwrites the cache
  when `True`, instead of serving the cached bytes.
- Every `ingest_*` and `pipeline.run(force_refresh=...)` thread the flag.
- **QCEW** re-fetches only its two most recent years (the quarters QCEW revises);
  older years' large zips stay cached — a genuinely incremental refresh.
- CLIs: `uv run python -m cil.data.pipeline --refresh` and
  `uv run python -m cil.orchestrate --refresh`.

**PIT integrity is preserved.** Refreshing only pulls newer raw payloads; the
strict-PIT series still record every observation's vintage and reconstruct via
`alfred.as_of(V)` with no look-ahead (ADR-0002). A refresh never rewrites history,
it only appends newer vintages/periods.

## Decision — DVC (not adopted)

DVC was considered for data versioning and rejected, for two concrete reasons:

1. **The design is a single shared DuckDB.** Every stage writes into
   `data/cil.duckdb`; DVC's model is one set of file `outs` per stage, so mapping
   the DAG onto DVC would require either splitting the store into per-stage files
   (a large, low-value refactor) or a single coarse stage (no granularity).
2. **No data remote.** DVC's value is content-addressed storage backed by a
   remote; none is available here.

The reproducibility need those would serve is already met: the **provenance
ledger** records, for every raw pull, the source, URL, retrieval timestamp,
vintage date, SHA-256 and byte count; the environment is fully locked
(`uv.lock`); and `cil.orchestrate` runs the whole DAG deterministically. If a data
remote is added later, a single-stage `dvc.yaml` over `data/cil.duckdb` is the
natural drop-in.

## Consequences

- The study can be updated with new data releases in one command, without
  re-downloading the full ~1990-onward QCEW history.
- Reproducibility rests on the provenance ledger + locked env + orchestrator,
  documented here rather than on an unused DVC configuration.
