# ADR-0026: Tier 4.1 — end-to-end orchestration

- Status: Accepted
- Date: 2026-08-26

## Context

The analysis was run as nine independent `python -m` entry points
(`cil.data.pipeline`, `cil.shocks.build`, `cil.estimators.build`,
`cil.estimators.aggregate`, `cil.estimators.heterogeneity`,
`cil.estimators.bayesian_build`, `cil.robustness.ces_reconciliation`,
`cil.robustness.build`, `cil.report.export`). Reproducing the full study meant
running them by hand in the right order, with the dependency DAG living only in
maintainers' heads.

## Decision

Add `cil.orchestrate`: a single entry point that runs the stages in the correct
dependency order, timing and logging each.

- Canonical order: `data -> ces_reconciliation -> shocks -> aggregate ->
  estimators -> heterogeneity -> bayesian -> robustness -> report`. CES
  reconciliation precedes robustness because the calibrated revision bound
  (ADR-0022) reads `ces_supersector`.
- `run_all(settings, *, stages=None, stage_map=None)` runs all stages or a named
  subset; a subset always executes in canonical order regardless of the order
  requested. `stage_map` is injectable so the ordering/selection logic is unit
  tested without running the real (network/compute-heavy) stages.
- CLI: `uv run python -m cil.orchestrate [--stages a,b,c] [--list]`.

## Consequences

- The full study is reproducible with one command, and a single stage can be
  re-run after upstream tables exist (e.g. `--stages robustness`).
- The dependency DAG is now explicit and enforced in code; adding a stage is a
  one-line registry entry in the right position.
- The orchestrator does not re-implement any stage — it composes the existing
  build functions, so per-stage behaviour is unchanged.
