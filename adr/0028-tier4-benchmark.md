# ADR-0028: Tier 4.3 — estimator regression benchmark

- Status: Accepted
- Date: 2026-08-26

## Context

`.github/workflows/benchmark.yml` was a placeholder: it only ran the test suite
with `--durations`, and its own comment said "estimators arrive in later phases".
The estimators are here now, but nothing guarded against a change silently moving
a headline number or an estimator getting much slower.

## Decision

Add `cil.benchmark`: an estimator benchmark on a **fixed deterministic synthetic
snapshot** (40 states x 15 sectors x 156 months, seeded), so it needs no network
and no real data and runs in CI.

- Runs the headline panel-LP estimators — Driscoll-Kraay `run_panel_lp`,
  `run_panel_lp_exposure_robust`, `run_panel_lp_conley` — and records their key
  outputs (`beta_h0/6/12`, the three h=12 standard errors) and per-estimator
  wall-time.
- **Numerical regression:** the outputs are compared against a committed baseline
  (`benchmarks/baseline.json`); drift beyond tolerance (5% relative / 1e-4
  absolute, generous enough to absorb cross-platform float differences) exits
  non-zero. `--update-baseline` regenerates the baseline deliberately.
- **Performance regression:** per-estimator timings are printed (CI records them);
  wall-time is not hard-asserted because it is machine-dependent.
- `benchmark.yml` now runs `python -m cil.benchmark` (weekly + on demand).

## Consequences

- A change that moves a headline coefficient or standard error, or breaks an
  estimator, is caught by the scheduled benchmark rather than going unnoticed.
- The benchmark is fully self-contained (synthetic data, committed baseline), so
  it is reproducible and cannot be broken by data-source changes.
- The 5% tolerance targets *gross* regressions (sign flips, magnitude changes,
  crashes); it is not a bit-reproducibility check.
