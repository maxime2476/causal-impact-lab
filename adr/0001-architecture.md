# ADR-0001: Architecture and project conventions

- Status: Accepted
- Date: 2026-06-21

## Context

The project estimates the causal effect of contractionary US monetary policy
shocks on US employment across several estimators (panel local projections,
LP-DiD, time-series LP, proxy-SVAR, double/debiased ML, and a Bayesian
hierarchical LP). The work is delivered incrementally, one roadmap phase per
pull request, and must be reproducible, strictly typed, and honest about
uncertainty. This record fixes the foundational technical choices so later
phases build on a stable base.

## Decision

### Language and environment

- Python 3.12, managed with `uv`. Dependency floors are declared in
  `pyproject.toml`; exact versions are pinned by a committed `uv.lock`. The
  resolved set is recorded in `docs/environment.md`.

### Package layout

- `src` layout with import package `cil`. One concern per module: I/O is
  isolated in `cil.data`; estimators are pure functions over typed inputs that
  return typed result objects. Subpackages mirror the analysis pipeline
  (`data`, `shocks`, `dag`, `exposure`, `estimators`, `inference`,
  `robustness`, `report`).

### Data and storage

- `polars` is the primary dataframe library; `duckdb` is the analysis-ready
  store. `pandas` is used only where downstream econometrics libraries require
  it. Dataset boundaries are guarded by schema contracts.

### Configuration

- Centralised, typed configuration via `pydantic-settings` (`cil.config`). No
  magic numbers, paths, thresholds, horizons, or dates in estimator code.

### Quality gates

- `ruff` (format + lint), `mypy` in strict mode, and `pytest` are required CI
  gates on every push and pull request. numpydoc docstrings are required on
  public APIs. A `pre-commit` configuration mirrors these gates locally.

### Testing philosophy

- Correctness of inference over coverage: property-based tests, positive
  controls on synthetic DGPs (confined to `tests/`), placebo/permutation
  checks, and leakage tests for time-respecting cross-fitting. No network in
  unit tests.

### Delivery

- `streamlit` is the interactive surface; `mkdocs-material` is the documentation
  site. FastAPI is out of scope unless explicitly requested.

### Process

- Feature branch per phase, one pull request into `main`, squash-merge. Each
  non-obvious decision is captured in a new ADR. An attribution policy is
  enforced by commit-message and staged-content scanners.

## Consequences

- The heavy causal stack (`econml`, `dowhy`, `pymc`) constrains the resolvable
  dependency set; conflicts are surfaced rather than resolved by silently
  downgrading a core library.
- The `src` layout requires an editable install (handled by `uv sync`) for tests
  to import `cil`.
- Strict typing imposes upfront cost on third-party libraries lacking stubs;
  these are allow-listed for missing imports in the `mypy` configuration.
