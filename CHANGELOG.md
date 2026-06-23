# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims to
adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Repository scaffold: `uv`-managed environment with a committed lockfile, `src`
  layout for the `cil` package, and typed configuration.
- Quality tooling: `ruff` (format + lint), `mypy` (strict), `pytest` with
  coverage, and a `pre-commit` configuration mirroring the CI gates.
- Continuous integration workflows: `ci`, `benchmark`, and `docs`.
- Attribution-policy enforcement via commit-message and staged-content scanners.
- Documentation site skeleton (`mkdocs-material`) and ADR-0001.
- Data layer: point-in-time ALFRED ingestion (national + macro confounders),
  QCEW state-by-supersector employment with suppression logging, Wu-Xia / EFFR
  policy-rate splice, Bu-Rogers-Wu shock series, and a CES-SAE state cross-check.
- Schema contracts (`pandera`) at every dataset boundary, a DuckDB store with a
  provenance ledger, and an end-to-end ingestion pipeline.
- `docs/data.md` and ADR-0002 documenting the scoped point-in-time policy.
- Causal DAG (`cil.dag`) for the headline design, DoWhy identification yielding
  the backdoor set, a refuter battery, and an assumptions registry linking each
  identifying assumption to its probe.
- Frozen `docs/analysis_plan.md` (the falsifiable claim, sign, horizons, spec
  list, and falsification conditions), `docs/methods.md`, and ADR-0003.
