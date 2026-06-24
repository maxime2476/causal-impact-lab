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
- Monetary shocks (`cil.shocks`): in-house Romer-Romer orthogonalization on
  real-time vintages, a proxy-SVAR external-instrument first stage with a
  weak-instrument flag, information-effect (Jarocinski-Karadi monthly proxy) and
  predictability (Bauer-Swanson) tests, and cross-correlation against the BRW
  benchmark, with results stored and ADR-0004 recording the headline-shock choice.
- Secret redaction in HTTP error handling so the FRED API key never appears in
  raised messages.
- Shift-share interest-rate exposure (estimated semi-elasticity + documented
  duration proxy), the headline interacted panel local projection with
  event-study leads, Driscoll-Kraay and wild-cluster-bootstrap inference, BH-FDR
  adjustment, an LP-DiD port (validated by TWFE equivalence) with a
  Goodman-Bacon diagnostic, and ADR-0005.
- `docs/results.md` reporting the preliminary headline as a pre-registered null.
- Aggregate complement: time-series local projection IRF (HAC) and an LP-IV
  proxy-SVAR IRF with first-stage F and Anderson-Rubin weak-instrument-robust
  intervals, with honest assumption-dependence framing in `results.md` and
  ADR-0006.
- DML heterogeneity: a purged/embargoed time-blocked cross-fitting splitter
  (with no-leakage tests), EconML LinearDML and CausalForestDML CATE estimation,
  a placebo refutation, and ADR-0007.
- Bayesian pillar: a PyMC hierarchical partial-pooling local projection
  (two-way FE absorbed, non-centred), posterior IRFs with ArviZ convergence
  diagnostics, prior-sensitivity analysis, posterior predictive checks, a
  frequentist-vs-Bayesian comparison, and ADR-0008.
