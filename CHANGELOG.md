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
