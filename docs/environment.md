# Environment

Python **3.12** (CI and local interpreter: CPython 3.12.13), managed with
[`uv`](https://docs.astral.sh/uv/). Dependency *floors* are declared in
`pyproject.toml`; *exact* versions are fixed by the committed `uv.lock` (166
packages resolved). Regenerate this page after any dependency change:

```bash
uv lock && uv sync --all-extras
```

## Resolution notes

The resolver selected newer major versions than some prose elsewhere implies;
per the project's version policy the lockfile is authoritative. Notably the
resolved set includes `pandas` 3.x, `pymc` 6.x (`pytensor` 3.x), `arviz` 1.x,
`mypy` 2.x, and `pytest` 9.x. The heavy causal stack (`econml`, `dowhy`, `pymc`)
co-resolved without conflict.

## Resolved versions (key packages)

| Package | Version |
|---|---|
| numpy | 2.4.6 |
| scipy | 1.15.3 |
| pandas | 3.0.3 |
| polars | 1.41.2 |
| duckdb | 1.5.4 |
| statsmodels | 0.14.6 |
| linearmodels | 7.0 |
| arch | 8.0.0 |
| econml | 0.16.0 |
| dowhy | 0.14 |
| pymc | 6.0.1 |
| pytensor | 3.0.7 |
| arviz | 1.2.0 |
| httpx | 0.28.1 |
| requests | 2.34.2 |
| fredapi | 0.5.2 |
| pandera | 0.32.0 |
| pydantic | 2.13.4 |
| pydantic-settings | 2.14.2 |
| streamlit | 1.58.0 |
| mkdocs-material | 9.7.6 |
| mkdocstrings | 1.0.4 |
| pytest | 9.1.1 |
| hypothesis | 6.155.7 |
| ruff | 0.15.18 |
| mypy | 2.1.0 |
| pre-commit | 4.6.0 |

The complete, hash-pinned set lives in `uv.lock`.
