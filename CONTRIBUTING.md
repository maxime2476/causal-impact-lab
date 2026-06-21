# Contributing

## Environment

```bash
uv sync --all-extras
pre-commit install
```

`pre-commit install` enables both the `pre-commit` and `commit-msg` hooks.

## Workflow

- Work on a short-lived feature branch; open one pull request into `main` and
  squash-merge. Never push to `main` directly.
- One roadmap phase per branch (see the project contract). No scope creep.
- Every decision that is not already specified in the project contract is
  recorded as an ADR under `adr/`.

## Quality gates

All of the following must pass locally and in CI before merge:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

- Full type hints; `mypy --strict` is a gate.
- numpydoc docstrings on every public function and class; estimators cite the
  paper and equation they implement.
- Tests favour correctness of inference over raw coverage: property-based tests,
  positive controls, placebo/permutation checks, and leakage tests.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/), imperative mood,
focused scope. Commit-message and staged-content scanners enforce the project's
attribution policy; commits that trip them are rejected locally.

## Dependencies

Declare floors in `pyproject.toml`; exact versions are fixed by `uv.lock`. Do
not hard-pin in `pyproject.toml`. If two libraries conflict, raise it rather
than silently downgrading a core dependency.
