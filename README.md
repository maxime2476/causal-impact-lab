# Causal Impact Lab

Estimating the causal effect of contractionary US monetary policy shocks on US
employment. The goal is an honest answer, not a headline number: null and
imprecise results are reported as prominently as positive ones.

**Live app:** <https://huggingface.co/spaces/maxime2476/causal-impact-lab>

## What this measures

Two estimands, in a deliberate hierarchy:

1. **Headline — relative effect (cleanly identified).** Do US states and
   industries with greater predetermined interest-rate exposure lose more
   employment after an identified contractionary monetary shock? This is
   identified off cross-sectional heterogeneity, with time fixed effects
   absorbing the aggregate component.
2. **Complement — aggregate dynamic effect (assumption-dependent).** The impulse
   response of national employment to an identified shock, reported separately
   with its identifying assumptions stated and stress-tested — never presented as
   "the" answer.

All published results are computed on real data (FRED/ALFRED, BLS QCEW and
CES-SAE, the Wu-Xia shadow rate, and the Bu-Rogers-Wu shock series). Synthetic
data is used only inside the test suite.

## Project layout

```
src/cil/        analysis library (import name: cil)
docs/           sources, methods, analysis plan, results
adr/            architecture decision records
app/            Streamlit delivery surface
tests/          unit, property, golden, and synthetic-DGP tests
```

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync --all-extras      # create the locked environment
uv run pytest             # run the test suite
uv run ruff check .       # lint
uv run mypy               # strict type check
pre-commit install        # enable local quality + policy hooks
```

Quality gates (ruff, mypy `--strict`, pytest) run in CI on every push and pull
request. The exact resolved dependency versions are recorded in
[`docs/environment.md`](docs/environment.md).

## License

MIT — see [`LICENSE`](LICENSE).
