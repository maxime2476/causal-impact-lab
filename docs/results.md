# Results

This page reports results honestly, including nulls and imprecise estimates, as
prominently as positive findings. It grows phase by phase; numbers are
regenerable with `uv run python -m cil.estimators.build`.

## Headline relative effect (Phase 4, preliminary)

The interacted panel local projection (unit + time fixed effects, BRW headline
shock, estimated shift-share exposure) on the 2014-2020 state-by-supersector
panel does **not** support the headline claim on the currently available sample.

| Horizon | beta_h | sign expected | significant (BH-FDR) |
|---|---|---|---|
| h = 0 | -0.11 | negative | no (p ~ 0.41) |
| h = 12 | +0.19 | negative | no (wrong sign) |
| h = 24 | +0.10 | negative | no (wrong sign) |

- **No response horizon survives BH-FDR** (adjusted p ~ 0.996 across `h = 0..24`).
- A **marginal pre-trend at `h = -2`** (t ~ -2.2) flags the
  conditional-parallel-trends assumption; the lead test is not clean.

Per the frozen `analysis_plan.md`, this is a **null / non-supportive** result.
The claim, sign, horizons, and falsification conditions were fixed before
estimation and are not revised to fit this outcome.

### Interpreting the null (caveats, not excuses)

- The QCEW open-data API limits the cross-sectional panel to **2014-2020**;
  pre-2014 history needs the bulk-file extension (a documented follow-up).
- Exposure varies only across **11 supersectors**, limiting cross-sectional
  power.
- **COVID months remain in the baseline** here; they are excluded only in the
  Phase 8 robustness, and the 2020 shock dominates the short window.

These are addressed by later phases (aggregate IRF, DML heterogeneity, Bayesian
partial pooling, and the COVID/break robustness). The headline number stands as
reported until then.

## Method notes

- Inference: Driscoll-Kraay standard errors (cross-sectional + serial robust),
  with a wild-cluster-bootstrap cross-check; BH-FDR across horizons.
- LP-DiD is implemented and validated on synthetic staggered panels but is not
  the headline for a single national shock (no staggered timing); see ADR-0005.
