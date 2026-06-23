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

## Aggregate dynamic effect (Phase 5, complement)

The aggregate IRF is the **assumption-dependent complement** to the headline
relative effect: it requires the shock to be exogenous to the aggregate state of
the economy. It is reported separately and never as "the" answer.

**Time-series LP** (national log employment on the BRW shock, full 1994-2020
sample, Newey-West SEs): the response is positive on impact and turns negative
over the medium run, consistent with a contractionary shock reducing employment.

| Horizon | theta (employment, % per unit shock) | p-value |
|---|---|---|
| h = 0 | +2.5 | 0.12 |
| h = 12 | -6.6 | 0.10 |
| h = 24 | -6.5 | 0.12 |

The medium-run effects have the expected sign but are only marginally
significant.

**LP-IV proxy-SVAR** (employment response to a +1pp policy-rate increase,
instrumented by BRW): point estimates are negative at the medium run
(theta_12 ~ -4), but the **first stage is weak** (robust F ~ 4-5, below any
threshold), so the Anderson-Rubin weak-instrument-robust intervals are very wide
(e.g. h = 12 ~ [-30, +1]). The rate-scaled IRF is therefore **not reliably
identified** on this sample; this is reported, not hidden, and follows directly
from the weak BRW first stage documented in ADR-0004.

**Bottom line:** the aggregate complement is *suggestive* of a contractionary
employment decline at the medium run but is imprecise, and the structural
rate-response is weakly identified. It does not overturn or substitute for the
headline relative design.

## Method notes

- Inference: Driscoll-Kraay standard errors (cross-sectional + serial robust),
  with a wild-cluster-bootstrap cross-check; BH-FDR across horizons.
- LP-DiD is implemented and validated on synthetic staggered panels but is not
  the headline for a single national shock (no staggered timing); see ADR-0005.
