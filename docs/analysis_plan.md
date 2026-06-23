# Analysis plan (frozen)

**Status: FROZEN. Pre-registered 2026-06-22, before any estimator was run on real
outcomes.** The claim, sign, horizons, specification list, and falsification
conditions below are fixed. They will **not** be edited to fit results. Any
deviation is logged in an ADR with justification, and both the original and the
deviated analysis are reported.

## Sign conventions (fixed)

- The monetary shock `s_t` is signed so that **`s_t > 0` denotes a contractionary
  (tightening) surprise** (the BRW benchmark convention; the in-house series are
  aligned to it).
- The outcome is the cumulative change in log employment,
  `Δ_h y_{i,t} = y_{i,t+h} - y_{i,t-1}`.
- Predetermined cell exposure `E_i ≥ 0` is increasing in interest-rate
  sensitivity, standardized to mean 0 / unit variance across cells for reporting.
- Treatment is the interaction `D_{i,t} = E_i · s_t`.
- **Headline-supporting sign: `β_h < 0`** — after a contractionary shock,
  higher-exposure cells lose *more* employment than lower-exposure cells.

## The single falsifiable claim

> Cells with higher predetermined interest-rate exposure exhibit a statistically
> larger decline in log employment over horizons `h` months following an
> identified contractionary monetary shock, with relative semi-elasticity `β_h`
> (95% CI), robust across the interacted panel-LP, LP-DiD episode, DML, and
> Bayesian-hierarchical estimators. The aggregate level effect is **not**
> point-identified by this design and is reported separately under explicit
> assumptions, with quantified fragility.

## Horizons (fixed)

- **Reported horizons:** `h = 0, 1, …, 24` (monthly; two years).
- **Pre-trend leads:** `h = -6, …, -1`, always plotted; the design is credible
  only if the leads are jointly insignificant.
- **Primary decision horizons:** `h = 12` and `h = 24`. The claim is adjudicated
  here.
- **Secondary summary:** the cumulative/average semi-elasticity over `h = 6..24`.

## Estimators (the claim must hold across all four)

1. **Headline — interacted panel local projection** with unit and time fixed
   effects; Driscoll-Kraay SEs (primary), wild cluster bootstrap by state
   (cross-check).
2. **LP-DiD** on discretized large-tightening episodes with a clean-control
   condition (validated against the Stata reference).
3. **DML / EconML** CATE with purged, time-blocked cross-fitting.
4. **Bayesian hierarchical LP** (partial pooling across cells), posterior IRFs.

## Inference and multiple testing (fixed)

- Two-sided 95% confidence intervals.
- **Benjamini-Hochberg FDR** control across horizons and across subgroups; both
  raw and adjusted results reported.
- Pre-trend leads tested jointly (event-study test).

## Pre-registered specification curve

All of the following are reported as a specification curve, not a selected table:

- **Shock series:** in-house Romer-Romer-style, proxy-SVAR external-instrument,
  and the BRW benchmark (headline shock chosen in ADR-0004).
- **Exposure definition:** baseline shift-share `E_s = Σ_k ω_{s,k} σ_k`;
  alternative `σ_k` (estimated semi-elasticity vs. documented
  duration/credit-dependence proxy); alternative base period.
- **Controls:** own lags of `y`; lagged shock; regional/macro covariates; lag
  depth `{3, 6, 12}`.
- **Sample:** baseline excludes Mar–Dec 2020; robustness adds a COVID dummy and a
  state-dependent specification.
- **Fixed effects:** two-way (unit + time) baseline; region×time as a robustness.

## Falsification conditions (fixed)

The headline claim is considered **falsified / not supported** if any of:

- `β_12` and `β_24` are **not** negative and significant (BH-FDR adjusted) in the
  headline interacted panel LP; or
- the negative, significant result does **not** replicate (same sign, overlapping
  CIs) across the LP-DiD, DML, and Bayesian estimators; or
- the event-study **leads are jointly significant**, indicating the
  conditional-parallel-trends assumption fails (the design is not credible); or
- the result is not robust across the pre-registered specification curve (sign
  flips or significance vanishes for a majority of specifications).

A precisely-estimated `β_h ≈ 0` with tight CIs is a **valid finding** (no
differential effect) and is reported as prominently as a non-zero result.

## What this design does not claim

The interacted design with time fixed effects identifies only the **relative**
(cross-sectional) effect. It does **not** identify the aggregate level effect of
monetary policy on national employment; that is reported separately (time-series
LP and proxy-SVAR) with its identifying assumptions stated and stress-tested.

## Assumptions and probes

Each identifying assumption maps to a probe in the assumptions registry
(`cil.dag.assumptions`); see `docs/methods.md`. The headline rests on
**conditional parallel trends**, tested by the event-study leads.
