# Results

This page reports results honestly, including nulls and imprecise estimates, as
prominently as positive findings. It grows phase by phase; numbers are
regenerable with `uv run python -m cil.estimators.build`.

## Bottom line

**The pre-registered headline claim is not supported on the available sample.**
Across four estimators -- the interacted panel local projection, the LP-DiD
machinery, double/debiased ML, and a Bayesian hierarchical LP -- the relative
employment response to interest-rate exposure after a contractionary shock is
**not robustly negative and significant** at the decision horizons (h = 12, 24),
and the specification curve, placebo tests, and prior-sensitivity analysis all
concur. A precisely-stated null is the finding.

The aggregate complement is *suggestive* of a medium-run contractionary
employment decline (clearer once the recession/COVID state is separated), but it
is assumption-dependent and weakly identified (a weak external instrument), so it
neither rescues nor substitutes for the cleanly-identified relative design.

The most important caveat is data coverage: the QCEW open-data API limits the
cross-sectional panel to **2014-2020**, a short window dominated by a single
tightening cycle and the pandemic, with exposure variation across only 11
supersectors. The pre-2014 bulk-file extension (documented in ADR-0002) is the
clearest path to more power. The claim, sign, horizons, and falsification
conditions were frozen before estimation and were **not** revised to fit any of
this.

## Update — Tier 0: extended 3-digit panel (1994–2020)

Post-v0.1.0, the cell panel was extended back to 1994 (QCEW bulk flat files) and
refined to **NAICS 3-digit** (~4,566 state × sector cells vs 547 supersector
cells; see ADR-0010/0011). The headline interacted panel LP on this larger,
finer panel is **markedly more credible**, though still a null in significance:

| Horizon | beta_h (3-digit, 1994–2020) | BH p | vs. registered (supersector, 2014–2020) |
|---|---|---|---|
| h = 0 | **-0.025** | 0.48 | -0.11 |
| h = 12 | **-0.021** | 0.77 | +0.19 (wrong sign) |
| h = 24 | **-0.012** | 0.83 | +0.10 (wrong sign) |

- **All 25 response horizons are now negative** (the expected sign), versus
  wrong-signed positives in the registered window.
- The **event-study leads are clean** (max |t| ≈ 1.3, none significant) — the
  marginal pre-trend that flagged the registered design is gone.
- Magnitudes remain small and **not BH-significant**: a correctly-signed,
  well-behaved **null**, reported as such.

This is a deliberate, disclosed deviation from the frozen plan (supersector →
3-digit, ADR-0011); the registered supersector result below stands as the
pre-registered benchmark and was not edited to fit this.

**Downstream triangulation on the 3-digit panel** (re-run at full settings; the
Bayesian was reformulated to sufficient statistics to scale — same posterior,
ADR-0008):

- **DML** (`dml_results`): the LinearDML effect is **significantly negative on
  impact** (h=0: −0.016, 95% CI [−0.021, −0.011]) with CausalForest agreeing in
  sign, drifting toward zero/positive at longer horizons; placebos ≈ 0.
- **Bayesian** (`bayes_vs_freq`): the population IRF is negative at every horizon
  (μ_β −0.004/−0.018/−0.006), a shrunk version of the frequentist β_h — the two
  paradigms now **agree in sign** (both negative), where they disagreed before.
- **Specification curve**: the median β is now **negative** at both decision
  horizons (h=12: −0.002, 50% of specs negative; h=24: −0.004, 67% negative),
  versus a positive-leaning minority in the registered window; still 0% BH-
  significant.
- **QCEW revision bound**: negligible (β ∈ [−0.021, −0.021]); **CES growth
  correlation 0.92** validates the panel (Tier 0.3).

**Net:** the power work (1994–2020, 3-digit) turns the headline from a
wrong-signed, pre-trend-flagged result into a **correctly-signed, clean-pre-trend,
cross-estimator-consistent null** — a materially more credible finding, still
short of conventional significance on this sample.

## Headline relative effect (Phase 4, preliminary; pre-registered benchmark)

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

**Tier 1.1 update — stronger instrument.** Replacing the BRW instrument with the
Bauer-Swanson high-frequency surprise (SF Fed) strengthens the LP-IV first stage
from robust F ~ 3-5 to **~13-15** at h = 0/12 (ADR-0013). Identification is now
strong, but the point IRF becomes **wrong-signed at the medium run** (a
price-puzzle-like `theta_12 > 0`) — an economically implausible result that
*reinforces* treating the aggregate as an assumption-dependent complement rather
than the answer. BRW is retained as a robustness variant.

**Tier 1.2 update — narrative-shock complement.** A second, independent aggregate
identification uses the updated Romer-Romer narrative shock (Breitenlechner 2018;
quarterly, forecast-purged intended-funds-rate changes) in a quarterly employment
LP (`rr_lp_irf`, ADR-0014). Here the response is **correctly signed** — employment
turns negative from ~ 7 quarters and troughs near **-0.43% around 3-4 years** —
but **imprecise**, with every HAC interval including zero. The two aggregate
identifications therefore *disagree on sign* (HF wrong-signed, narrative
right-signed but insignificant); the disagreement is the point — the aggregate is
assumption-dependent, and the cleanly-identified relative design stays the
headline.

**Tier 1.3 update — high-frequency information effect.** The Jarocinski-Karadi
test, run on the true announcement window (`mps_fomc` rate vs same-window S&P 500)
rather than a monthly proxy, finds **32.5% of FOMC surprises are information-type**
(105 / 323 events) — versus 52% for the monthly proxy of the same series, which is
inflated by non-FOMC equity news (ADR-0015). The information effect is real but
smaller than the proxy implied; a decontaminated instrument (`mps_clean`) is
retained for robustness.

## Heterogeneity via DML (Phase 6)

Double/debiased ML (EconML) with **purged time-blocked cross-fitting** estimates
the average and exposure-heterogeneous effect of the interaction. The placebo
(treatment permuted across time) collapses to ~0 at every horizon, so the
estimates are not spurious -- but the effect itself is **not robust**:

| Horizon | LinearDML ATE (95% CI) | CausalForest ATE | placebo |
|---|---|---|---|
| h = 0 | -0.15 ([-0.25, -0.06]) | -1.40 | +0.03 |
| h = 12 | +0.07 ([+0.02, +0.13]) | -0.61 | -0.00 |
| h = 24 | +0.17 ([+0.09, +0.26]) | +0.09 | -0.02 |

- The LinearDML effect is significantly negative on impact (the expected sign)
  but **flips to significantly positive** at h = 12 and h = 24.
- LinearDML and CausalForestDML **disagree sharply** (e.g. -0.15 vs -1.40 at
  h = 0), indicating instability / unmodelled nonlinearity on the short sample.
- Placebos are ~0 throughout: the refutation passes (no spurious effect), but the
  sign instability and estimator disagreement mean the heterogeneity evidence
  **does not support** the headline claim either.

This corroborates the pre-registered null: across the panel-LP and DML estimators
the relative effect is not robustly negative on the 2014-2020 sample.

## Bayesian hierarchical LP (Phase 7)

A PyMC partial-pooling LP (supersector responses shrunk toward a population mean,
two-way FE absorbed) provides a Bayesian read with full posterior uncertainty.
All fits converged (max R-hat = 1.00); the posterior predictive matches the
outcome dispersion (sigma ratio ~ 1.0); and the population IRF is robust to the
prior (mu_beta spread ~ 0.06 across prior SDs {0.5, 1, 2}).

| Horizon | Bayesian mu_beta (94% HDI) | Frequentist beta_h |
|---|---|---|
| h = 0 | -0.01 ([-1.80, +1.86]) | -0.11 |
| h = 12 | +0.03 ([-2.03, +1.63]) | +0.19 |
| h = 24 | +0.01 ([-1.99, +1.74]) | +0.10 |

- The Bayesian population IRF is **centred near zero with very wide credible
  intervals** at every horizon -- no robust relative effect.
- The posterior means are **shrunk versions of the frequentist estimates** (same
  signs, smaller magnitudes); between-sector heterogeneity ``tau`` is modest
  (0.13-0.35).

**Triangulation:** the frequentist panel-LP, the DML estimators, and the Bayesian
hierarchical LP **agree** -- the relative effect is not robustly identified on the
2014-2020 sample. The agreement across very different inference paradigms
strengthens the pre-registered null.

## Robustness (Phase 8)

**Specification curve** (24 specs: shock x exposure x lags x COVID handling), at
the primary decision horizons:

| Horizon | share negative | share sig. negative (BH<0.10) | median beta |
|---|---|---|---|
| h = 12 | 25% | 0% | +0.015 |
| h = 24 | 42% | 12.5% | +0.015 |

The headline-supporting (negative) sign is a **minority** of specifications and
is significant in essentially none -- the relative effect is **not robust**.

**Placebo / permutation** (h = 12): permuting the shock across time gives a clean
null (p = 0.17; placebo mean ~ 0); permuting exposure flags the (wrong-signed)
estimate as somewhat extreme (p = 0.04). No evidence of a robust negative effect.

**Structural breaks** (Bai-Perron on national employment growth): breaks around
the WWII demobilization and, recently, 2019, **2020-05 (COVID)**, and 2021 -- the
pandemic is the dominant recent break.

**COVID / state-dependence** (Auerbach-Gorodnichenko aggregate LP): the
**expansion-state** response is negative at the medium run (theta_12 = -3.0),
the expected contractionary sign, while the **recession-state** response is
positive (theta_12 = +5.3), reflecting the COVID-era collapse in which employment
and the shock move atypically. The expected effect is visible once the recession/
COVID state is separated.

**QCEW revision bound** (simulated): the headline coefficient is essentially
unchanged under simulated revisions of the documented magnitude
(beta in [+0.187, +0.190] around +0.189) -- revision-induced bias is negligible.

**Overall:** every robustness lens agrees with the headline -- the relative
effect is not robustly identified on the 2014-2020 sample. The one suggestive
signal (expansion-state aggregate decline) is consistent with a contractionary
effect but is assumption-dependent and outside the cleanly-identified relative
design.

## Data validation — CES vs QCEW reconciliation (Tier 0.3)

The QCEW panel (administrative, near-census) is cross-checked against the
independent CES State-and-Area survey at the supersector level, 1994–2020 (545
state × supersector pairs):

- **Median correlation of year-on-year growth: 0.92** (log levels: 0.95) — strong
  agreement validating the QCEW employment used for the panel and exposure.
- By supersector the agreement is excellent for the major private sectors
  (Manufacturing, Construction, Trade/Transport, Leisure, Professional/Business
  ≈ 0.97–0.99) and weaker for **Mining & Logging (0.56)** and **Government /
  Public Administration (0.53)** — exactly the two looser CES↔QCEW definitional
  correspondences, flagged honestly rather than smoothed over.

The QCEW panel is therefore well-validated for the sectors that carry the
identification; the divergent categories are documented (ADR-0012).

## Method notes

- Inference: Driscoll-Kraay standard errors (cross-sectional + serial robust),
  with a wild-cluster-bootstrap cross-check; BH-FDR across horizons.
- LP-DiD is implemented and validated on synthetic staggered panels but is not
  the headline for a single national shock (no staggered timing); see ADR-0005.
