# The Employment Effects of Contractionary Monetary Policy: A Credible Null

*A reproducible study whose deliverable is an honest answer, including when that
answer is "we cannot detect the effect."*

## Abstract

We estimate the causal effect of contractionary U.S. monetary-policy shocks on
U.S. employment using a state × 3-digit-industry panel (1994–2020, ~1.43M
cell-months) and an interacted panel local projection whose identifying variation
is the interaction of an identified national shock with predetermined
cross-sectional exposure. The headline **relative** semi-elasticity is
**correctly signed (negative) at every horizon** — a tightening lowers employment
more in exposed industries — but is **not statistically significant** after
false-discovery control, and this null survives an unusually broad battery of
robustness checks: exposure-robust (Borusyak-Hull-Jaravel / Adão-Kolesár-Morales)
inference, Conley spatial standard errors, double/debiased machine learning, a
cell-level Bayesian hierarchy, a specification curve, placebo and design-valid
randomization-inference tests, a calibrated data-revision bound, and a full
Goodman-Bacon decomposition. The **aggregate** rate response — the
assumption-dependent complement — is only weakly identified and its sign depends
on the instrument, so we do not report it as the answer. The contribution is not
a new number; it is a **credible, correctly-signed null** delivered with the full
apparatus that would be needed to believe a positive result, and with every
fragility disclosed.

## 1. Introduction

Does contractionary monetary policy reduce employment, and by how much? The
mechanism is textbook, but credible identification on U.S. data is hard: policy
responds to the economy, shocks are serially correlated and few, and the outcome
is buffeted by everything else. This project treats **intellectual honesty as the
deliverable**. Rather than search specifications until an effect appears, we
pre-register a falsifiable claim, build the estimator that could support it, and
report what the data say — prominently including the case where the effect is not
robustly distinguishable from zero.

Two estimands are separated throughout:

- **Estimand 1 — the relative effect (headline, cleanly identified).** Within a
  month, do high-exposure industries lose more employment than low-exposure ones
  after a tightening? Time fixed effects absorb the aggregate component, so this
  is identified from cross-sectional variation in exposure and does not rest on
  the shock being exogenous to the aggregate state.
- **Estimand 2 — the aggregate effect (assumption-dependent complement).** What
  is the economy-wide employment response to a policy-rate increase? This requires
  the shock to be exogenous to the aggregate state — an assumption we state and
  stress-test rather than assume away.

The headline is Estimand 1. The aggregate is reported separately, with its
fragility quantified, and never as "the" answer.

## 2. Data

| Series | Source | PIT treatment |
|---|---|---|
| National employment (`PAYEMS`) & macro confounders | FRED / ALFRED | strict point-in-time (vintages) |
| State × 3-digit-industry employment | BLS QCEW (bulk flat files) | revised/final, documented (§7) |
| State employment cross-check | BLS CES-SAE | as-published |
| Policy rate + shadow rate | FRED + Wu-Xia (2016) | spliced at the ZLB |
| Monetary shocks | Bu-Rogers-Wu (2021); Bauer-Swanson (2023); Romer-Romer (2004) / Breitenlechner (2018) | as-published |

The analysis panel is 4,566 (state × 3-digit-industry) cells over 1994–2020.
Every raw pull is content-addressed in a provenance ledger (source, URL,
retrieval time, vintage, SHA-256, byte count). An independent cross-check against
the CES state-and-area survey gives a **median cell-level year-on-year growth
correlation of 0.92**, validating the QCEW panel.

## 3. Identification and estimators

### 3.1 Headline: interacted panel local projection

For horizon `h`,

```
y_{i,t+h} − y_{i,t−1} = β_h (E_i · s_t) + γ_i + δ_t + Σ_l φ_{h,l} Δy_{i,t−l} + ε,
```

with `y` = log employment, unit (`γ_i`) and time (`δ_t`) fixed effects, `E_i` the
predetermined industry exposure (a standardised interest-rate sensitivity), and
`s_t` the identified national shock. Time effects absorb the aggregate/common
component, so `β_h` is the **relative** semi-elasticity per unit exposure per unit
shock (Jordà, 2005). Identification rests on **conditional parallel trends**,
tested with event-study leads.

### 3.2 Shocks and instruments

The headline uses the Bu-Rogers-Wu (2021) shock. For the aggregate complement we
add a high-frequency instrument — the Bauer-Swanson (2023) monetary-policy
surprise, whose LP-IV **first stage is strong (robust F ≈ 13–15 at h = 0/12)**
versus a materially weaker BRW first stage — and a non-market **narrative** shock,
the Breitenlechner (2018) update of Romer-Romer (2004). A high-frequency
information-effect test (Jarociński-Karadi, 2020) using the same-window rate and
equity moves classifies **32.5%** of FOMC surprises as information-driven, versus
**52%** for a monthly proxy that is inflated by non-FOMC news.

### 3.3 Robust inference and complementary estimators

Because the design is a shift-share, inference is reported under Driscoll-Kraay
standard errors and, additionally, the **exposure-robust** two-way (supersector ×
time) covariance of Borusyak-Hull-Jaravel (2022) / Adão-Kolesár-Morales (2019),
and **Conley (1999) spatial** standard errors. The relative effect is
triangulated with double/debiased machine learning (Chernozhukov et al., 2018,
with purged time-blocked cross-fitting), a cell-level Bayesian hierarchy (Gelman
& Hill, 2007), the LP-DiD estimator (Dube-Girardi-Jordà-Taylor, 2025), and a full
Goodman-Bacon (2021) decomposition.

## 4. The headline result

On the 3-digit panel the relative semi-elasticity is **negative at every one of
the 25 response horizons**, the event-study leads are clean (no pre-trend), and
**no horizon survives Benjamini-Hochberg control**:

| Horizon | `β_h` | Driscoll-Kraay SE | BH-adjusted p |
|---|---|---|---|
| 0 | −0.025 | 0.017 | 0.48 |
| 12 | −0.021 | 0.046 | 0.77 |
| 24 | −0.012 | 0.057 | 0.83 |

This is the finding: a **correctly-signed, precisely-stated, statistically
insignificant** relative effect. The magnitudes are economically small, and the
uncertainty bands comfortably include zero at the decision horizons.

## 5. Triangulation

Very different inference paradigms agree that the relative effect is not robustly
identified on this sample:

- **Double/debiased ML.** The LinearDML effect is **significantly negative on
  impact** (h = 0: −0.016, 95% CI [−0.021, −0.011]) with the causal forest
  agreeing in sign, then drifts toward zero/positive at longer horizons; placebo
  (time-permuted treatment) effects are ≈ 0.
- **Bayesian hierarchy.** The pooled posterior population IRF is negative at
  every horizon (μ_β = −0.004 / −0.018 / −0.006), a shrunk version of the
  frequentist estimate — the two paradigms now agree in sign.
- **Specification curve.** The median coefficient is negative at both decision
  horizons, with 0% of specifications BH-significant.

## 6. Heterogeneity

A cell-level Bayesian decomposition of where the response heterogeneity lives is
stark: at h = 12 the between-supersector standard deviation is 0.060 versus a
within-supersector (across-state) standard deviation of 0.0016, a **between-share
of 99.9%**. Almost all of the cross-cell variation is explained by industry
composition, essentially none by state-specific idiosyncrasies — confirming the
premise of the shift-share design from the data rather than assuming it. A
multi-feature causal forest finds only weak, cardinality-unbiased driver signal,
consistent with this.

## 7. Robustness

The null is not an artefact of one inferential choice:

- **Exposure-robust inference.** Two-way (sector × time) clustering — the
  design-appropriate exposure-robust covariance — lands on top of Driscoll-Kraay
  (h = 12 SE 0.049 vs 0.046) and preserves the decision-horizon null. A cautionary
  finding is documented: *naive one-way* clustering understates the SE ~3× and
  would spuriously flip the result, because it ignores the within-time
  correlation the common aggregate shock induces.
- **Conley spatial.** The Conley standard error is strongly cutoff-dependent; as
  the spatial kernel widens to admit full cross-sectional dependence it converges
  to Driscoll-Kraay (0.041 vs 0.046) and the null holds. Short-cutoff
  "significance" is a distance-decay artefact, reported via a sensitivity table.
- **Randomization inference.** A circular-shift test (which preserves the shock's
  serial dependence, unlike an iid permutation) does **not reject the joint sharp
  null** (joint max|β| p = 0.78); the decision horizons are clearly
  non-significant (h = 12 p = 0.65, h = 24 p = 0.84).
- **Structural breaks.** A full Bai-Perron analysis (exact dynamic-programming
  segmentation, BIC selection) finds **zero breaks** in national employment growth
  over 1994–2020; the 2020 COVID episode is a transient spike, not a mean-regime
  shift.
- **Data-revision bound.** Real state-industry QCEW vintages do not exist, so a
  benchmark-step revision model calibrated to the QCEW-vs-CES growth discrepancy
  bounds the coefficient at **[−0.025, −0.019]** — firmly negative, never crossing
  zero.
- **Goodman-Bacon.** On a staggered operationalisation of the exposure design, a
  naive two-way-FE estimator would place **~38% of its weight on "forbidden"**
  already-treated-as-control comparisons — quantifying exactly why the headline
  uses the clean LP/interaction and clean-control LP-DiD designs, which avoid it.

## 8. The aggregate complement

The economy-wide rate response is the assumption-dependent complement, and it is
**not reliably identified**. With the strong high-frequency instrument the LP-IV
first stage is strong but the point IRF is *wrong-signed* at the medium run
(price-puzzle-like). With the narrative shock the response is *correctly signed*
(employment troughs near −0.43% around three to four years) but imprecise, with
every band including zero. The two identifications disagree on sign; that
disagreement is the message. The aggregate is a complement, not the answer.

## 9. Discussion and limitations

The honest reading is that, on publicly reproducible U.S. data for 1994–2020, the
relative employment effect of contractionary monetary shocks is **correctly
signed but too imprecise to call significant**, and the aggregate effect is
**weakly and assumption-dependently identified**. This is not a failure of the
study; it is the study's result, and it is delivered with the full apparatus one
would demand before believing the opposite conclusion.

Limitations are stated plainly: QCEW publishes no state-industry point-in-time
vintages, so the panel uses revised data (bounded in §7); the aggregate
identification is fragile; the sample is a single country over one policy regime;
and the impact-horizon significance under some estimators (DML, exposure-robust
h = 2–3, randomization inference h = 0 borderline) is suggestive but does not
carry to the pre-registered decision horizons.

## 10. Conclusion

We set out to estimate the causal effect of contractionary monetary policy on
employment and to report it honestly. The answer is a **credible null**: a
correctly-signed relative effect that is not statistically significant and that
survives an unusually demanding robustness battery, alongside an aggregate
complement that is not reliably identified. The value is in *how* the null is
established — with pre-registration, clean identification, and every fragility
disclosed — so that the reader can trust the "no" as much as they would have
trusted a "yes."

## References

- Adão, R., Kolesár, M., & Morales, E. (2019). Shift-share designs: theory and
  inference. *Quarterly Journal of Economics* 134(4).
- Bai, J. (1997). Estimation of a change point in multiple regression models.
  *Review of Economics and Statistics* 79(4).
- Bai, J., & Perron, P. (2003). Computation and analysis of multiple structural
  change models. *Journal of Applied Econometrics* 18(1).
- Bauer, M. D., & Swanson, E. T. (2023). A reassessment of monetary policy
  surprises and high-frequency identification. *NBER Macroeconomics Annual* 37.
- Borusyak, K., Hull, P., & Jaravel, X. (2022). Quasi-experimental shift-share
  research designs. *Review of Economic Studies* 89(1).
- Breitenlechner, M. (2018). *An update of Romer and Romer (2004) narrative U.S.
  monetary policy shocks up to 2012Q4.* University of Innsbruck.
- Bu, C., Rogers, J., & Wu, W. (2021). A unified measure of Fed monetary policy
  shocks. *Journal of Monetary Economics* 118.
- Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment
  and structural parameters. *Econometrics Journal* 21(1).
- Conley, T. G. (1999). GMM estimation with cross sectional dependence. *Journal
  of Econometrics* 92(1).
- Driscoll, J. C., & Kraay, A. C. (1998). Consistent covariance matrix estimation
  with spatially dependent panel data. *Review of Economics and Statistics* 80(4).
- Dube, A., Girardi, D., Jordà, Ò., & Taylor, A. M. (2025). A local projections
  approach to difference-in-differences. *Journal of Applied Econometrics*.
- Gelman, A., & Hill, J. (2007). *Data Analysis Using Regression and
  Multilevel/Hierarchical Models.* Cambridge University Press.
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment
  timing. *Journal of Econometrics* 225(2).
- Jarociński, M., & Karadi, P. (2020). Deconstructing monetary policy surprises.
  *American Economic Journal: Macroeconomics* 12(2).
- Jordà, Ò. (2005). Estimation and inference of impulse responses by local
  projections. *American Economic Review* 95(1).
- Romer, C. D., & Romer, D. H. (2004). A new measure of monetary shocks:
  derivation and implications. *American Economic Review* 94(4).
- Wu, J. C., & Xia, F. D. (2016). Measuring the macroeconomic impact of monetary
  policy at the zero lower bound. *Journal of Money, Credit and Banking* 48(2–3).
