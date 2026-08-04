# ADR-0013: Tier 1.1 — stronger high-frequency instrument

- Status: Accepted
- Date: 2026-08-04

## Context

The Phase 5 proxy-SVAR / LP-IV used BRW as the external instrument, giving a
**weak first stage** (robust F ~ 3-5), so the rate-scaled aggregate IRF was not
reliably identified. Tier 1.1 replaces the instrument with a purpose-built
high-frequency surprise.

## Decision

- Ingest the **Bauer-Swanson monetary policy surprises** published (and
  maintained) by the San Francisco Fed (`cil.data.mps`): 30-minute interest-rate
  changes around FOMC announcements, as the raw `MPS`, an orthogonalized
  `MPS_ORTH`, and — per FOMC — the same-window S&P 500 change (for the
  information-effect test, Tier 1.3). Monthly, 1988-2023.
- Use **MPS as the headline LP-IV instrument** for the aggregate IRF; keep BRW as
  a robustness variant (`lpiv_irf_brw`).

## Result (honest)

- The first stage **strengthens materially**: robust F ~ 13-15 at h = 0 and
  h = 12 (min across horizons ~ 6.8) versus ~ 3-5 for BRW — clearing the
  weak-instrument threshold at the horizons that matter.
- The point IRF, however, is **wrong-signed at the medium run** (theta_12 > 0:
  employment appears to *rise* after a tightening), with an Anderson-Rubin
  interval excluding zero. This is a counter-intuitive, price-puzzle-like result
  of the monthly LP-IV design (instrumenting the monthly effective-funds-rate
  change with a 30-minute surprise). It is reported, not hidden.

## Consequences

- Identification of the aggregate rate response is now strong, but the estimate
  is economically implausible at the medium run — reinforcing the project's
  stance that the **aggregate effect is an assumption-dependent complement**, not
  the reliable answer; the cleanly-identified relative design remains the
  headline.
- The ingested MPS also supplies Tier 1.3 (HF information-effect) and Tier 1.4
  (the orthogonalized shock variant, `MPS_ORTH`).
