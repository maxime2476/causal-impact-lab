---
title: Causal Impact Lab
emoji: 📉
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Causal Impact Lab

Interactive results for an honest estimate of the causal effect of contractionary
US monetary policy shocks on US employment.

The app is self-contained: it reads the committed CSV result artifacts in
`assets/` (exported from the analysis store), so it runs without the analysis
stack or any API key. It presents:

- the **headline relative effect** (interacted panel local projection) with
  event-study leads,
- the **aggregate complement** (time-series LP and a weak-IV LP-IV),
- the predetermined **exposure map** by state,
- the **specification curve**, and
- the cross-estimator **triangulation** (frequentist, Bayesian, DML).

Results are reported honestly, including the pre-registered null.

## Run locally

```bash
pip install -r requirements.txt
streamlit run main.py
```
