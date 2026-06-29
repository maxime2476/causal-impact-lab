"""Causal Impact Lab -- interactive results app (Streamlit).

Self-contained: reads the committed CSV result artifacts in ``assets/`` (exported
by ``cil.report.export``), so it runs on a lightweight Hugging Face Space without
the analysis stack. Presents the headline relative effect, the aggregate
complement, the exposure map, the specification curve, and the cross-estimator
triangulation -- reporting the pre-registered null honestly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ASSETS = Path(__file__).parent / "assets"


@st.cache_data
def load(name: str) -> pd.DataFrame:
    """Load a committed result CSV by file name."""
    return pd.read_csv(ASSETS / name)


def _irf_figure(
    df: pd.DataFrame, y: str, lo: str, hi: str, title: str, ylab: str
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["horizon"],
            y=df[hi],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["horizon"],
            y=df[lo],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(70,130,180,0.2)",
            line={"width": 0},
            name="95% CI",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["horizon"],
            y=df[y],
            mode="lines+markers",
            name="estimate",
            line={"color": "steelblue"},
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=title,
        xaxis_title="Horizon h (months)",
        yaxis_title=ylab,
        height=440,
        template="simple_white",
    )
    return fig


def headline_tab() -> None:
    st.subheader("Headline: relative effect (cleanly identified)")
    st.markdown(
        "Do higher-exposure cells lose **more** employment after a contractionary "
        "shock? The supporting sign is **negative**. Pre-registered decision "
        "horizons: h = 12 and h = 24."
    )
    df = load("headline_irf.csv")
    response = df[df["horizon"] >= 0]
    st.plotly_chart(
        _irf_figure(
            response,
            "beta",
            "ci_low",
            "ci_high",
            "Relative semi-elasticity beta_h (Driscoll-Kraay 95% CI)",
            "beta_h",
        ),
        use_container_width=True,
    )
    sig = int(((response["p_value_bh"] < 0.10) & (response["beta"] < 0)).sum())
    st.metric("Response horizons negative & BH-significant", f"{sig} / {len(response)}")
    st.info(
        "Finding: **no** response horizon is robustly negative and significant. "
        "The pre-registered claim is **not supported** on the 2014-2020 sample."
    )
    leads = df[df["horizon"] < 0]
    if not leads.empty:
        st.caption("Pre-trend leads (should be flat if parallel-trends holds):")
        st.plotly_chart(
            _irf_figure(
                leads, "beta", "ci_low", "ci_high", "Event-study leads", "beta_h"
            ),
            use_container_width=True,
        )


def aggregate_tab() -> None:
    st.subheader("Complement: aggregate dynamic effect (assumption-dependent)")
    st.markdown(
        "The impulse response of **national** employment to the shock. Reported "
        "separately, with its identifying assumptions stress-tested -- never as "
        "'the' answer."
    )
    ts = load("aggregate_ts_irf.csv")
    st.plotly_chart(
        _irf_figure(
            ts,
            "theta",
            "ci_low",
            "ci_high",
            "Time-series LP: employment response to the shock (HAC 95% CI)",
            "theta_h (%)",
        ),
        use_container_width=True,
    )
    iv = load("aggregate_lpiv_irf.csv")
    st.plotly_chart(
        _irf_figure(
            iv,
            "theta",
            "ar_low",
            "ar_high",
            "LP-IV: response to a +1pp policy rate (Anderson-Rubin interval)",
            "theta_h (%)",
        ),
        use_container_width=True,
    )
    st.warning(
        f"The LP-IV first stage is **weak** (min F = "
        f"{iv['first_stage_f'].min():.1f}); the rate-scaled IRF is not reliably "
        "identified, hence the wide Anderson-Rubin intervals."
    )


def exposure_tab() -> None:
    st.subheader("Predetermined interest-rate exposure by state")
    df = load("state_exposure.csv")
    fig = px.choropleth(
        df,
        locations="state",
        locationmode="USA-states",
        color="exposure",
        scope="usa",
        color_continuous_scale="RdBu_r",
        labels={"exposure": "Exposure (z)"},
    )
    fig.update_layout(height=480, margin={"r": 0, "t": 10, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Shift-share exposure E_s = sum_k omega_{s,k} sigma_k, standardized. "
        "Higher = more exposed to interest-rate-sensitive sectors."
    )
    st.dataframe(
        load("exposure_sigma.csv").rename(columns={"supersector_code": "supersector"}),
        use_container_width=True,
        hide_index=True,
    )


def spec_curve_tab() -> None:
    st.subheader("Specification curve")
    df = load("spec_curve.csv")
    df["spec"] = (
        df["shock"]
        + " / "
        + df["exposure"]
        + " / lag"
        + df["lags"].astype(str)
        + " / "
        + df["sample"]
    )
    horizon = st.selectbox("Horizon", sorted(df["horizon"].unique()))
    sub = df[df["horizon"] == horizon].sort_values("beta").reset_index(drop=True)
    sub["significant"] = (sub["p_value_bh"] < 0.10).map(
        {True: "BH-significant", False: "not significant"}
    )
    fig = px.scatter(
        sub,
        x=sub.index,
        y="beta",
        color="significant",
        hover_name="spec",
        error_y=1.96 * sub["se"],
        color_discrete_map={
            "BH-significant": "crimson",
            "not significant": "steelblue",
        },
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        height=460,
        xaxis_title="Specification (sorted by beta)",
        yaxis_title=f"beta at h={horizon}",
        template="simple_white",
    )
    st.plotly_chart(fig, use_container_width=True)
    share_neg = float((sub["beta"] < 0).mean())
    st.metric(
        "Share of specifications with the supporting (negative) sign",
        f"{share_neg:.0%}",
    )


def triangulation_tab() -> None:
    st.subheader("Cross-estimator triangulation")
    st.markdown(
        "Frequentist panel-LP, Bayesian hierarchical LP, and DML -- do they agree?"
    )
    bf = load("bayes_vs_freq.csv")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=bf["horizon"], y=bf["freq_beta"], name="Frequentist beta"))
    fig.add_trace(go.Bar(x=bf["horizon"], y=bf["bayes_mu"], name="Bayesian mu"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        barmode="group",
        height=420,
        xaxis_title="Horizon",
        yaxis_title="Effect",
        template="simple_white",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(load("dml_results.csv"), use_container_width=True, hide_index=True)
    st.success(
        "All three estimators agree: the relative effect is not robustly "
        "identified on this sample. Agreement across paradigms strengthens the "
        "pre-registered null."
    )


def main() -> None:
    """Render the application."""
    st.set_page_config(page_title="Causal Impact Lab", page_icon="📉", layout="wide")
    st.title("Causal Impact Lab")
    st.markdown(
        "Causal effect of contractionary US monetary policy shocks on US "
        "employment. **The deliverable is an honest answer, including nulls.**"
    )
    estimand = st.radio(
        "Estimand",
        ["Headline (relative)", "Complement (aggregate)"],
        horizontal=True,
    )
    tabs = st.tabs(
        ["Effect", "Exposure map", "Specification curve", "Triangulation", "About"]
    )
    with tabs[0]:
        if estimand == "Headline (relative)":
            headline_tab()
        else:
            aggregate_tab()
    with tabs[1]:
        exposure_tab()
    with tabs[2]:
        spec_curve_tab()
    with tabs[3]:
        triangulation_tab()
    with tabs[4]:
        st.markdown(
            "Built phase by phase with a frozen pre-registered analysis plan. "
            "The headline relative effect is identified off cross-sectional "
            "exposure heterogeneity with time fixed effects; the aggregate effect "
            "is an assumption-dependent complement. See the project documentation "
            "for methods, data, and the full results write-up."
        )


if __name__ == "__main__":
    main()
