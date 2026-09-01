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
    df: pd.DataFrame,
    y: str,
    lo: str,
    hi: str,
    title: str,
    ylab: str,
    xlab: str = "Horizon h (months)",
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
        xaxis_title=xlab,
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
    share_neg = float((response["beta"] < 0).mean())
    col_a, col_b = st.columns(2)
    col_a.metric("Response horizons negative", f"{share_neg:.0%}")
    col_b.metric("… and BH-significant", f"{sig} / {len(response)}")
    st.info(
        "Finding on the 3-digit panel (1994-2020, ~1.43M cell-months): the relative "
        "effect is **correctly signed (negative) at every horizon** but **not "
        "BH-significant** at the decision horizons — a credible, precisely-stated "
        "**null**, not a supported claim."
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


def robust_tab() -> None:
    st.subheader("Robust inference: does the null survive?")
    st.markdown(
        "The headline effect under three covariance choices: **Driscoll-Kraay**, "
        "the **exposure-robust** two-way (sector x time) cluster of "
        "Borusyak-Hull-Jaravel / Adao-Kolesar-Morales, and **Conley spatial**. The "
        "decision-horizon null holds under all three."
    )
    dk = load("headline_irf.csv")
    er = load("headline_exposure_robust.csv")
    cn = load("headline_conley.csv")
    dk, er, cn = (d[d["horizon"] >= 0] for d in (dk, er, cn))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dk["horizon"],
            y=dk["ci_high"],
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dk["horizon"],
            y=dk["ci_low"],
            fill="tonexty",
            fillcolor="rgba(70,130,180,0.15)",
            line={"width": 0},
            name="Driscoll-Kraay 95% CI",
        )
    )
    for frame, color, label in (
        (er, "seagreen", "exposure-robust CI"),
        (cn, "darkorange", "Conley CI (500 km)"),
    ):
        fig.add_trace(
            go.Scatter(
                x=frame["horizon"],
                y=frame["ci_low"],
                line={"color": color, "dash": "dot", "width": 1},
                name=label,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=frame["horizon"],
                y=frame["ci_high"],
                line={"color": color, "dash": "dot", "width": 1},
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=dk["horizon"],
            y=dk["beta"],
            mode="lines+markers",
            name="beta_h",
            line={"color": "steelblue"},
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Headline beta_h under three covariances",
        xaxis_title="Horizon h (months)",
        yaxis_title="beta_h",
        height=440,
        template="simple_white",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "The two-way exposure-robust band sits on top of Driscoll-Kraay. The tight "
        "Conley 500 km band is a distance-decay artifact: as the spatial kernel "
        "widens it converges to Driscoll-Kraay (below)."
    )
    cs = load("conley_sensitivity.csv")
    figc = px.line(cs, x="cutoff_km", y="se", markers=True, log_x=True)
    figc.update_layout(
        title="Conley SE (h=12) vs spatial cutoff -> converges to Driscoll-Kraay",
        xaxis_title="cutoff (km, log)",
        yaxis_title="standard error",
        height=340,
        template="simple_white",
    )
    st.plotly_chart(figc, use_container_width=True)
    ri = load("randomization_inference.csv")
    figr = px.bar(ri, x="horizon", y="ri_p_value", text_auto=".2f")
    figr.add_hline(y=0.05, line_dash="dash", line_color="crimson")
    figr.update_layout(
        title="Randomization inference (circular-shift) p-values",
        xaxis_title="Horizon h (months)",
        yaxis_title="RI p-value",
        height=340,
        template="simple_white",
    )
    st.plotly_chart(figr, use_container_width=True)
    st.caption(
        "Circular shifts preserve the shock's serial dependence. The decision "
        "horizons are far from significant; the joint sharp null is not rejected."
    )


def shocks_tab() -> None:
    st.subheader("Shocks & instruments")
    st.markdown(
        "Identification uses a high-frequency instrument (Bauer-Swanson MPS) and a "
        "narrative shock (updated Romer-Romer), cross-checked for the "
        "central-bank information effect."
    )
    mps = load("aggregate_lpiv_irf.csv")
    brw = load("aggregate_lpiv_brw.csv")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mps["horizon"],
            y=mps["first_stage_f"],
            name="MPS (high-frequency)",
            line={"color": "steelblue"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=brw["horizon"], y=brw["first_stage_f"], name="BRW", line={"color": "gray"}
        )
    )
    fig.add_hline(
        y=10,
        line_dash="dash",
        line_color="crimson",
        annotation_text="weak-instrument threshold (F=10)",
    )
    fig.update_layout(
        title="LP-IV first-stage strength: MPS vs BRW",
        xaxis_title="Horizon h (months)",
        yaxis_title="robust first-stage F",
        height=380,
        template="simple_white",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "The MPS instrument clears the weak-instrument threshold at h=0/12 "
        "(F ~ 13-15); BRW is materially weaker."
    )
    rr = load("narrative_lp_irf.csv")
    st.plotly_chart(
        _irf_figure(
            rr,
            "theta",
            "ci_low",
            "ci_high",
            "Narrative (Romer-Romer) shock: quarterly employment LP",
            "response (%)",
            xlab="Horizon h (quarters)",
        ),
        use_container_width=True,
    )
    st.caption(
        "Correctly signed (negative at the medium run) but imprecise — every band "
        "includes zero."
    )
    sd = load("shock_diagnostics.csv").set_index("metric")["value"]
    col_a, col_b = st.columns(2)
    col_a.metric(
        "Information share (HF window)",
        f"{sd.get('info_hf_contamination_share', float('nan')):.0%}",
    )
    col_b.metric(
        "Information share (monthly proxy)",
        f"{sd.get('info_mps_monthly_contamination_share', float('nan')):.0%}",
    )
    st.caption(
        "The monthly proxy overstates central-bank information contamination; the "
        "high-frequency test (Jarocinski-Karadi) is the valid one."
    )


def design_tab() -> None:
    st.subheader("Design diagnostics")
    bc = load("bayes_cell_summary.csv").iloc[0]
    ba = load("bacon_summary.csv").iloc[0]
    bk = load("break_selection.csv")
    rb = load("revision_bound.csv").iloc[0]
    n_breaks = int(bk.loc[bk["bic"].idxmin(), "n_breaks"])
    cols = st.columns(4)
    cols[0].metric("Between-industry share", f"{bc['between_share']:.1%}")
    cols[1].metric("TWFE forbidden weight", f"{ba['forbidden_weight']:.0%}")
    cols[2].metric("Structural breaks (BIC)", f"{n_breaks}")
    cols[3].metric(
        "Revision-bound range", f"[{rb['beta_min']:.3f}, {rb['beta_max']:.3f}]"
    )
    st.markdown(
        "- **Heterogeneity** is ~99.9% between industries, ~0% within an industry "
        "across states — confirming the shift-share premise from the data.\n"
        "- **Goodman-Bacon**: a naive staggered two-way-FE estimator would put "
        "~38% of its weight on *forbidden* already-treated-control comparisons; the "
        "clean LP/interaction design avoids this.\n"
        "- **No structural break** in national employment growth over 1994-2020 "
        "(the 2020 COVID episode is a transient spike, not a regime shift).\n"
        "- Under a conservative calibrated QCEW-revision model the headline "
        "**stays negative and never crosses zero**."
    )
    fig = px.bar(
        x=[
            "treated vs never",
            "earlier vs later",
            "later vs already-treated (forbidden)",
        ],
        y=[
            ba["weight_treated_vs_untreated"],
            ba["weight_earlier_vs_later"],
            ba["weight_later_vs_earlier"],
        ],
        labels={"x": "comparison type", "y": "TWFE weight"},
        color=["clean", "clean", "forbidden"],
        color_discrete_map={"clean": "steelblue", "forbidden": "crimson"},
    )
    fig.update_layout(
        title="Goodman-Bacon: TWFE weight by comparison type",
        height=360,
        template="simple_white",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


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
        [
            "Effect",
            "Robust inference",
            "Shocks & instruments",
            "Design diagnostics",
            "Exposure map",
            "Specification curve",
            "Triangulation",
            "About",
        ]
    )
    with tabs[0]:
        if estimand == "Headline (relative)":
            headline_tab()
        else:
            aggregate_tab()
    with tabs[1]:
        robust_tab()
    with tabs[2]:
        shocks_tab()
    with tabs[3]:
        design_tab()
    with tabs[4]:
        exposure_tab()
    with tabs[5]:
        spec_curve_tab()
    with tabs[6]:
        triangulation_tab()
    with tabs[7]:
        st.markdown(
            "Built phase by phase with a frozen pre-registered analysis plan. "
            "The headline relative effect is identified off cross-sectional "
            "exposure heterogeneity with time fixed effects; the aggregate effect "
            "is an assumption-dependent complement. See the project documentation "
            "for methods, data, and the full results write-up."
        )


if __name__ == "__main__":
    main()
