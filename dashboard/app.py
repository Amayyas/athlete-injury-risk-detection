"""Streamlit dashboard — Injury risk detection.

Two readings of the same athlete, side by side:

- a **rule-based score** (`injury_risk.features.risk_factors`), always available and
  fully transparent — the score *is* the sum of its displayed factors;
- the **trained model's** calibrated probability and its cost-based decision
  (`injury_risk.inference`), shown with a live SHAP explanation of *this* athlete.

The app stays usable with no trained model: the rule score needs none, and the model
sections simply announce how to train one.

This file only renders. Scoring, inference and explanation all live in the package,
where they are tested and where the future API reuses them unchanged.

Launch:
    injury-risk dashboard
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import shap
import streamlit as st

from injury_risk.features.engineering import acwr_zone
from injury_risk.features.risk_factors import HIGH, INFO, assess
from injury_risk.inference import AthleteInputs
from injury_risk.service import resolve_service

st.set_page_config(page_title="Athlete Injury Risk", page_icon="🩺", layout="wide")

POSITIONS = ("goalkeeper", "defender", "midfielder", "forward")
RISK_LABELS = {0: "Low", 1: "Moderate", 2: "High"}

# --------------------------------------------------------------------------- #
# Sidebar: athlete parameters
# --------------------------------------------------------------------------- #
st.sidebar.header("Athlete parameters")

age = st.sidebar.slider("Age", 16, 40, 24)
position = st.sidebar.selectbox("Position", POSITIONS, index=2)

# Loads are daily training loads (RPE x duration), the scale the model was trained on
# — an acute/chronic average of a few dozen, not a weekly total in the hundreds.
st.sidebar.markdown("**Training load** (daily, avg)")
acute_load = st.sidebar.slider("Acute load (7-day avg)", 0, 200, 66)
chronic_load = st.sidebar.slider("Chronic load (28-day avg)", 1, 200, 63)

st.sidebar.markdown("**Physiology**")
sleep_hours = st.sidebar.slider("Sleep (h/night)", 3.0, 10.0, 7.4, 0.1)
soreness = st.sidebar.slider("Soreness (0–10)", 0.0, 10.0, 3.0, 0.5)
resting_hr = st.sidebar.slider("Resting HR (bpm)", 40, 95, 55)
baseline_hr = st.sidebar.slider("Usual resting HR (baseline)", 40, 80, 55)

st.sidebar.markdown("**History**")
injury_prone = st.sidebar.checkbox("Injury proneness", value=False)
previous_injuries = st.sidebar.number_input("Previous injuries", 0, 20, 1)
days_since_injury = st.sidebar.number_input("Days since last injury", 0, 2000, 200)

inputs = AthleteInputs(
    age=age,
    position=position,
    acute_load=acute_load,
    chronic_load=chronic_load,
    sleep_hours=sleep_hours,
    soreness=soreness,
    resting_hr=resting_hr,
    baseline_hr=baseline_hr,
    injury_prone=injury_prone,
    previous_injuries=int(previous_injuries),
    days_since_injury=float(days_since_injury),
)

# --------------------------------------------------------------------------- #
# Rule-based score (always available)
# --------------------------------------------------------------------------- #
acwr = inputs.acwr
zone = acwr_zone(acwr)
assessment = assess(
    acwr=acwr,
    soreness=soreness,
    sleep_hours=sleep_hours,
    resting_hr=resting_hr,
    baseline_hr=baseline_hr,
    injury_prone=injury_prone,
    previous_injuries=int(previous_injuries),
    days_since_injury=float(days_since_injury),
)
score, level, factors = assessment.score, assessment.level, assessment.factors

ZONE_COLORS = {
    "under": "#3b82f6",
    "optimal": "#22c55e",
    "elevated": "#f59e0b",
    "danger": "#ef4444",
    "unknown": "#9ca3af",
}
LEVEL_COLORS = {0: "#22c55e", 1: "#f59e0b", 2: "#ef4444"}
SEVERITY_ICONS = {HIGH: "🔴", INFO: "🔵"}


@st.cache_resource(show_spinner=False)
def _resolved():
    """Where predictions come from: the API if one is configured, else this process.

    Set INJURY_RISK_API_URL to point the dashboard at a running service; without it
    the model is loaded in-process, which is what the single-process free hosting
    needs. Either way the rest of this file is identical — that is the point.
    """
    return resolve_service()


resolution = _resolved()
service = resolution.service

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🩺 Muscle injury risk detection")
st.caption(
    "Two readings of the same athlete: a transparent rule-based score, and the "
    "trained model's calibrated probability of an injury in the next 7 days."
)

SOURCE_BADGES = {
    "api": ("🌐", "#2563eb", "Predictions served by the REST API"),
    "local": ("💻", "#6b7280", "Model loaded in this process"),
    "none": ("⚠️", "#f59e0b", "No model available"),
}
icon, colour, label = SOURCE_BADGES[resolution.source]
st.markdown(
    f"<span style='background:{colour};color:white;padding:2px 10px;border-radius:12px;"
    f"font-size:0.8em'>{icon} {label}</span> "
    f"<span style='color:#9ca3af;font-size:0.8em'>— {resolution.detail}</span>",
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rule-based score", f"{score:.2f}", help="0 = low, 1 = very high")
    st.markdown(
        f"<div style='padding:8px;border-radius:8px;background:{LEVEL_COLORS[level]};"
        f"color:white;text-align:center;font-weight:600'>Level: {RISK_LABELS[level]}</div>",
        unsafe_allow_html=True,
    )

with col2:
    st.metric("ACWR", f"{acwr:.2f}")
    st.markdown(
        f"<div style='padding:8px;border-radius:8px;background:{ZONE_COLORS[zone]};"
        f"color:white;text-align:center;font-weight:600'>Zone: {zone}</div>",
        unsafe_allow_html=True,
    )

with col3:
    if service is not None:
        prediction = service.predict(inputs)
        at_risk = prediction.at_risk
        st.metric(
            "Model — injury in 7 days",
            f"{prediction.probability:.1%}",
            help=f"Calibrated probability. Flags at ≥ {prediction.threshold:.1%} "
            f"(threshold set by a cost-based rule: a miss costs 10× a false alarm).",
        )
        st.markdown(
            f"<div style='padding:8px;border-radius:8px;"
            f"background:{'#ef4444' if at_risk else '#22c55e'};color:white;"
            f"text-align:center;font-weight:600'>"
            f"{'⚠️ At risk' if at_risk else '✅ Not flagged'}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.metric("Model — injury in 7 days", "—")
        st.caption("Train a model to enable: `injury-risk train --track synthetic`")

st.divider()

# --------------------------------------------------------------------------- #
# Active risk factors — the rule score's own decomposition.
# --------------------------------------------------------------------------- #
st.subheader("Active risk factors (rule-based)")

if factors:
    st.caption(
        "Each factor shows the risk points it contributes; they sum to the rule score "
        "above, so nothing can raise it without appearing here."
    )
    for factor in factors:
        icon = SEVERITY_ICONS.get(factor.severity, "🟠")
        st.markdown(
            f"- {icon} **{factor.label}** — {factor.detail} &nbsp;`+{factor.percent} pts`",
            unsafe_allow_html=True,
        )
else:
    st.success("No risk factor detected ✅")

st.divider()

# --------------------------------------------------------------------------- #
# Model explainability (SHAP) — live, for the athlete above.
# --------------------------------------------------------------------------- #
st.subheader("Why the model says so (SHAP)")

if service is not None:
    st.caption(
        f"Live explanation of the **{service.model}** model's prediction for the "
        "current athlete — each bar is how a feature pushed the probability up or down."
    )
    explanation = service.explain(inputs)
    shap.plots.waterfall(explanation, show=False, max_display=12)
    fig = plt.gcf()
    fig.set_size_inches(9, 6)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    st.caption(
        "ℹ️ The model reads a daily time series; here it works from a snapshot, so the "
        "rolling features assume the current values are steady-state. Directional, not "
        "a substitute for the full series."
    )
else:
    st.info(
        "The live SHAP explanation appears here once a model is trained:\n\n"
        "```\ninjury-risk train --track synthetic\n```"
    )

with st.expander("ℹ️ About the data & the two readings"):
    st.markdown(
        "- **Rule-based score** — transparent business rules (ACWR, sleep, soreness, "
        "HR, history). No model needed; the factor list is its exact decomposition.\n"
        "- **Model** — logistic regression trained on simulated injury *events*, "
        "predicting an injury within 7 days. Its probability is **calibrated**, and it "
        "is flagged against a **cost-based threshold** (a miss costs 10× a false "
        "alarm).\n"
        "- The synthetic hazard is logit-linear by construction, which is why a linear "
        "model wins there; on the real SIRP-600 dataset a random forest wins instead."
    )
