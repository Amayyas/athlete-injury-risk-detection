"""Streamlit dashboard — Injury risk detection.

The app is **functional even without a trained model**: the risk score is computed
live from the business rules defined in :mod:`injury_risk.features.risk_factors`.
If an XGBoost model has been trained, a section additionally displays the SHAP
plots.

This file is deliberately thin: it renders, it does not decide. The scoring and
the risk-factor logic live in the package, where they can be tested.

Launch:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import streamlit as st

from injury_risk.config import FIGURES_DIR, POSITIONS, RISK_LABELS
from injury_risk.features.engineering import acwr_zone
from injury_risk.features.risk_factors import HIGH, INFO, assess

st.set_page_config(page_title="Athlete Injury Risk", page_icon="🩺", layout="wide")

# --------------------------------------------------------------------------- #
# Sidebar: athlete parameters
# --------------------------------------------------------------------------- #
st.sidebar.header("Athlete parameters")

age = st.sidebar.slider("Age", 16, 40, 24)
position = st.sidebar.selectbox("Position", POSITIONS, index=2)

st.sidebar.markdown("**Training load**")
acute_load = st.sidebar.slider("Acute load (7-day avg)", 0, 1200, 520)
chronic_load = st.sidebar.slider("Chronic load (28-day avg)", 1, 1200, 470)

st.sidebar.markdown("**Physiology**")
sleep_hours = st.sidebar.slider("Sleep (h/night)", 3.0, 10.0, 7.4, 0.1)
soreness = st.sidebar.slider("Soreness (0–10)", 0.0, 10.0, 3.0, 0.5)
resting_hr = st.sidebar.slider("Resting HR (bpm)", 40, 95, 55)
baseline_hr = st.sidebar.slider("Usual resting HR (baseline)", 40, 80, 55)

st.sidebar.markdown("**History**")
injury_prone = st.sidebar.checkbox("Injury proneness", value=False)
previous_injuries = st.sidebar.number_input("Previous injuries", 0, 20, 1)
days_since_injury = st.sidebar.number_input("Days since last injury", 0, 2000, 200)

# --------------------------------------------------------------------------- #
# Scoring — all the logic lives in the package, this file only renders it.
# --------------------------------------------------------------------------- #
acwr = acute_load / chronic_load if chronic_load else 1.0
zone = acwr_zone(acwr)

# One evaluation produces the score, its level and its explanation — so the gauge
# and the factor list below are, by construction, the same computation.
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

# --------------------------------------------------------------------------- #
# Main body
# --------------------------------------------------------------------------- #
st.title("🩺 Muscle injury risk detection")
st.caption(
    "Score computed in real time from business rules (ACWR, sleep, soreness, "
    "HR, injury history). Recall comes first: better a false alarm than a "
    "missed injury."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Risk score", f"{score:.2f}", help="0 = low, 1 = very high")
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
    # Simple gauge via a colored progress bar.
    st.metric("Risk gauge", f"{int(score * 100)} %")
    st.progress(min(int(score * 100), 100))

st.divider()

# --------------------------------------------------------------------------- #
# Active risk factors — this *is* the score's decomposition, so it can never
# contradict the gauge above.
# --------------------------------------------------------------------------- #
st.subheader("Active risk factors")

if factors:
    st.caption(
        "Each factor shows how many risk points it contributes. They sum to the "
        "score above — so nothing can raise the gauge without appearing here."
    )
    for factor in factors:
        icon = SEVERITY_ICONS.get(factor.severity, "🟠")
        st.markdown(
            f"- {icon} **{factor.label}** — {factor.detail} " f"&nbsp;`+{factor.percent} pts`",
            unsafe_allow_html=True,
        )
else:
    st.success("No risk factor detected ✅")

st.divider()

# --------------------------------------------------------------------------- #
# SHAP section (if a model is trained)
# --------------------------------------------------------------------------- #
st.subheader("Model explainability (SHAP)")

summary_synth = FIGURES_DIR / "shap_summary_synthetic.png"
waterfall_synth = FIGURES_DIR / "shap_waterfall_synthetic_athlete0.png"

if summary_synth.exists() or waterfall_synth.exists():
    c1, c2 = st.columns(2)
    if summary_synth.exists():
        c1.image(str(summary_synth), caption="Global feature importance")
    if waterfall_synth.exists():
        c2.image(str(waterfall_synth), caption="Individual explanation (waterfall)")
else:
    st.info(
        "SHAP plots will show up here after the model is trained:\n\n"
        "```\ninjury-risk train --track synthetic\n"
        "injury-risk shap --track synthetic\n```"
    )

with st.expander("ℹ️ About the data & limitations"):
    st.markdown(
        "- **Synthetic dataset** (200 athletes × 730 days): the only basis allowing "
        "a realistic temporal ACWR/rolling and 3 imbalanced classes.\n"
        "- **Real SIRP-600 dataset** (600 athletes): validates the approach on "
        "imperfect data, but snapshot (no ACWR) and binary target.\n"
        "- The live score above is **based on business rules**, not on the XGBoost "
        "model (which is used for validation and SHAP analysis)."
    )
