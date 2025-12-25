# ============================================================
# KeraRisk-CXL — Online Risk Calculator (Streamlit App)
# Uses trained models saved as .skops (safe & portable)
# Endpoints:
#   A — ΔKmax progression
#   B — Kmax slope (D/year)
#   C — Composite clinical endpoint
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import skops.io as sio
import json

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="KeraRisk-CXL Calculator",
    layout="centered"
)

st.title("🩺 KeraRisk-CXL Calculator")
st.markdown(
    """
Clinical decision-support tool for **risk stratification and progression modeling in keratoconus**  
based on baseline tomographic and functional parameters.
"""
)

# ------------------------------------------------------------
# LOAD MODELS + METADATA
# ------------------------------------------------------------
@st.cache_resource
def load_assets():
    trusted = [
        "sklearn.pipeline.Pipeline",
        "sklearn.compose._column_transformer.ColumnTransformer",
        "sklearn.impute._base.SimpleImputer",
        "sklearn.preprocessing._encoders.OneHotEncoder",
        "sklearn.preprocessing._data.StandardScaler",
        "sklearn.linear_model._logistic.LogisticRegression",
        "sklearn.linear_model._coordinate_descent.ElasticNet",
    ]

    modelA = sio.load("assets/KeraRisk_modelA.skops", trusted=trusted)
    modelB = sio.load("assets/KeraRisk_modelB.skops", trusted=trusted)
    modelC = sio.load("assets/KeraRisk_modelC.skops", trusted=trusted)

    with open("assets/kerarisk_meta.json", "r") as f:
        meta = json.load(f)

    return modelA, modelB, modelC, meta


modelA, modelB, modelC, meta = load_assets()

# ------------------------------------------------------------
# SIDEBAR — INPUTS
# ------------------------------------------------------------
st.sidebar.header("📥 Patient baseline data")

age = st.sidebar.number_input("Age (years)", 8, 80, 18)
kmax0 = st.sidebar.number_input("Kmax (D)", 40.0, 90.0, 55.0, step=0.1)
pachy0 = st.sidebar.number_input("Minimum pachymetry (µm)", 300, 600, 450)
bcva0 = st.sidebar.number_input("BCVA (decimal)", 0.0, 1.2, 0.8, step=0.05)
cyl0 = st.sidebar.number_input("Cylinder (D)", 0.0, 15.0, 4.0, step=0.25)
group = st.sidebar.selectbox("Treatment group", meta["groups"])

X = pd.DataFrame([{
    "age": age,
    "kmax0": kmax0,
    "pachy0": pachy0,
    "bcva0": bcva0,
    "cyl0": cyl0,
    "group": group
}])

# ------------------------------------------------------------
# PREDICTIONS
# ------------------------------------------------------------
risk_A = float(modelA.predict_proba(X)[0, 1])
risk_C = float(modelC.predict_proba(X)[0, 1])
slope_B = float(modelB.predict(X)[0])

# ------------------------------------------------------------
# OUTPUTS
# ------------------------------------------------------------
st.subheader("📊 Model outputs")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Endpoint A\nΔKmax progression",
        f"{risk_A*100:.1f} %",
        help="Probability of ΔKmax ≥ threshold within 5 years"
    )

with col2:
    st.metric(
        "Endpoint C\nComposite risk",
        f"{risk_C*100:.1f} %",
        help="Composite endpoint: ΔKmax / BCVA / Cylinder"
    )

with col3:
    st.metric(
        "Endpoint B\nKmax slope",
        f"{slope_B:+.2f} D/year",
        help="Predicted annual Kmax progression rate"
    )

# ------------------------------------------------------------
# RISK INTERPRETATION
# ------------------------------------------------------------
st.subheader("🧠 Clinical interpretation")

def risk_tier(p):
    if p < 0.15:
        return "Low"
    elif p < 0.35:
        return "Intermediate"
    else:
        return "High"

tier = risk_tier(risk_C)

st.markdown(
    f"""
**Composite risk category (Endpoint C):**  
### **{tier} risk**
"""
)

if tier == "High":
    st.warning("⚠️ High risk of clinical progression. Consider close monitoring or early intervention.")
elif tier == "Intermediate":
    st.info("ℹ️ Intermediate risk. Regular follow-up recommended.")
else:
    st.success("✅ Low risk. Standard follow-up may be sufficient.")

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("---")
st.caption(
    "KeraRisk-CXL | Research use only — not a diagnostic device. "
    "Model trained and validated as described in the associated manuscript."
)


