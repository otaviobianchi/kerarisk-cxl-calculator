# ============================================================
# KeraRisk-CXL — Online Risk Calculator (Streamlit App)
# Uses trained models saved as .skops
# Endpoints:
#   A — ΔKmax progression
#   B — Kmax slope (D/year)
#   C — Composite clinical endpoint
# ============================================================

import streamlit as st
import skops.io as sio
import json
import pandas as pd
from pathlib import Path

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
Clinical decision-support tool for **risk stratification and progression modeling in keratoconus**,  
based on baseline tomographic and functional parameters.
"""
)

# ------------------------------------------------------------
# LOAD MODELS + METADATA (SAFE)
# ------------------------------------------------------------
@st.cache_resource
def load_assets():
    base = Path(__file__).parent
    assets = base / "assets"

    # ---- Debug (can remove later)
    st.write("📂 Assets directory:", assets.resolve())
    if not assets.exists():
        st.error("❌ Assets directory not found")
        st.stop()

    st.write("📄 Files found:", [p.name for p in assets.iterdir()])

    modelA_path = assets / "KeraRisk_modelA.skops"
    modelB_path = assets / "KeraRisk_modelB.skops"
    modelC_path = assets / "KeraRisk_modelC.skops"
    meta_path   = assets / "kerarisk_meta.json"

    for p in [modelA_path, modelB_path, modelC_path, meta_path]:
        if not p.exists():
            st.error(f"❌ Missing file: {p.name}")
            st.stop()

    trusted = [
        "sklearn.pipeline.Pipeline",
        "sklearn.compose._column_transformer.ColumnTransformer",
        "sklearn.impute._base.SimpleImputer",
        "sklearn.preprocessing._encoders.OneHotEncoder",
        "sklearn.preprocessing._data.StandardScaler",
        "sklearn.linear_model._logistic.LogisticRegression",
        "sklearn.linear_model._coordinate_descent.ElasticNet",
        "numpy.ndarray",
    ]

    modelA = sio.load(modelA_path, trusted=trusted)
    modelB = sio.load(modelB_path, trusted=trusted)
    modelC = sio.load(modelC_path, trusted=trusted)

    with open(meta_path, "r") as f:
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
    )

with col2:
    st.metric(
        "Endpoint C\nComposite risk",
        f"{risk_C*100:.1f} %",
    )

with col3:
    st.metric(
        "Endpoint B\nKmax slope",
        f"{slope_B:+.2f} D/year",
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

st.markdown(f"### **{tier} risk**")

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


