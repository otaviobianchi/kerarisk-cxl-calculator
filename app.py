import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="KeraRisk-CXL Calculator", layout="centered")

@st.cache_resource
def load_assets():
    model = joblib.load("KeraRisk_modelC.joblib")
    with open("KeraRisk_metadata.json", "r") as f:
        meta = json.load(f)
    return model, meta

model, meta = load_assets()

st.title("KeraRisk-CXL — Clinical Risk Calculator (Research Use)")
st.caption("Predicts 5-year risk for Composite Endpoint (C) using baseline variables.")

with st.expander("Endpoint definition"):
    st.write(meta.get("endpoint_definition", ""))

# --- Inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=5.0, max_value=80.0, value=18.0, step=0.5)
    kmax0 = st.number_input("Baseline Kmax (D)", min_value=35.0, max_value=100.0, value=54.5, step=0.1)
    pachy0 = st.number_input("Baseline Pachymetry min (µm)", min_value=250.0, max_value=700.0, value=435.0, step=1.0)

with col2:
    bcva0 = st.number_input("Baseline BCVA (decimal)", min_value=0.0, max_value=2.0, value=0.9, step=0.01)
    cyl0 = st.number_input("Baseline Cylinder (D)", min_value=0.0, max_value=20.0, value=4.5, step=0.1)
    group = st.selectbox("Group", options=meta.get("groups_allowed", ["FRAK", "FRAKcross"]))

# --- Predict
x = pd.DataFrame([{
    "age": float(age),
    "kmax0": float(kmax0),
    "pachy0": float(pachy0),
    "bcva0": float(bcva0),
    "cyl0": float(cyl0),
    "group": str(group)
}])

if st.button("Calculate 5-year risk"):
    p = float(model.predict_proba(x)[0, 1])
    st.subheader(f"Predicted 5-year risk (Endpoint C): {p*100:.1f}%")

    # risk tier (ajuste os cortes como preferir)
    if p < 0.15:
        tier = "Low"
        msg = "Consider annual follow-up (context-dependent)."
    elif p < 0.35:
        tier = "Intermediate"
        msg = "Consider closer follow-up (e.g., 6–12 months)."
    else:
        tier = "High"
        msg = "Consider closer monitoring and discussion of early intervention."

    st.write(f"**Risk tier:** {tier}")
    st.info(msg)

    st.caption("Disclaimer: Research tool. Not for direct clinical decision-making without external validation.")
