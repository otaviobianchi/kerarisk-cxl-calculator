import json
from pathlib import Path

import pandas as pd
import streamlit as st
import skops.io as sio


# -----------------------------
# PAGE
# -----------------------------
st.set_page_config(page_title="KeraRisk-CXL Calculator", layout="centered")
st.title("🩺 KeraRisk-CXL Calculator")
st.caption("Research use only — not a diagnostic device.")


# -----------------------------
# LOAD ASSETS (NO UI HERE)
# -----------------------------
@st.cache_resource
def load_assets():
    base = Path(__file__).resolve().parent
    assets = base / "assets"

    modelA_path = assets / "KeraRisk_modelA.skops"
    modelB_path = assets / "KeraRisk_modelB.skops"
    modelC_path = assets / "KeraRisk_modelC.skops"
    meta_path   = assets / "kerarisk_meta.json"

    # hard checks
    if not assets.exists():
        raise FileNotFoundError(f"assets/ not found at: {assets}")

    for p in [modelA_path, modelB_path, modelC_path, meta_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

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

    meta = json.loads(meta_path.read_text())

    # fallback: if meta doesn't have groups, infer from training categories
    if "groups" not in meta or not meta["groups"]:
        try:
            # works if OneHotEncoder was used on "group"
            ohe = modelC.named_steps["pre"].named_transformers_["cat"].named_steps["oh"]
            meta["groups"] = list(ohe.categories_[0])
        except Exception:
            meta["groups"] = ["FRAK", "FRAKcross"]  # safe fallback

    return modelA, modelB, modelC, meta, assets


# -----------------------------
# SAFE LOADING WRAPPER + DEBUG UI
# -----------------------------
debug = st.sidebar.checkbox("Debug (show assets)", value=False)

try:
    modelA, modelB, modelC, meta, assets_dir = load_assets()
except Exception as e:
    st.error("❌ Failed to load models/assets.")
    st.code(str(e))
    st.stop()

if debug:
    st.sidebar.write("Base:", Path(__file__).resolve().parent)
    st.sidebar.write("Assets:", assets_dir)
    st.sidebar.write("Files:", sorted([p.name for p in assets_dir.iterdir()]))

# -----------------------------
# INPUTS (UI ONLY HERE)
# -----------------------------
st.sidebar.header("📥 Patient baseline data")

age = st.sidebar.number_input("Age (years)", min_value=8, max_value=80, value=18)
kmax0 = st.sidebar.number_input("Kmax (D)", min_value=40.0, max_value=90.0, value=55.0, step=0.1)
pachy0 = st.sidebar.number_input("Minimum pachymetry (µm)", min_value=300, max_value=650, value=450)
bcva0 = st.sidebar.number_input("BCVA (decimal)", min_value=0.0, max_value=1.2, value=0.8, step=0.05)
cyl0 = st.sidebar.number_input("Cylinder (D)", min_value=0.0, max_value=20.0, value=4.0, step=0.25)

group = st.sidebar.selectbox("Treatment group", meta["groups"])

X = pd.DataFrame([{
    "age": float(age),
    "kmax0": float(kmax0),
    "pachy0": float(pachy0),
    "bcva0": float(bcva0),
    "cyl0": float(cyl0),
    "group": str(group),
}])


# -----------------------------
# PREDICTIONS
# -----------------------------
risk_A = float(modelA.predict_proba(X)[0, 1])
risk_C = float(modelC.predict_proba(X)[0, 1])
slope_B = float(modelB.predict(X)[0])


# -----------------------------
# OUTPUTS
# -----------------------------
st.subheader("📊 Model outputs")
c1, c2, c3 = st.columns(3)
c1.metric("Endpoint A (ΔKmax progression)", f"{risk_A*100:.1f}%")
c2.metric("Endpoint C (Composite risk)", f"{risk_C*100:.1f}%")
c3.metric("Endpoint B (Kmax slope)", f"{slope_B:+.2f} D/year")

st.subheader("🧠 Clinical interpretation (Endpoint C)")

def tier(p):
    if p < 0.15:
        return "Low"
    if p < 0.35:
        return "Intermediate"
    return "High"

t = tier(risk_C)
st.markdown(f"### **{t} risk**")

if t == "High":
    st.warning("⚠️ High risk of clinical progression. Consider closer monitoring / earlier intervention.")
elif t == "Intermediate":
    st.info("ℹ️ Intermediate risk. Consider 6–12 month follow-up.")
else:
    st.success("✅ Low risk. Standard follow-up may be sufficient.")

st.markdown("---")
st.caption("KeraRisk-CXL | Research use only. Validate locally before clinical deployment.")



