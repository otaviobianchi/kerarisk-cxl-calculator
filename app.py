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
# CONFIG: FILENAMES
# (Expected in repo ROOT, same folder as app.py)
# -----------------------------
MODEL_A_NAME = "KeraRisk_modelA.skops"
MODEL_B_NAME = "KeraRisk_modelB.skops"
MODEL_C_NAME = "KeraRisk_modelC.skops"
META_NAME    = "kerarisk_meta.json"


# -----------------------------
# LOAD ASSETS (NO UI INSIDE CACHE)
# Looks in ROOT first; if not found, tries assets/
# -----------------------------
@st.cache_resource
def load_assets():
    base = Path(__file__).resolve().parent

    # 1) Prefer ROOT (repo root where app.py lives)
    root_dir = base

    # 2) Fallback to assets/
    assets_dir = base / "assets"

    def resolve_file(filename: str) -> Path:
        p1 = root_dir / filename
        if p1.exists():
            return p1
        p2 = assets_dir / filename
        if p2.exists():
            return p2
        return p1  # expected (for clean error msg)

    modelA_path = resolve_file(MODEL_A_NAME)
    modelB_path = resolve_file(MODEL_B_NAME)
    modelC_path = resolve_file(MODEL_C_NAME)
    meta_path   = resolve_file(META_NAME)

    # hard checks
    missing = [p for p in [modelA_path, modelB_path, modelC_path, meta_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required file(s):\n"
            + "\n".join([f"- {p.name} (looked in: {p.parent})" for p in missing])
            + f"\n\nBase directory: {base}"
            + f"\nAlso checked: {assets_dir}"
        )

    # ✅ Trusted types for skops safe loading (FIXES Untrusted types)
    trusted = [
        # sklearn core
        "sklearn.pipeline.Pipeline",
        "sklearn.compose._column_transformer.ColumnTransformer",
        "sklearn.compose._column_transformer._RemainderColsList",

        # preprocessing
        "sklearn.impute._base.SimpleImputer",
        "sklearn.preprocessing._encoders.OneHotEncoder",
        "sklearn.preprocessing._data.StandardScaler",

        # models
        "sklearn.linear_model._logistic.LogisticRegression",
        "sklearn.linear_model._coordinate_descent.ElasticNet",

        # numpy internals (safe)
        "numpy.ndarray",
        "numpy.dtype",
    ]

    modelA = sio.load(modelA_path, trusted=trusted)
    modelB = sio.load(modelB_path, trusted=trusted)
    modelC = sio.load(modelC_path, trusted=trusted)

    meta = json.loads(meta_path.read_text())

    # Ensure meta["groups"]
    if "groups" not in meta or not meta["groups"]:
        try:
            # Infer categories from OneHotEncoder if present
            ohe = (
                modelC.named_steps["pre"]
                      .named_transformers_["cat"]
                      .named_steps["oh"]
            )
            meta["groups"] = [str(x) for x in list(ohe.categories_[0])]
        except Exception:
            meta["groups"] = ["FRAK", "FRAKcross"]

    used_paths = {
        "modelA": str(modelA_path),
        "modelB": str(modelB_path),
        "modelC": str(modelC_path),
        "meta": str(meta_path),
        "base": str(base),
        "assets_dir": str(assets_dir),
    }

    return modelA, modelB, modelC, meta, used_paths


# -----------------------------
# SAFE LOADING WRAPPER + DEBUG
# -----------------------------
debug = st.sidebar.checkbox("Debug (show file paths)", value=False)

try:
    modelA, modelB, modelC, meta, used_paths = load_assets()
except Exception as e:
    st.error("❌ Failed to load models/assets.")
    st.code(str(e))
    st.stop()

if debug:
    st.sidebar.write("Resolved paths:")
    st.sidebar.json(used_paths)

    base = Path(used_paths["base"])
    st.sidebar.write("Files in base:")
    st.sidebar.write(sorted([p.name for p in base.iterdir() if p.is_file()]))

    assets_dir = Path(used_paths["assets_dir"])
    if assets_dir.exists():
        st.sidebar.write("Files in assets/:")
        st.sidebar.write(sorted([p.name for p in assets_dir.iterdir() if p.is_file()]))
    else:
        st.sidebar.write("assets/ folder does not exist (OK).")


# -----------------------------
# INPUTS
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

def tier(p: float) -> str:
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





