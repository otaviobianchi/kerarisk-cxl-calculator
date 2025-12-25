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
# Helpers
# -----------------------------
def resolve_paths(base: Path, filename: str) -> Path:
    """Look in repo root first; if not found, try assets/."""
    p1 = base / filename
    if p1.exists():
        return p1
    p2 = base / "assets" / filename
    if p2.exists():
        return p2
    return p1  # return expected path (clean error message)


def safe_skops_load(path: Path, extra_trusted=None, debug=False):
    """
    Robust skops loader across sklearn versions:
    1) try strict load with a minimal trusted list
    2) if it fails with untrusted types, retrieve them and retry
    """
    # minimal safe baseline (these are stable)
    trusted = [
        "sklearn.pipeline.Pipeline",
        "sklearn.compose._column_transformer.ColumnTransformer",
        "sklearn.impute._base.SimpleImputer",
        "sklearn.preprocessing._encoders.OneHotEncoder",
        "sklearn.preprocessing._data.StandardScaler",
        "sklearn.linear_model._logistic.LogisticRegression",
        "sklearn.linear_model._coordinate_descent.ElasticNet",
        "numpy.ndarray",
        "numpy.dtype",
    ]
    if extra_trusted:
        trusted.extend(list(extra_trusted))

    # 1) attempt
    try:
        return sio.load(path, trusted=trusted), trusted, []
    except Exception as e:
        msg = str(e)
        if "Untrusted types found in the file" not in msg:
            raise  # genuine error (file missing/corrupt/etc.)

    # 2) get actual untrusted types
    untrusted = sio.get_untrusted_types(path)
    # convert to strings (skops accepts list[str])
    untrusted_str = [str(t) for t in untrusted]

    # 3) retry trusting exactly what's required
    trusted2 = sorted(set(trusted + untrusted_str))

    if debug:
        st.sidebar.write(f"🔐 Auto-trusting {len(untrusted_str)} extra type(s) for {path.name}:")
        st.sidebar.code("\n".join(untrusted_str))

    obj = sio.load(path, trusted=trusted2)
    return obj, trusted2, untrusted_str


# -----------------------------
# LOAD ASSETS
# -----------------------------
@st.cache_resource
def load_assets(debug_flag=False):
    base = Path(__file__).resolve().parent

    modelA_path = resolve_paths(base, MODEL_A_NAME)
    modelB_path = resolve_paths(base, MODEL_B_NAME)
    modelC_path = resolve_paths(base, MODEL_C_NAME)
    meta_path   = resolve_paths(base, META_NAME)

    missing = [p for p in [modelA_path, modelB_path, modelC_path, meta_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required file(s):\n"
            + "\n".join([f"- {p.name} (expected at: {p})" for p in missing])
            + f"\n\nBase directory: {base}"
            + f"\nAlso checked: {base/'assets'}"
        )

    # load meta
    meta = json.loads(meta_path.read_text())

    # robust skops loading
    modelA, trustedA, extraA = safe_skops_load(modelA_path, debug=debug_flag)
    modelB, trustedB, extraB = safe_skops_load(modelB_path, debug=debug_flag)
    modelC, trustedC, extraC = safe_skops_load(modelC_path, debug=debug_flag)

    # Ensure meta["groups"]
    if "groups" not in meta or not meta["groups"]:
        try:
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
        "assets_dir": str(base / "assets"),
    }

    trusted_report = {
        "modelA_trusted_count": len(trustedA),
        "modelB_trusted_count": len(trustedB),
        "modelC_trusted_count": len(trustedC),
        "modelA_extra_trusted": extraA,
        "modelB_extra_trusted": extraB,
        "modelC_extra_trusted": extraC,
    }

    return modelA, modelB, modelC, meta, used_paths, trusted_report


# -----------------------------
# DEBUG UI
# -----------------------------
debug = st.sidebar.checkbox("Debug (show paths & trusted types)", value=False)

try:
    modelA, modelB, modelC, meta, used_paths, trusted_report = load_assets(debug_flag=debug)
except Exception as e:
    st.error("❌ Failed to load models/assets.")
    st.code(str(e))
    st.stop()

if debug:
    st.sidebar.write("Resolved paths:")
    st.sidebar.json(used_paths)
    st.sidebar.write("Trusted report:")
    st.sidebar.json(trusted_report)


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
