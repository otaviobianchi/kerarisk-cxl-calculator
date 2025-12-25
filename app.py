# app.py
import json
import re
import ast
from pathlib import Path
from datetime import datetime, timezone

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
# CONFIG: FILENAMES (repo ROOT)
# -----------------------------
MODEL_A_NAME = "KeraRisk_modelA.skops"
MODEL_B_NAME = "KeraRisk_modelB.skops"
MODEL_C_NAME = "KeraRisk_modelC.skops"
META_NAME = "kerarisk_meta.json"


# -----------------------------
# Helpers
# -----------------------------
def resolve_paths(base: Path, filename: str) -> Path:
    """Prefer repo root; fallback to assets/ if present."""
    p1 = base / filename
    if p1.exists():
        return p1
    p2 = base / "assets" / filename
    if p2.exists():
        return p2
    return p1


def _extract_untrusted_types_from_message(msg: str) -> list[str]:
    """
    skops may report untrusted types in different formats:
    - "Untrusted types found in the file: ['a.b.Type', 'c.d.Other']"
    - multiline list with hyphens:
        Untrusted types found in the file:
        - a.b.Type
        - c.d.Other
    This helper supports both.
    """
    m = re.search(r"Untrusted types found in the file:\s*(\[[\s\S]*\])", msg)
    if m:
        try:
            parsed = ast.literal_eval(m.group(1))
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass

    extra = re.findall(r"^\s*-\s*(.+?)\s*$", msg, flags=re.MULTILINE)
    return [x.strip() for x in extra if x.strip()]


def safe_skops_load(path: Path, debug=False):
    """
    CVE-2024-37065 safe loader for skops >= 0.10
    - trusted MUST be list[str]
    - NEVER uses trusted=True
    - NEVER calls get_untrusted_types(path)

    Strategy:
    1) try baseline trusted list
    2) if blocked, parse required untrusted types from error message and retry
    """
    trusted_base = [
        "sklearn.pipeline.Pipeline",
        "sklearn.compose._column_transformer.ColumnTransformer",
        "sklearn.compose._column_transformer._RemainderColsList",
        "sklearn.impute._base.SimpleImputer",
        "sklearn.preprocessing._encoders.OneHotEncoder",
        "sklearn.preprocessing._data.StandardScaler",
        "sklearn.linear_model._logistic.LogisticRegression",
        "sklearn.linear_model._coordinate_descent.ElasticNet",
        "numpy.ndarray",
        "numpy.dtype",
    ]

    try:
        return sio.load(path, trusted=trusted_base), trusted_base, []
    except Exception as e:
        msg = str(e)
        if "Untrusted types found in the file" not in msg:
            raise RuntimeError(f"Failed to load {path.name}:\n{msg}")

        extra_types = _extract_untrusted_types_from_message(msg)
        if not extra_types:
            raise RuntimeError(
                f"skops blocked {path.name} but could not extract untrusted types.\n\nFull error:\n{msg}"
            )

        trusted_final = sorted(set(trusted_base + extra_types))

        if debug:
            st.sidebar.write(f"🔐 Auto-trusting {len(extra_types)} extra type(s) for **{path.name}**:")
            st.sidebar.code("\n".join(extra_types))

        return sio.load(path, trusted=trusted_final), trusted_final, extra_types


def tier(p: float) -> str:
    if p < 0.15:
        return "Low"
    if p < 0.35:
        return "Intermediate"
    return "High"


# -----------------------------
# LOAD ASSETS
# -----------------------------
@st.cache_resource
def load_assets(debug_flag=False):
    base = Path(__file__).resolve().parent

    modelA_path = resolve_paths(base, MODEL_A_NAME)
    modelB_path = resolve_paths(base, MODEL_B_NAME)
    modelC_path = resolve_paths(base, MODEL_C_NAME)
    meta_path = resolve_paths(base, META_NAME)

    missing = [p for p in [modelA_path, modelB_path, modelC_path, meta_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required file(s):\n"
            + "\n".join([f"- {p.name} (expected at: {p})" for p in missing])
            + f"\n\nBase directory: {base}"
            + f"\nAlso checked: {base/'assets'}"
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

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
debug = st.sidebar.checkbox("Debug (paths & trusted types)", value=False)

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

group = st.sidebar.selectbox("Treatment group", meta.get("groups", ["FRAK", "FRAKcross"]))

X = pd.DataFrame([{
    "age": float(age),
    "kmax0": float(kmax0),
    "pachy0": float(pachy0),
    "bcva0": float(bcva0),
    "cyl0": float(cyl0),
    "group": str(group),
}])


# -----------------------------
# INPUT ECHO + PLAUSIBILITY FLAGS
# -----------------------------
st.subheader("🧾 Inputs (echo)")
c_in1, c_in2, c_in3 = st.columns(3)
c_in1.metric("Age (y)", f"{float(age):.0f}")
c_in2.metric("Kmax (D)", f"{float(kmax0):.1f}")
c_in3.metric("Min pachy (µm)", f"{float(pachy0):.0f}")

c_in4, c_in5, c_in6 = st.columns(3)
c_in4.metric("BCVA (decimal)", f"{float(bcva0):.2f}")
c_in5.metric("Cylinder (D)", f"{float(cyl0):.2f}")
c_in6.metric("Group", f"{group}")

flags = []
if float(pachy0) < 400:
    flags.append("Very thin cornea (pachymetry < 400 µm).")
if float(kmax0) > 60:
    flags.append("Very steep cornea (Kmax > 60 D).")
if float(age) < 12:
    flags.append("Very young age (age < 12).")
if float(age) > 45:
    flags.append("Older age (age > 45).")

if flags:
    st.warning("⚠️ Plausibility flags (may be outside training distribution):\n- " + "\n- ".join(flags))


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
st.caption("Risk tiers: Low < 15% • Intermediate 15–35% • High ≥ 35%")

t = tier(risk_C)
st.markdown(f"### **{t} risk**")

if t == "High":
    st.warning("⚠️ High risk of clinical progression. Consider closer monitoring / earlier intervention.")
elif t == "Intermediate":
    st.info("ℹ️ Intermediate risk. Consider 6–12 month follow-up.")
else:
    st.success("✅ Low risk. Standard follow-up may be sufficient.")


# -----------------------------
# SLOPE PROJECTIONS (simple linear projection)
# -----------------------------
st.subheader("📈 Endpoint B: simple projections (linear)")
proj = pd.DataFrame(
    {
        "Horizon": ["1 year", "2 years", "3 years"],
        "Projected ΔKmax (D)": [1 * slope_B, 2 * slope_B, 3 * slope_B],
    }
)
st.dataframe(
    proj.style.format({"Projected ΔKmax (D)": "{:+.2f}"}),
    use_container_width=True,
    hide_index=True,
)
st.caption("Note: Linear projection shown for interpretability only; not a mechanistic forecast.")


# -----------------------------
# EXPORT (JSON audit bundle)
# -----------------------------
st.subheader("⬇️ Export (audit bundle)")
bundle = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "app": {"name": "KeraRisk-CXL", "purpose": "Research use only"},
    "inputs": {
        "age": float(age),
        "kmax0": float(kmax0),
        "pachy0": float(pachy0),
        "bcva0": float(bcva0),
        "cyl0": float(cyl0),
        "group": str(group),
    },
    "outputs": {
        "risk_A": risk_A,
        "risk_C": risk_C,
        "slope_B": slope_B,
        "tier_C": t,
        "tier_thresholds": {"low_lt": 0.15, "intermediate_lt": 0.35, "high_ge": 0.35},
        "slope_projection": {
            "1y": 1 * slope_B,
            "2y": 2 * slope_B,
            "3y": 3 * slope_B,
        },
    },
    "plausibility_flags": flags,
}
st.download_button(
    label="Download JSON report",
    data=json.dumps(bundle, indent=2).encode("utf-8"),
    file_name="kerarisk_report.json",
    mime="application/json",
)

st.markdown("---")
st.caption("KeraRisk-CXL | Research use only. Validate locally before clinical use.")

