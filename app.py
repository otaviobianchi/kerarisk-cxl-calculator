import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import skops.io as sio

APP_TITLE = "KeraRisk-CXL — Clinical Risk Calculator"

MODEL_A_FILE = "KeraRisk_modelA.skops"
MODEL_B_FILE = "KeraRisk_modelB.skops"
MODEL_C_FILE = "KeraRisk_modelC.skops"
META_FILE = "meta.json"  # optional

DEFAULT_FEATURES = ["age", "kmax0", "pachy0", "bcva0", "cyl0", "group"]

# Default thresholds / UI bins (adjust if you want)
DEFAULTS = {
    "endpoint_A_threshold_dkmax": 1.5,
    "endpoint_C_threshold_dkmax": 1.5,
    "endpoint_C_bcva_drop": 0.2,
    "endpoint_C_cyl_increase": 1.0,
    "risk_tiers": [
        {"name": "Low", "max": 0.15, "hint": "Annual follow-up"},
        {"name": "Intermediate", "max": 0.35, "hint": "Consider 6-month follow-up"},
        {"name": "High", "max": 1.00, "hint": "Closer monitoring / consider early CXL"},
    ],
}


def _safe_read_json(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_resource(show_spinner=False)
def load_assets():
    """
    Loads models and metadata.
    Uses skops to avoid joblib/pickle portability issues.
    """
    meta = {**DEFAULTS}
    meta.update(_safe_read_json(Path(META_FILE)))

    # Validate expected files
    missing = [f for f in [MODEL_A_FILE, MODEL_B_FILE, MODEL_C_FILE] if not Path(f).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model file(s): " + ", ".join(missing) +
            "\nUpload them to the repo root (same folder as app.py)."
        )

    modelA = sio.load(MODEL_A_FILE, trusted=True)
    modelB = sio.load(MODEL_B_FILE, trusted=True)
    modelC = sio.load(MODEL_C_FILE, trusted=True)

    # Features order can be specified in meta.json, else defaults
    features = meta.get("features", DEFAULT_FEATURES)
    groups = meta.get("groups", ["FRAK", "FRAKcross"])

    return modelA, modelB, modelC, meta, features, groups


def make_input_df(age, kmax0, pachy0, bcva0, cyl0, group, features):
    row = {
        "age": float(age),
        "kmax0": float(kmax0),
        "pachy0": float(pachy0),
        "bcva0": float(bcva0),
        "cyl0": float(cyl0),
        "group": str(group),
    }
    # Ensure column order
    df = pd.DataFrame([row])
    # Add any missing columns (future-proof)
    for c in features:
        if c not in df.columns:
            df[c] = np.nan
    return df[features]


def predict_prob(model, X):
    # Works for sklearn classifiers with predict_proba
    p = float(model.predict_proba(X)[0, 1])
    return float(np.clip(p, 0.0, 1.0))


def risk_tier(p, tiers):
    for t in tiers:
        if p <= float(t["max"]):
            return t["name"], t.get("hint", "")
    return tiers[-1]["name"], tiers[-1].get("hint", "")


def format_pct(p):
    return f"{100*p:.1f}%"


def main():
    st.set_page_config(page_title=APP_TITLE, layout="centered")
    st.title(APP_TITLE)
    st.caption("Educational decision-support tool. Not a substitute for clinical judgment.")

    with st.spinner("Loading models..."):
        modelA, modelB, modelC, meta, features, groups = load_assets()

    # Sidebar — model and endpoint info
    with st.sidebar:
        st.header("About")
        st.write(
            "This calculator estimates:\n"
            "- **Endpoint A**: ΔKmax progression within 5 years (binary)\n"
            "- **Endpoint B**: annual **Kmax slope** (D/year)\n"
            "- **Endpoint C**: composite endpoint within 5 years (binary)\n"
        )
        st.subheader("Endpoints (study definitions)")
        st.write(
            f"**A:** max(Kmax≤5y) − Kmax0 ≥ **{meta.get('endpoint_A_threshold_dkmax', 1.5)} D**\n\n"
            f"**C:** (ΔKmax ≥ **{meta.get('endpoint_C_threshold_dkmax', 1.5)} D**) OR "
            f"(BCVA drop ≥ **{meta.get('endpoint_C_bcva_drop', 0.2)}**) OR "
            f"(Cylinder increase ≥ **{meta.get('endpoint_C_cyl_increase', 1.0)} D**)"
        )
        st.subheader("Model inputs")
        st.code(", ".join(features))

    st.subheader("Patient baseline inputs")

    # Input form
    with st.form("patient_form"):
        col1, col2 = st.columns(2)

        with col1:
            group = st.selectbox("Group", options=groups, index=0)
            age = st.number_input("Age (years)", min_value=5, max_value=90, value=18, step=1)
            kmax0 = st.number_input("Kmax at baseline (D)", min_value=30.0, max_value=90.0, value=54.5, step=0.1)

        with col2:
            pachy0 = st.number_input("Min pachymetry at baseline (µm)", min_value=200.0, max_value=700.0, value=435.0, step=1.0)
            bcva0 = st.number_input("BCVA at baseline (decimal)", min_value=0.0, max_value=2.0, value=0.9, step=0.01)
            cyl0 = st.number_input("Cylinder at baseline (D)", min_value=0.0, max_value=20.0, value=4.5, step=0.05)

        submitted = st.form_submit_button("Calculate")

    if not submitted:
        st.info("Fill the inputs and click **Calculate**.")
        return

    X = make_input_df(age, kmax0, pachy0, bcva0, cyl0, group, features)

    # Predictions
    try:
        pA = predict_prob(modelA, X)
    except Exception as e:
        pA = None
        st.error(f"Endpoint A prediction failed: {e}")

    try:
        slopeB = float(modelB.predict(X)[0])
    except Exception as e:
        slopeB = None
        st.error(f"Endpoint B prediction failed: {e}")

    try:
        pC = predict_prob(modelC, X)
    except Exception as e:
        pC = None
        st.error(f"Endpoint C prediction failed: {e}")

    st.divider()
    st.subheader("Results")

    # Display as metrics
    m1, m2, m3 = st.columns(3)

    with m1:
        if pC is not None:
            tier_name, tier_hint = risk_tier(pC, meta.get("risk_tiers", DEFAULTS["risk_tiers"]))
            st.metric("Endpoint C risk (5y)", format_pct(pC), help="Composite endpoint probability within 5 years.")
            st.write(f"**Tier:** {tier_name}")
            st.caption(tier_hint)
        else:
            st.metric("Endpoint C risk (5y)", "—")

    with m2:
        if pA is not None:
            st.metric("Endpoint A risk (5y)", format_pct(pA), help="ΔKmax progression probability within 5 years.")
            st.caption("Note: Endpoint A may be rare; probabilities can be less stable.")
        else:
            st.metric("Endpoint A risk (5y)", "—")

    with m3:
        if slopeB is not None:
            st.metric("Endpoint B slope", f"{slopeB:+.2f} D/year", help="Predicted annual Kmax slope.")
        else:
            st.metric("Endpoint B slope", "—")

    # Simple interpretation
    st.subheader("Clinical interpretation (suggested)")
    bullets = []
    if pC is not None:
        tier_name, tier_hint = risk_tier(pC, meta.get("risk_tiers", DEFAULTS["risk_tiers"]))
        bullets.append(f"Composite risk (Endpoint C) is **{format_pct(pC)}** → **{tier_name}**.")
        if tier_hint:
            bullets.append(f"Follow-up suggestion: {tier_hint}.")
    if slopeB is not None:
        bullets.append(f"Predicted Kmax slope: **{slopeB:+.2f} D/year**.")
    if pA is not None:
        bullets.append(f"ΔKmax progression risk (Endpoint A): **{format_pct(pA)}**.")
    if bullets:
        st.write("- " + "\n- ".join(bullets))

    # Export / copy
    st.subheader("Export (for notes / EMR)")
    export = {
        "group": group,
        "age": float(age),
        "kmax0_D": float(kmax0),
        "pachy0_um": float(pachy0),
        "bcva0_decimal": float(bcva0),
        "cyl0_D": float(cyl0),
        "endpointC_risk_5y": None if pC is None else float(pC),
        "endpointA_risk_5y": None if pA is None else float(pA),
        "endpointB_slope_D_per_year": None if slopeB is None else float(slopeB),
    }
    st.code(json.dumps(export, indent=2))

    # Optional: quick “what-if” group switch
    st.subheader("What-if: compare groups (same baseline)")
    if len(groups) >= 2:
        g1, g2 = groups[0], groups[1]
        X1 = make_input_df(age, kmax0, pachy0, bcva0, cyl0, g1, features)
        X2 = make_input_df(age, kmax0, pachy0, bcva0, cyl0, g2, features)

        try:
            pC1 = predict_prob(modelC, X1)
            pC2 = predict_prob(modelC, X2)
            st.write(f"Endpoint C risk: **{g1} = {format_pct(pC1)}** vs **{g2} = {format_pct(pC2)}**")
        except Exception:
            st.caption("Could not compute group comparison for Endpoint C.")

    st.divider()
    st.caption(
        "Disclaimer: This tool is for research/education. "
        "Clinical decisions should consider full clinical context and confirmatory testing."
    )


if __name__ == "__main__":
    main()
