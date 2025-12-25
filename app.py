# app.py
# KeraRisk-CXL Calculator (Streamlit)
# - Robust skops loader (CVE-2024-37065 compatible)
# - Optional meta (won't crash if missing)
# - Safe handling for Model B not fitted (won't crash)
# - Cache fingerprint to avoid Streamlit Cloud stale artifacts
# - Exports JSON + PDF report

import json
import re
import ast
from pathlib import Path
from datetime import datetime, timezone
from io import BytesIO

import pandas as pd
import streamlit as st
import skops.io as sio

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline


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
# Helpers: path resolution
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


# -----------------------------
# Helpers: skops safe load (CVE-2024-37065 compatible)
# -----------------------------
def _extract_untrusted_types_from_message(msg: str) -> list[str]:
    """
    skops may report untrusted types in different formats:
    - "Untrusted types found in the file: ['a.b.Type', 'c.d.Other']"
    - multiline list with hyphens:
        Untrusted types found in the file:
        - a.b.Type
        - c.d.Other
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
    Safe loader for skops >= 0.10:
    - trusted MUST be list[str]
    - NEVER uses trusted=True
    - NEVER calls get_untrusted_types(path)
    Strategy:
      try baseline trusted list; if blocked, parse required types from exception and retry.
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


# -----------------------------
# Helpers: fitted check (prevents NotFittedError crashes)
# -----------------------------
def is_fitted(est) -> bool:
    try:
        check_is_fitted(est)
        return True
    except Exception:
        pass
    if isinstance(est, Pipeline) and len(est.steps) > 0:
        try:
            check_is_fitted(est.steps[-1][1])
            return True
        except Exception:
            return False
    return False


# -----------------------------
# Helpers: risk tier
# -----------------------------
def tier(p: float) -> str:
    if p < 0.15:
        return "Low"
    if p < 0.35:
        return "Intermediate"
    return "High"


# -----------------------------
# Helpers: PDF generator
# -----------------------------
def build_pdf_report(bundle: dict) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    _, h = A4

    left = 2.0 * cm
    y = h - 2.2 * cm

    def line(text, dy=0.55 * cm, font="Helvetica", size=11):
        nonlocal y
        c.setFont(font, size)
        max_chars = 110
        chunks = [text[i : i + max_chars] for i in range(0, len(text), max_chars)] or [""]
        for ch in chunks:
            c.drawString(left, y, ch)
            y -= dy
            if y < 2.0 * cm:
                c.showPage()
                y = h - 2.2 * cm

    # Header
    line("KeraRisk-CXL Calculator — Report", dy=0.75 * cm, font="Helvetica-Bold", size=14)
    line("Research use only — not a diagnostic device.", dy=0.75 * cm, font="Helvetica-Oblique", size=10)
    line(f"Generated (UTC): {bundle.get('timestamp_utc','')}", dy=0.8 * cm, size=10)

    # Inputs
    line("Inputs", dy=0.65 * cm, font="Helvetica-Bold", size=12)
    inp = bundle.get("inputs", {})
    for k in ["age", "kmax0", "pachy0", "bcva0", "cyl0", "group"]:
        if k in inp:
            line(f"- {k}: {inp[k]}")

    # Outputs
    line("", dy=0.35 * cm)
    line("Model outputs", dy=0.65 * cm, font="Helvetica-Bold", size=12)
    out = bundle.get("outputs", {})

    if "risk_A" in out:
        line(f"- Endpoint A (ΔKmax progression): {out['risk_A']*100:.1f}%")
    if "risk_C" in out:
        line(f"- Endpoint C (Composite risk): {out['risk_C']*100:.1f}%")

    slope = out.get("slope_B", None)
    if slope is None:
        line("- Endpoint B (Kmax slope): N/A (Model B not fitted in deployed artifact)")
    else:
        line(f"- Endpoint B (Kmax slope): {float(slope):+.2f} D/year")

    if "tier_C" in out:
        line(f"- Tier (Endpoint C): {out['tier_C']}")

    # Projections (optional)
    sp = out.get("slope_projection", None)
    if isinstance(sp, dict) and sp:
        line("", dy=0.35 * cm)
        line("Endpoint B — projections", dy=0.65 * cm, font="Helvetica-Bold", size=12)
        line(f"- Method: {sp.get('method','')}")
        line(f"- Max years: {sp.get('max_years','')}")
        thr = sp.get("stability_threshold_D_per_year", None)
        if thr is not None:
            line(f"- Stability threshold: |slope| < {thr:.2f} D/year (operational)")
        line(f"- Stability assumed: {bool(sp.get('stable_assumed', False))}")

        vals = sp.get("values", [])
        if isinstance(vals, list) and vals:
            for v in vals:
                try:
                    yy = int(v.get("year"))
                    dk = float(v.get("delta_kmax_D"))
                    line(f"  • Year {yy}: ΔKmax {dk:+.2f} D")
                except Exception:
                    continue

    # Flags
    flags = bundle.get("plausibility_flags", [])
    line("", dy=0.35 * cm)
    line("Plausibility flags", dy=0.65 * cm, font="Helvetica-Bold", size=12)
    if flags:
        for f in flags:
            line(f"- {f}")
    else:
        line("- None")

    # Footer
    line("", dy=0.5 * cm)
    line("Disclaimer: This report is for research/educational use only.", dy=0.55 * cm, size=9)
    line("Do not use as a standalone basis for clinical decisions.", dy=0.55 * cm, size=9)

    c.save()
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes


# -----------------------------
# LOAD ASSETS (with cache fingerprint)
# -----------------------------
base_dir = Path(__file__).resolve().parent

paths_for_fp = [
    resolve_paths(base_dir, MODEL_A_NAME),
    resolve_paths(base_dir, MODEL_B_NAME),
    resolve_paths(base_dir, MODEL_C_NAME),
    resolve_paths(base_dir, META_NAME),
]
fingerprint = "|".join([f"{p.name}:{p.stat().st_mtime_ns if p.exists() else 'missing'}" for p in paths_for_fp])


@st.cache_resource
def load_assets(debug_flag=False, fingerprint: str = ""):
    base = Path(__file__).resolve().parent

    modelA_path = resolve_paths(base, MODEL_A_NAME)
    modelB_path = resolve_paths(base, MODEL_B_NAME)
    modelC_path = resolve_paths(base, MODEL_C_NAME)
    meta_path = resolve_paths(base, META_NAME)

    # Require models (meta optional)
    missing_models = [p for p in [modelA_path, modelB_path, modelC_path] if not p.exists()]
    if missing_models:
        raise FileNotFoundError(
            "Missing required model file(s):\n"
            + "\n".join([f"- {p.name} (expected at: {p})" for p in missing_models])
            + f"\n\nBase directory: {base}"
            + f"\nAlso checked: {base/'assets'}"
        )

    # Meta optional: load if present, else defaults
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {
            "name": "KeraRisk-CXL",
            "version": "1.0",
            "groups": [],
            "notes": "Research use only — not a diagnostic device.",
        }

    modelA, trustedA, extraA = safe_skops_load(modelA_path, debug=debug_flag)
    modelB, trustedB, extraB = safe_skops_load(modelB_path, debug=debug_flag)
    modelC, trustedC, extraC = safe_skops_load(modelC_path, debug=debug_flag)

    # Ensure meta["groups"]
    if "groups" not in meta or not meta["groups"]:
        try:
            ohe = modelC.named_steps["pre"].named_transformers_["cat"].named_steps["oh"]
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
    modelA, modelB, modelC, meta, used_paths, trusted_report = load_assets(debug_flag=debug, fingerprint=fingerprint)
except Exception as e:
    st.error("❌ Failed to load models/assets.")
    st.code(str(e))
    st.stop()

if debug:
    st.sidebar.write("Resolved paths:")
    st.sidebar.json(used_paths)
    st.sidebar.write("Trusted report:")
    st.sidebar.json(trusted_report)
    st.sidebar.write("Files seen in app root:")
    try:
        st.sidebar.code("\n".join(sorted([p.name for p in Path(used_paths["base"]).iterdir()])))
    except Exception as _:
        pass

# Model B fitted?
modelB_is_fitted = is_fitted(modelB)
if debug:
    st.sidebar.write("Model B fitted?")
    st.sidebar.code(str(modelB_is_fitted))

if not modelB_is_fitted:
    st.warning(
        "⚠️ Endpoint B (Kmax slope) is temporarily unavailable because Model B is not fitted in the deployed artifact. "
        "Endpoints A and C still work and you can export PDF/JSON."
    )


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

slope_B = None
if modelB_is_fitted:
    slope_B = float(modelB.predict(X)[0])


# -----------------------------
# OUTPUTS
# -----------------------------
st.subheader("📊 Model outputs")
c1, c2, c3 = st.columns(3)
c1.metric("Endpoint A (ΔKmax progression)", f"{risk_A*100:.1f}%")
c2.metric("Endpoint C (Composite risk)", f"{risk_C*100:.1f}%")
c3.metric("Endpoint B (Kmax slope)", "N/A" if slope_B is None else f"{slope_B:+.2f} D/year")

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
# SLOPE PROJECTIONS (up to 5 years or stability)
# -----------------------------
st.subheader("📈 Endpoint B: projections (up to 5 years or stability)")

st.sidebar.subheader("⚙️ Projection settings")
STABILITY_THRESHOLD = st.sidebar.number_input(
    "Stability threshold |slope| < (D/year)",
    min_value=0.05,
    max_value=1.00,
    value=0.25,
    step=0.05,
)
MAX_YEARS = 5

proj = None
stable_assumed = None

if slope_B is None:
    st.info("Endpoint B projections are unavailable because Model B is not fitted in the deployed artifact.")
else:
    rows = []
    stable_assumed = abs(slope_B) < float(STABILITY_THRESHOLD)

    for year in range(1, MAX_YEARS + 1):
        rows.append({"Year": year, "Projected ΔKmax (D)": year * slope_B, "Assumption": "Linear"})
        if stable_assumed:
            break

    proj = pd.DataFrame(rows)

    st.dataframe(
        proj.style.format({"Projected ΔKmax (D)": "{:+.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

    if stable_assumed:
        st.success(
            f"🟢 Operational stability assumed (|slope| < {float(STABILITY_THRESHOLD):.2f} D/year). "
            f"Projection truncated at year {int(proj['Year'].max())}."
        )
    else:
        st.info(
            f"Projection shown up to {MAX_YEARS} years (|slope| ≥ {float(STABILITY_THRESHOLD):.2f} D/year)."
        )

    st.caption(
        "Notes: Projections are linear extrapolations based on Endpoint B (slope). "
        "Stability is an operational definition for interpretability, not a mechanistic model."
    )


# -----------------------------
# EXPORT (JSON + PDF)
# -----------------------------
st.subheader("⬇️ Export (audit bundle)")

bundle = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "app": {"name": meta.get("name", "KeraRisk-CXL"), "purpose": "Research use only"},
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
        "slope_B": slope_B,  # None if unavailable
        "tier_C": t,
        "tier_thresholds": {"low_lt": 0.15, "intermediate_lt": 0.35, "high_ge": 0.35},
    },
    "plausibility_flags": flags,
}

if slope_B is not None and proj is not None and stable_assumed is not None:
    bundle["outputs"]["slope_projection"] = {
        "method": "linear",
        "stability_threshold_D_per_year": float(STABILITY_THRESHOLD),
        "max_years": int(MAX_YEARS),
        "stable_assumed": bool(stable_assumed),
        "values": [
            {"year": int(r["Year"]), "delta_kmax_D": float(r["Projected ΔKmax (D)"])}
            for _, r in proj.iterrows()
        ],
    }

st.download_button(
    label="Download JSON report",
    data=json.dumps(bundle, indent=2).encode("utf-8"),
    file_name="kerarisk_report.json",
    mime="application/json",
)

pdf_bytes = build_pdf_report(bundle)
st.download_button(
    label="Download PDF report",
    data=pdf_bytes,
    file_name="kerarisk_report.pdf",
    mime="application/pdf",
)

st.markdown("---")
st.caption("KeraRisk-CXL | Research use only. Validate locally before clinical use.")
