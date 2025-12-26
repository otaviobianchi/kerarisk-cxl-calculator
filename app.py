# app.py
# KeraRisk-CXL Calculator (Streamlit)
# - Robust skops loader (CVE-2024-37065 compatible)
# - Meta optional (won't crash if missing / deploy stale)
# - Safe handling for Model B not fitted (won't crash)
# - Cache fingerprint to reduce Streamlit Cloud stale artifacts
# - Exports JSON + PDF report
# - Endpoint B projections: Auto(validated) / Linear / Damped exponential + optional uncertainty band (λ CI)
# - Adds continuous Auto(validated) curve (0–5y) with optional uncertainty band + export to JSON

import json
import re
import ast
from pathlib import Path
from datetime import datetime, timezone
from io import BytesIO

import numpy as np
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

    line("KeraRisk-CXL Calculator — Report", dy=0.75 * cm, font="Helvetica-Bold", size=14)
    line("Research use only — not a diagnostic device.", dy=0.75 * cm, font="Helvetica-Oblique", size=10)
    line(f"Generated (UTC): {bundle.get('timestamp_utc','')}", dy=0.8 * cm, size=10)

    line("Inputs", dy=0.65 * cm, font="Helvetica-Bold", size=12)
    inp = bundle.get("inputs", {})
    for k in ["age", "kmax0", "pachy0", "bcva0", "cyl0", "group"]:
        if k in inp:
            line(f"- {k}: {inp[k]}")

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

    sp = out.get("slope_projection", None)
    if isinstance(sp, dict) and sp:
        line("", dy=0.35 * cm)
        line("Endpoint B — projections", dy=0.65 * cm, font="Helvetica-Bold", size=12)
        line(f"- Method: {sp.get('method','')}")
        lam = sp.get("lambda_central", None)
        if lam is not None:
            line(f"- λ central (1/year): {float(lam):.2f}")
        ci = sp.get("lambda_ci95", None)
        if isinstance(ci, list) and len(ci) == 2:
            line(f"- λ 95% CI (1/year): [{float(ci[0]):.2f}, {float(ci[1]):.2f}]")
        line(f"- Max years: {sp.get('max_years','')}")

        vals = sp.get("values", [])
        if isinstance(vals, list) and vals:
            for v in vals:
                try:
                    yy = int(v.get("year"))
                    dk = float(v.get("delta_kmax_D"))
                    line(f"  • Year {yy}: ΔKmax {dk:+.2f} D")
                except Exception:
                    continue

    # Auto curve summary (optional)
    ac = out.get("auto_validated_curve", None)
    if isinstance(ac, dict) and ac.get("years_grid"):
        line("", dy=0.35 * cm)
        line("Auto (validated) curve", dy=0.65 * cm, font="Helvetica-Bold", size=12)
        line("Definition: linear 0–1y; then ΔK(t)=ΔK(1)+(slope/λ)(1-exp(-λ(t-1))) for t>1", dy=0.55 * cm, size=10)
        line(f"λ central: {ac.get('lambda_central')}", dy=0.55 * cm, size=10)
        ci = ac.get("lambda_ci95", None)
        if isinstance(ci, list) and len(ci) == 2:
            line(f"λ 95% CI: [{ci[0]}, {ci[1]}]", dy=0.55 * cm, size=10)

    flags = bundle.get("plausibility_flags", [])
    line("", dy=0.35 * cm)
    line("Plausibility flags", dy=0.65 * cm, font="Helvetica-Bold", size=12)
    if flags:
        for f in flags:
            line(f"- {f}")
    else:
        line("- None")

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

    # Meta optional (use defaults if not found)
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

    # Ensure projection meta defaults exist (so app behavior is reproducible)
    proj = meta.get("projection", {})
    meta["projection"] = {
        "default_method": proj.get("default_method", "damped_exponential"),
        "lambda": float(proj.get("lambda", 0.6)),
        "lambda_ci95": proj.get("lambda_ci95", [0.4, 0.9]),
        "validated_horizons_years": proj.get("validated_horizons_years", [1, 2, 3, 5]),
        "notes": proj.get(
            "notes",
            "Damped exponential reduces long-term divergence; validate on your cohort for MAE/RMSE at 2–5y.",
        ),
    }

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
    modelA, modelB, modelC, meta, used_paths, trusted_report = load_assets(
        debug_flag=debug, fingerprint=fingerprint
    )
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
    except Exception:
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

X = pd.DataFrame(
    [
        {
            "age": float(age),
            "kmax0": float(kmax0),
            "pachy0": float(pachy0),
            "bcva0": float(bcva0),
            "cyl0": float(cyl0),
            "group": str(group),
        }
    ]
)


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
# Endpoint B: Projections (Auto/Linear/Damped + uncertainty band)
# -----------------------------
st.subheader("📈 Endpoint B: projections (up to 5 years)")

proj = None
proj_method = None
lambda_central = None
lambda_ci95 = None
show_uncertainty = None
MAX_YEARS = 5

# For exporting Auto curve
t_grid = None
y_c = None
y_low = None
y_high = None
auto_curve_df = None
auto_curve_available = False

if slope_B is None:
    st.info("Endpoint B projections are unavailable because Model B is not fitted in the deployed artifact.")
else:
    proj_meta = meta.get("projection", {})
    default_method = proj_meta.get("default_method", "damped_exponential")

    lambda_central = float(proj_meta.get("lambda", 0.6))
    ci = proj_meta.get("lambda_ci95", [0.4, 0.9])
    if isinstance(ci, list) and len(ci) == 2:
        lambda_ci95 = [float(ci[0]), float(ci[1])]
    else:
        lambda_ci95 = [0.4, 0.9]

    # For damped exponential: higher λ -> faster stabilization -> smaller ΔK
    lambda_low = float(min(lambda_ci95))
    lambda_high = float(max(lambda_ci95))

    # Sidebar controls
    st.sidebar.subheader("⚙️ Projection settings")

    options = ["Auto (validated)", "Linear (baseline)", "Damped exponential (recommended)"]
    default_index = 0
    if default_method == "linear":
        default_index = 1
    elif default_method in ("damped_exponential", "damped"):
        default_index = 2

    proj_method = st.sidebar.selectbox(
        "Projection model",
        options,
        index=default_index if default_index < len(options) else 0,
        help="Auto uses linear at 1y and damped exponential for ≥2y to reduce long-term divergence.",
    )

    show_uncertainty = st.sidebar.checkbox(
        "Show uncertainty band (λ 95% CI)",
        value=True,
        help="Uses λ CI from kerarisk_meta.json (or defaults if missing).",
    )

    # Allow user to override λ if they want
    with st.sidebar.expander("Advanced: override λ", expanded=False):
        lambda_central = st.number_input(
            "λ central (1/year)",
            min_value=0.1,
            max_value=2.0,
            value=float(lambda_central),
            step=0.05,
        )
        l1, l2 = lambda_ci95
        l1 = st.number_input("λ low (1/year)", min_value=0.05, max_value=2.0, value=float(l1), step=0.05)
        l2 = st.number_input("λ high (1/year)", min_value=0.05, max_value=2.0, value=float(l2), step=0.05)
        lambda_ci95 = [float(min(l1, l2)), float(max(l1, l2))]
        lambda_low, lambda_high = lambda_ci95[0], lambda_ci95[1]

    def damped(delta_slope: float, lam: float, t_years: float) -> float:
        return (delta_slope / lam) * (1.0 - np.exp(-lam * t_years))

    rows = []
    for year in range(1, MAX_YEARS + 1):
        # method per year if Auto
        if proj_method == "Auto (validated)":
            method_used = "Linear (baseline)" if year == 1 else "Damped exponential (recommended)"
        else:
            method_used = proj_method

        if method_used == "Linear (baseline)":
            delta = slope_B * year
            delta_lo = None
            delta_hi = None
        else:
            delta = damped(slope_B, float(lambda_central), year)
            if show_uncertainty:
                # higher λ => smaller ΔK (faster stabilization)
                delta_lo = damped(slope_B, float(lambda_high), year)
                delta_hi = damped(slope_B, float(lambda_low), year)
            else:
                delta_lo = None
                delta_hi = None

        rows.append(
            {
                "Year": year,
                "Projected ΔKmax (D)": delta,
                "Model (used)": method_used,
                "ΔK low (λ_high)": delta_lo,
                "ΔK high (λ_low)": delta_hi,
            }
        )

    proj = pd.DataFrame(rows)

    if show_uncertainty:
        st.dataframe(
            proj.style.format(
                {
                    "Projected ΔKmax (D)": "{:+.2f}",
                    "ΔK low (λ_high)": "{:+.2f}",
                    "ΔK high (λ_low)": "{:+.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Auto (validated) uses linear at 1y and damped exponential for ≥2y. "
            "Uncertainty band uses λ 95% CI (higher λ → smaller ΔK). "
            "Interpret as an aid, not a deterministic forecast."
        )
    else:
        proj_view = proj[["Year", "Projected ΔKmax (D)", "Model (used)"]]
        st.dataframe(
            proj_view.style.format({"Projected ΔKmax (D)": "{:+.2f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Auto (validated) uses linear at 1y and damped exponential for ≥2y to avoid long-term divergence. "
            "Damped exponential assumes stabilization; interpret as an aid, not a deterministic forecast."
        )

    with st.expander("Why these projection models?"):
        st.markdown(
            """
- **Linear** extrapolates the predicted slope indefinitely: ΔK(t)=slope·t.  
- **Damped exponential** assumes progressive stabilization: ΔK(t)=(slope/λ)·(1−e^(−λt)).  
- **Auto (validated)** uses linear at 1 year and damped exponential for t>1 year, enforcing continuity at 1y to reduce long-term overestimation.

For rigorous selection, calibrate **λ** and compare **MAE/RMSE** at 2–5y using your cohort.
            """.strip()
        )

    # -----------------------------
    # Curve for Auto (validated): continuous ΔK(t) from 0..5y
    # Linear 0..1y, then damped-exponential for t>1 with continuity at t=1
    # -----------------------------
    st.subheader("📉 Auto (validated) curve (continuous, 0–5 years)")

    slope = float(slope_B)
    lam_c = float(lambda_central)
    lam_lo = float(lambda_low)
    lam_hi = float(lambda_high)

    t_grid = np.linspace(0.0, float(MAX_YEARS), 251)

    def auto_curve(t, lam):
        t = float(t)
        if t <= 1.0:
            return slope * t
        # ΔK(1)=slope*1, then damped for the remaining time (t-1)
        return slope * 1.0 + damped(slope, lam, (t - 1.0))

    y_c = np.array([auto_curve(ti, lam_c) for ti in t_grid])

    auto_curve_df = pd.DataFrame(
        {
            "Years": t_grid,
            "ΔKmax (Auto, central)": y_c,
        }
    )

    if show_uncertainty:
        # higher λ => faster stabilization => smaller ΔK
        y_low = np.array([auto_curve(ti, lam_hi) for ti in t_grid])   # conservative low ΔK
        y_high = np.array([auto_curve(ti, lam_lo) for ti in t_grid])  # higher ΔK
        auto_curve_df["ΔKmax (Auto, low band)"] = y_low
        auto_curve_df["ΔKmax (Auto, high band)"] = y_high
        st.caption(
            "Auto curve = linear (0–1y) + damped exponential (>1y) with continuity at 1y. "
            "Band uses λ 95% CI (higher λ → smaller ΔK)."
        )
    else:
        st.caption(
            "Auto curve = linear (0–1y) + damped exponential (>1y) with continuity at 1y."
        )

    st.line_chart(auto_curve_df.set_index("Years"), use_container_width=True)

    with st.expander("Show curve data (Auto validated)"):
        st.dataframe(auto_curve_df, use_container_width=True, hide_index=True)

    auto_curve_available = True


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
    "meta_notes": meta.get("notes", "Research use only — not a diagnostic device."),
}

if slope_B is not None and proj is not None:
    values = []
    for _, r in proj.iterrows():
        values.append(
            {
                "year": int(r["Year"]),
                "delta_kmax_D": float(r["Projected ΔKmax (D)"]),
                "model_used": str(r["Model (used)"]),
                "delta_kmax_low_lambda_high": None
                if pd.isna(r["ΔK low (λ_high)"])
                else float(r["ΔK low (λ_high)"]),
                "delta_kmax_high_lambda_low": None
                if pd.isna(r["ΔK high (λ_low)"])
                else float(r["ΔK high (λ_low)"]),
            }
        )

    bundle["outputs"]["slope_projection"] = {
        "method": proj_method,
        "lambda_central": float(lambda_central) if lambda_central is not None else None,
        "lambda_ci95": [float(lambda_ci95[0]), float(lambda_ci95[1])] if lambda_ci95 is not None else None,
        "show_uncertainty_band": bool(show_uncertainty) if show_uncertainty is not None else None,
        "max_years": int(MAX_YEARS),
        "values": values,
    }

# Export Auto curve data (if available)
if auto_curve_available and t_grid is not None and y_c is not None:
    bundle["outputs"]["auto_validated_curve"] = {
        "years_grid": [float(x) for x in t_grid],
        "delta_kmax_central": [float(x) for x in y_c],
        "lambda_central": float(lambda_central) if lambda_central is not None else None,
        "lambda_ci95": [float(lambda_ci95[0]), float(lambda_ci95[1])] if lambda_ci95 is not None else None,
        "definition": "linear 0–1y; then ΔK(t)=ΔK(1)+ (slope/λ)(1-exp(-λ(t-1))) for t>1",
    }
    if show_uncertainty and y_low is not None and y_high is not None:
        bundle["outputs"]["auto_validated_curve"]["delta_kmax_low_band"] = [float(x) for x in y_low]
        bundle["outputs"]["auto_validated_curve"]["delta_kmax_high_band"] = [float(x) for x in y_high]

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
