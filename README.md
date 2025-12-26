# 🩺 KeraRisk-CXL (Academic Research Tool)

**KeraRisk-CXL** is an **academic, data-driven research tool** for **risk stratification and progression modeling in keratoconus**, based on baseline tomographic and functional parameters.

> ⚠️ **Research / educational use only.**  
> This project is **not** a medical device, **not** a diagnostic system, and **must not** be used as a standalone basis for clinical decisions.  
> The objective is **scientific reproducibility, methodology transparency, and academic validation**.

---

## 📌 Overview

KeraRisk-CXL estimates keratoconus progression using **three complementary endpoints** derived from longitudinal clinical data and evaluated with **out-of-fold cross-validation**:

- **Endpoint A — Structural progression (ΔKmax)** *(classification)*
- **Endpoint B — Annual Kmax slope (D/year)** *(regression)*
- **Endpoint C — Composite clinical progression (structural + functional)** *(classification)*

The repository provides an **online-style calculator** (Streamlit) that enables **reproducible inference** from trained models and generates **exportable reports (JSON + PDF)** for academic auditing and documentation.

---

## 🧠 Scientific Rationale

Keratoconus progression is heterogeneous and multifactorial. Single-threshold criteria (e.g., ΔKmax ≥ 1.0 D) may fail to capture:

- measurement noise and variable follow-up schedules  
- functional deterioration without major tomographic change  
- heterogeneous rates of progression (fast vs. slow progressors)

To address these limitations, KeraRisk-CXL adopts a **multi-endpoint strategy**:

- **Endpoint A** captures **peak structural worsening**
- **Endpoint B** captures **velocity of progression**
- **Endpoint C** captures **clinically meaningful deterioration** even when structural change alone is equivocal

This design is intended to support **academic studies on progression definitions**, model comparison, calibration, and external validation workflows.

---

## 📊 Endpoints Definition

### 🔹 Endpoint A — Structural Progression (ΔKmax)
- **Type:** Binary classification  
- **Definition:**  
  \[
  \max(Kmax \le 5\ \text{years}) - Kmax_{baseline} \ge 1.5\ D
  \]
- Captures peak structural worsening  
- More robust to visit timing variation and noise  
- Aligned with common tomographic criteria used in the literature

---

### 🔹 Endpoint B — Annual Kmax Slope
- **Type:** Regression (D/year)  
- Estimated using robust longitudinal fitting (e.g., **Theil–Sen regression**) on Kmax time series  
- Quantifies progression rate (velocity), useful for academic analyses of individualized follow-up strategies  
- Less sensitive to outliers than simple least-squares slope

---

### 🔹 Endpoint C — Composite Clinical Progression
- **Type:** Binary classification  
- **Definition:** progression is flagged if **any** occur within 5 years:
  - ΔKmax ≥ 1.5 D  
  - BCVA decrease ≥ 0.2 (decimal)  
  - Cylinder increase ≥ 1.0 D  
- Captures clinically meaningful deterioration beyond tomographic changes alone  
- Designed for academically exploring “actionable deterioration” definitions

---

## 🧮 Model Architecture

### Inputs (baseline)
- Age (years)  
- Kmax (D)  
- Minimum pachymetry (µm)  
- BCVA (decimal)  
- Cylinder (D)  
- Treatment group (categorical)

### Preprocessing (shared)
- Median imputation (numeric)  
- Standard scaling  
- One-hot encoding (group)

### Models
- Endpoint A: Logistic Regression (class-weighted)  
- Endpoint B: ElasticNet regression  
- Endpoint C: Logistic Regression (class-weighted)

Models are trained under cross-validation, and **OOF predictions** are used for reported metrics, calibration analyses, and figures.

---

## 📈 Performance (Out-of-Fold) *(example summary)*
> Replace with your final manuscript numbers if updated.

| Endpoint | Task | Metric (OOF) |
|---|---|---|
| A | ΔKmax progression | AUC ≈ 0.71 |
| B | Kmax slope | MAE ≈ 2.3 D/year |
| C | Composite endpoint | AUC ≈ 0.70 |

Calibration curves, ROC curves, and observed-vs-predicted analyses are produced by the training/validation pipeline.

---

## 🌐 Online Calculator (Streamlit)

The app allows you to:
- input baseline patient variables  
- compute Endpoint A/C probabilities and Endpoint B slope  
- export **JSON + PDF** reports for academic documentation

### Risk tiers (Endpoint C)
- **Low:** < 15%  
- **Intermediate:** 15–35%  
- **High:** ≥ 35%

> Note: Tier thresholds are an **interpretability layer** and must be validated for any new cohort.

---

## 📈 Endpoint B Projections (Academic Aid)

The app can show projections up to **5 years** using:

### 1) Linear (baseline)
\[
\Delta K(t)=\text{slope}\cdot t
\]

### 2) Damped exponential (recommended for long horizons)
\[
\Delta K(t)=\frac{\text{slope}}{\lambda}\left(1-e^{-\lambda t}\right)
\]

### 3) Auto (validated): piecewise + continuous
Auto uses:
- **Linear** for 0–1 year  
- **Damped exponential** for >1 year **with continuity at t=1**:

\[
\Delta K(t)=
\begin{cases}
\text{slope}\cdot t, & 0 \le t \le 1 \\
\Delta K(1) + \frac{\text{slope}}{\lambda}\left(1-e^{-\lambda (t-1)}\right), & t>1
\end{cases}
\]

The app also plots a **continuous Auto curve (0–5y)** and can display an uncertainty band using a **λ 95% CI** (higher λ → smaller long-term ΔK).

> ⚠️ These projections are intended for **academic interpretability and hypothesis generation**, not for clinical forecasting.

---

## ✅ How to Validate that a Non-Linear Projection is “Best” (Recommended Academic Workflow)

A non-linear projection is only “better” if it improves **out-of-sample** accuracy on your cohort.

Recommended approach (offline):
1. Use K-fold CV on your cohort  
2. Predict slopes on validation folds  
3. For candidate λ values (grid search), compute ΔK at 2–5 years  
4. Compare **MAE/RMSE** at each horizon  
5. Select λ minimizing error and quantify uncertainty (bootstrap / CV distribution)

---

## 📁 Repository Structure

kerarisk-cxl-calculator/
├── app.py
├── README.md
├── requirements.txt
├── KeraRisk_modelA.skops
├── KeraRisk_modelB.skops
├── KeraRisk_modelC.skops
├── kerarisk_meta.json # optional (groups, projection defaults, notes)
├── assets/ # optional alternative location
├── figures/ # manuscript figures
├── tables/ # manuscript tables
└── notebooks/ # training/validation notebooks

yaml
Copiar código

> The app searches models/metadata in the repo root first, and then in `assets/`.

---

## 🚀 How to Run Locally (Free)

### 1) Create a virtual environment (recommended)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
2) Install dependencies
bash
Copiar código
pip install -r requirements.txt
3) Run the app
bash
Copiar código
streamlit run app.py
🔐 Model Security & Reproducibility
Models are stored as .skops (not pickle/joblib)

Safe loading with explicit trusted classes (CVE-aware)

Reproducible preprocessing + inference pipeline

⚠️ Limitations (Academic Transparency)
Single-center dataset

Modest sample size

External validation pending

Does not include biomechanical or epithelial mapping variables by default

All limitations should be discussed in the manuscript and considered when interpreting results.

📚 Citation (suggested)
KeraRisk-CXL: A multi-endpoint machine learning framework for predicting keratoconus progression. Journal under review.

📬 Contact
For questions, collaboration, academic validation studies, or method discussions:

Halina Sitnik
Email: (add institutional email here)
