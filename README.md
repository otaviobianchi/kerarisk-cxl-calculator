🩺 KeraRisk-CXL

Clinical risk stratification and progression modeling in keratoconus

📌 Overview

KeraRisk-CXL is a data-driven clinical decision-support tool designed to estimate the risk of keratoconus progression using baseline tomographic and functional parameters.

The platform integrates three complementary predictive endpoints, derived from longitudinal clinical data and validated using out-of-fold cross-validation:

Endpoint A — Structural progression (ΔKmax)

Endpoint B — Annual Kmax slope (D/year)

Endpoint C — Composite clinical progression (structural + functional)

The tool is implemented as an online calculator for ophthalmologists and researchers, ensuring interpretability, reproducibility, and clinical relevance.

⚠️ Research use only. This tool is not intended as a standalone diagnostic device.

🧠 Scientific Rationale

Keratoconus progression is heterogeneous and multifactorial. Traditional single-threshold definitions (e.g., ΔKmax ≥ 1.0 D) may fail to capture:

Noisy longitudinal measurements

Functional deterioration without marked tomographic change

Variable progression rates over time

To address these limitations, KeraRisk-CXL models progression using:

Binary structural change (Endpoint A)

Continuous progression velocity (Endpoint B)

Composite clinically meaningful deterioration (Endpoint C)

This multi-endpoint strategy reflects real-world clinical decision-making, particularly in the context of CXL indication and follow-up planning.

📊 Endpoints Definition
🔹 Endpoint A — Structural Progression

Binary classification

Progression defined as:
max(Kmax ≤ 5 years) − Kmax_baseline ≥ 1.5 D

Captures peak structural worsening

Robust to visit timing and noise

Aligned with common tomographic criteria in literature

🔹 Endpoint B — Annual Kmax Slope

Regression (D/year)

Estimated using robust Theil–Sen regression on longitudinal Kmax

Quantifies rate of progression

Less sensitive to outliers

Clinically useful for individualized follow-up intervals

🔹 Endpoint C — Composite Clinical Endpoint

Binary classification

Progression defined as any of the following within 5 years:

ΔKmax ≥ 1.5 D

BCVA decrease ≥ 0.2 (decimal)

Cylinder increase ≥ 1.0 D

This endpoint reflects clinically actionable deterioration, even when tomographic progression alone is equivocal.

🧮 Model Architecture

All models share a unified preprocessing pipeline:

Inputs (baseline):

Age (years)

Kmax (D)

Minimum pachymetry (µm)

BCVA (decimal)

Cylinder (D)

Treatment group (categorical)

Preprocessing:

Median imputation (numeric)

Standard scaling

One-hot encoding (group)

Models:

Endpoint A: Logistic Regression (class-weighted)

Endpoint B: ElasticNet regression

Endpoint C: Logistic Regression (class-weighted)

Models are trained using cross-validation, and OOF predictions are used for all reported metrics and figures.

📈 Model Performance (Out-of-Fold)
Model	Endpoint	Metric
A	ΔKmax progression	AUC ≈ 0.71
B	Kmax slope	MAE ≈ 2.3 D/year
C	Composite endpoint	AUC ≈ 0.70

Calibration, ROC curves, and observed vs. predicted analyses are included in the manuscript and generated automatically by the pipeline.

🌐 Online Calculator (Streamlit)

The calculator allows clinicians to:

Enter baseline patient data

Instantly obtain:

Probability of structural progression

Predicted Kmax slope

Composite clinical risk

Receive risk stratification:

Low (<15%)

Intermediate (15–35%)

High (>35%)

The app is implemented using Streamlit and loads models stored safely as .skops files.

📁 Repository Structure
kerarisk-cxl-calculator/
│
├── app.py                     # Streamlit application
├── README.md                  # This file
├── requirements.txt           # Dependencies
│
├── assets/
│   ├── KeraRisk_modelA.skops  # Endpoint A model
│   ├── KeraRisk_modelB.skops  # Endpoint B model
│   ├── KeraRisk_modelC.skops  # Endpoint C model
│   └── kerarisk_meta.json     # Metadata (groups, thresholds)
│
├── figures/                   # Figures for manuscript
├── tables/                    # Tables for manuscript
└── notebooks/                 # Model training & validation

🚀 How to Run Locally
pip install -r requirements.txt
streamlit run app.py

🔐 Model Security & Reproducibility

Models are stored as .skops, not pickle/joblib

Safe loading with explicit trusted classes

Fully reproducible preprocessing and inference

🩺 Clinical Use — Practical Guidance

High composite risk: consider closer monitoring or early intervention

High slope but low ΔKmax: progression may be imminent

Functional deterioration alone: still flagged by Endpoint C

The tool is designed to support, not replace, clinical judgment.

⚠️ Limitations

Single-center dataset

Modest sample size

External validation pending

Does not replace biomechanical or epithelial mapping

These aspects are discussed in detail in the manuscript.

📚 Citation (suggested)


KeraRisk-CXL: A multi-endpoint machine learning framework for predicting keratoconus progression
Journal under review

📬 Contact

For questions, collaboration, or validation studies:

Oss
