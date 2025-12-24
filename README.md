# KeraRisk-CXL  
### A Web-Based Clinical Risk Calculator for Keratoconus Progression

KeraRisk-CXL is an open-access clinical decision-support tool designed to estimate the risk of keratoconus progression after corneal cross-linking (CXL).  
The calculator is based on longitudinal clinical data and interpretable statistical and machine learning models.

---

## 🔬 Scientific Background

Keratoconus progression after CXL remains heterogeneous and difficult to predict at the individual level.  
This project implements and validates three complementary predictive endpoints:

- **Endpoint A (Binary progression)**  
  ΔKmax ≥ 1.5 D within 5 years

- **Endpoint B (Continuous progression rate)**  
  Annual Kmax slope (D/year)

- **Endpoint C (Composite clinical endpoint)**  
  ΔKmax ≥ 1.5 D **OR** BCVA loss ≥ 0.2 **OR** Cylinder increase ≥ 1.0 D within 5 years

The models were trained and evaluated using patient-level baseline data and longitudinal follow-up, with out-of-fold (OOF) validation.

---

## 🧠 Models Implemented

| Model | Type | Output |
|------|------|--------|
| Model A | Logistic regression | Probability of Kmax progression |
| Model B | ElasticNet regression | Annual Kmax slope (D/year) |
| Model C | Logistic regression | Composite clinical risk |

Key features:
- Age
- Baseline Kmax
- Minimum pachymetry
- BCVA
- Refractive cylinder
- Treatment group (FRAK vs FRAKcross)

Class imbalance is handled using class-weighted models and leave-one-out or stratified cross-validation, depending on event prevalence.

---

## 🩺 Clinical Calculator (Web App)

The calculator allows clinicians to:

1. Input baseline patient data  
2. Estimate:
   - 5-year progression risk (Endpoint C)
   - Expected Kmax change or slope
3. Classify patients into **Low / Intermediate / High risk** categories

⚠️ The calculator is intended as **decision support**, not as a standalone diagnostic tool.

---

## 🌐 Online Application

The calculator is deployed using **Streamlit Cloud** and hosted directly from this GitHub repository.

👉 **Live App:**  
`https://kerarisk-cxl.streamlit.app`  
*kerarisk-cxl-calculator-eqoukesapp2t3toahqtuv9r.streamlit.app*

---

## 📁 Repository Structure

