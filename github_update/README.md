# EndoSysScore (ESS) v3

**Phenotype-aware synthetic data-augmented risk prediction for 30-day mortality after cardiac surgery for infective endocarditis**

*European Heart Journal submission · Gelsomino S, Parise G, Parise O, Di Mauro M, Actis Dato G, Lorusso R*
*Maastricht University Medical Centre · CARIM · The Netherlands*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20327975.svg)](https://doi.org/10.5281/zenodo.20327975)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Calculator](https://img.shields.io/badge/Calculator-endosysscore.org-blue)](https://endosysscore.org)

---

## Overview

EndoSysScore (ESS) is a stacking ensemble clinical prediction model for 30-day mortality after cardiac surgery for infective endocarditis (IE). It was developed on the GIROC Italian multicenter registry (24 centres, 2010–2023; derivation cohort n=4,334) and evaluated in an independent external validation cohort of 881 patients from three centres.

The model introduces a **phenotype-aware augmentation framework**: six rare high-lethality phenotypes are identified in the derivation data, and synthetic patients (CTGAN + TIMA clinician-supervised filtering) are generated specifically for these underrepresented groups to correct the systematic mortality underestimation that conventional scores produce in these patients.

### Key performance (independent external validation, n=881)

| Metric | Value |
|--------|-------|
| AUC | 0.789 (95% CI 0.751–0.827) |
| DeLong superiority | **6/6** (all p<0.05 vs comparators) |
| O/E ratio | 0.994 |
| Brier score | 0.092 |
| Calibration slope | 0.845 |

### High-risk phenotype detection (mean absolute deviation vs observed mortality)

| Score | MAD (pp) |
|-------|----------|
| **EndoSysScore** | **3.8** |
| EndoSCORE | 9.4 |
| RISK-E | 15.8 |
| EuroSCORE II | 21.3 |
| AEPEI / APORTEI / STS-IE | 24–38 |

---

## Repository structure

```
sys-score/
├── phase57c_run.py              # Full training pipeline (ESS v3)
├── requirements.txt             # Python dependencies
├── CLINICAL_CONSTRAINTS_v5.md   # 31 TIMA clinical constraints
├── LICENSE                      # MIT
├── README.md
│
├── data/
│   ├── ZENODO_STUDY_development_4285.csv     # Derivation cohort (real patients)
│   ├── ZENODO_CONTROL_external_930.csv       # External validation cohort
│   └── ZENODO_DATABASE_pre_intra_post_5215.csv  # Full registry (descriptive)
│
├── models/
│   ├── Model_S_v3_WINNER.pkl        # Final trained model (LightGBM meta)
│   └── feature_names_v3_G.json      # Feature names and order
│
├── predictions/
│   └── control_predictions_WINNER.csv   # Model + comparator predictions on CONTROL
│
├── results/
│   ├── G_BEST_full_metrics.json     # Global performance metrics
│   ├── delong_WINNER.csv            # DeLong pairwise comparisons
│   ├── dca_WINNER.csv               # Decision curve analysis
│   ├── patterns_WINNER.csv          # Phenotype-level validation
│   └── architecture_comparison.csv  # M1/M2/M3 comparison
│
├── tima3/
│   ├── TIMA3_responses_raw.csv      # 1000 individual rater responses
│   ├── TIMA3_statistics_.csv        # Per-rater and pooled statistics
│   └── EHJ_TIMA_COMPLETED_WITH_10_BLINDED_CLINICIANS.xlsx
│
└── calculator/
    └── index.html                   # Standalone web calculator (endosysscore.org)
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.10. GPU recommended for training (NVIDIA CUDA).

### 2. Run training pipeline

```bash
python phase57c_run.py
```

Reproduces the full ESS v3 pipeline:
- Loads derivation + synthetic data
- Assigns phenotype features (A–F)
- Trains base learners (XGBoost shallow, XGBoost moderate, logistic regression)
- Trains LightGBM meta-learner with G1 weighting
- Applies isotonic + beta calibration
- Evaluates on external validation cohort
- Saves model, predictions, and metrics

### 3. Use the calculator

Open `calculator/index.html` in any browser — runs entirely client-side, no data transmitted.
Or use the live version: **[endosysscore.org](https://endosysscore.org)**

---

## Model architecture

### Base learners
- XGBoost shallow (max_depth=3, n_estimators=100)
- XGBoost moderate (max_depth=4, n_estimators=150)
- Logistic regression (L1 regularisation)

### Meta-learner input (15 features)
- 3 base learner OOF predictions
- 7 phenotype features (in_pattern_A–F + n_patterns_satisfied)
- 5 clinical features (SHOCK, IABP_PRE, ASCESSO, saureus_b, ENDOATTIVA)

### Meta-learner
LightGBM (max_depth=4) with G1 sample weighting:
- Real patients: weight = 1.0
- Synthetic in-phenotype: weight = 1.0
- Synthetic out-of-phenotype: weight = 0.01

### Calibration
Primary isotonic calibration → beta calibration (Phase 5.4, applied on real development data only)

---

## High-risk phenotypes (A–F)

| Phenotype | Definition | Risk tier | Observed mortality (validation) |
|-----------|-----------|-----------|--------------------------------|
| A | Multivalve + *S. aureus* + CKD≥3/dialysis/PAPs>50 | Very high | 54.5% (n=11) |
| B | NVE + age≥75 + culture-negative | Very high | 28.6% (n=21) |
| C | Heart failure + COPD + age≥65 | Very high | 36.8% (n=19) |
| D | PVE + *S. aureus* + aortic + mitral | High | 40.0% (n=15) |
| E | NVE + age≥80 + LVEF<50% | Very high | 54.5% (n=11) |
| F | NVE + isolated mitral + female + age≥65 | High | 20.0% (n=35) |

Conventional scores underestimate mortality in these phenotypes by 10–38 percentage points. ESS mean absolute deviation: 3.8 pp.

---

## TIMA-3 synthetic data realism test

10 blinded cardiac surgeons classified 1,000 mixed real/synthetic profiles (100 each):

| Metric | Value |
|--------|-------|
| Pooled accuracy | 49.5% (95% CI 46.4–52.6%) |
| Cohen κ | −0.01 |
| Binomial p vs 50% chance | 0.776 |
| AUROC | 0.494 |

Synthetic patients are statistically indistinguishable from real ones by expert clinicians.

---

## Validation cohort

The independent external validation cohort comprised 881 patients from three centres:
- Centre 10 (n=339): entirely absent from derivation data
- Centre 15 (n=320): contributed 11 patients to derivation, 320 to validation (no individual patient overlap)
- Centre 23 (n=222): entirely absent from derivation data

No validation-cohort patient was used for model training, synthetic augmentation, calibration, or phenotype optimisation. During phenotype definition, candidate phenotypes were required to demonstrate non-zero representation in the validation cohort to confirm applicability; phenotype definitions were fixed before any outcome data from the validation cohort were examined.

---

## Comparator scores

Six established IE-specific surgical risk scores applied with published coefficients without recalibration:
EuroSCORE II (Nashef 2012), EndoSCORE (Di Mauro 2017), RISK-E (Olmos 2017), AEPEI (Gatti 2016), APORTEI (Varela 2020), STS-IE (Gaca 2011).

**Conflict-of-interest disclosure:** M. Di Mauro, G. Actis Dato, and S. Gelsomino are co-authors of EndoSCORE. EndoSCORE is used here only as an external comparator applied with its originally published coefficients.

---

## Citation

```
Gelsomino S, Parise G, Parise O, Di Mauro M, Actis Dato G, Lorusso R.
EndoSysScore (ESS): Phenotype-Aware Synthetic Data-Augmented Risk Prediction
for 30-Day Mortality after Cardiac Surgery for Infective Endocarditis.
European Heart Journal, 2026. doi:10.5281/zenodo.20327975
```

---

## License

MIT License — see [LICENSE](LICENSE) for full text.

**Clinical disclaimer:** This software is a research tool and must not be the sole basis for clinical decisions. It has not been approved by any regulatory agency. Users are responsible for local clinical validation before any use beyond research.

---

*© 2026 Sandro Gelsomino, Maastricht University Medical Centre. Software engineering with AI-assisted development.*
