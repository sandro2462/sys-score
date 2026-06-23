# EndoSysScore (ESS) v3

**Phenotype-aware, synthetic-data-augmented risk prediction for 30-day mortality after cardiac surgery for infective endocarditis**

European Heart Journal submission · on behalf of the Italian Group of Research for Outcome in Cardiac Surgery (GIROC)
Maastricht University Medical Centre · CARIM · The Netherlands

License: MIT · DOI (all versions): [10.5281/zenodo.20327974](https://doi.org/10.5281/zenodo.20327974) · Calculator: https://endosysscore.org

---

## Overview

EndoSysScore (ESS) is a clinical prediction model for 30-day mortality after cardiac surgery for infective endocarditis (IE), developed on the multicentre Italian GIROC registry.

It introduces a **phenotype-aware augmentation framework**: six rare, high-lethality pre-operative phenotypes that conventional scores systematically underestimate are identified in the development data, and synthetic patients (CTGAN generation under the clinician-supervised TIMA method) are added **only** for these under-represented phenotypes, **only** from development data, solely to rebalance phenotype frequency during training and correct that underestimation.

- **Development cohort** (real patients): **4,285**
- **Independent external validation cohort** (real patients only): **930** (109 deaths, 11.7%)
- The validation cohort contributed to **neither** phenotype definition, synthetic generation, training, **nor** calibration. Phenotype definitions were fixed a priori on clinical criteria, independent of any model output.

## Key performance (external validation, n = 930)

| Metric | Value |
|---|---|
| AUC | 0.78 (95% CI 0.74–0.83) |
| DeLong superiority | 6/6 comparators (all p < 0.05) |
| O/E ratio | 1.05 |
| Brier score | 0.09 |
| Calibration slope | 0.88 |
| Calibration intercept | −0.14 |
| Hosmer–Lemeshow | p = 0.26 |
| Phenotype mean absolute deviation (MAD) | 3.4 pp |

Comparator discrimination on the same cohort (AUC): EuroSCORE II 0.744 · EndoSCORE 0.749 · RISK-E 0.722 · AEPEI 0.724 · APORTEI 0.721 · STS-IE 0.695. EndoSysScore was significantly higher than each by DeLong (p from 0.0003 to 0.030).

## The six rare high-lethality phenotypes

| Pattern | Pre-operative definition | n | Events | Observed 30-day mortality |
|---|---|---|---|---|
| A | Multivalve involvement + *S. aureus* + renal impairment (CKD stage ≥3 or dialysis) **or** PAPS > 50 mmHg | 11 | 6 | 54.5% (6/11) |
| B | Age ≥80 + native valve endocarditis + COPD **or** LVEF < 50% | 15 | 8 | 53.3% (8/15) |
| C | Periannular abscess or periannular extension + isolated mitral involvement + age ≥65 | 10 | 5 | 50.0% (5/10) |
| D | Prosthetic valve endocarditis + *S. aureus* + combined mitral and aortic involvement | 10 | 5 | 50.0% (5/10) |
| E | Native valve endocarditis + age ≥80 + LVEF < 50% | 11 | 6 | 54.5% (6/11) |
| F | Periannular abscess + isolated mitral involvement + age ≥65 | 10 | 5 | 50.0% (5/10) |

Each phenotype accounts for < 1% of the surgical population. Patterns E and F are **pre-specified refinements** of B and C (E restricts B to impaired LV function; F restricts C to periannular abscess), retained to confirm that the extreme risk persists in the more narrowly defined subgroup.

## Observed vs predicted 30-day mortality by phenotype

| Pattern | Observed | EndoSysScore (gap) | RISK-E (gap) | Other 5 scores, range (gap) |
|---|---|---|---|---|
| A | 54.5% | 46.2% (−8.3) | 31.6% (−22.9) | 7–42% (−12 to −47) |
| B | 53.3% | 46.3% (−7.1) | 21.9% (−31.4) | 7–31% (−22 to −46) |
| C | 50.0% | 50.0% (0.0) | 45.2% (−4.8) | 10–46% (−4 to −40) |
| D | 50.0% | 46.5% (−3.5) | 50.0% (0.0) | 12–41% (−9 to −38) |
| E | 54.5% | 52.8% (−1.7) | 24.0% (−30.5) | 7–33% (−21 to −47) |
| F | 50.0% | 50.0% (0.0) | 45.2% (−4.8) | 10–46% (−4 to −40) |
| **Mean MAD** | — | **3.4 pp** | **15.7 pp** | **30–47 pp** |

Gap = predicted − observed (percentage points); negative indicates under-prediction. Conventional scores underestimated observed mortality by 30–47 pp on average; EndoSysScore reduced this to a 3.4 pp mean absolute deviation. RISK-E, the strongest comparator, was aligned with observed mortality in phenotypes C, D and F but underestimated it by 23–31 pp in the rarer A, B and E.

## Model architecture

Stacking ensemble. Three base learners feed a meta-learner; three candidate meta-learners were compared and the best (M3) was selected:

| Model | Meta-learner | AUC | Cal. slope | HL p | DeLong wins | Phenotype MAD |
|---|---|---|---|---|---|---|
| M1 | XGBoost (shallow, depth 3, n = 100) | 0.773 | 1.00 | 0.38 | 4/6 | 13.3 pp |
| M2 | XGBoost (moderate, depth 4, n = 200) | 0.779 | 0.91 | 0.16 | 4/6 | 7.4 pp |
| **M3** | **LightGBM (depth 4, n = 150) — selected** | **0.785** | **0.88** | **0.26** | **6/6** | **3.4 pp** |

Meta-learner inputs combine base-learner out-of-fold predictions with phenotype indicators (`in_pattern_A`–`in_pattern_F`, `n_patterns_satisfied`) and key clinical features. Synthetic in-phenotype patients carry full weight in training; out-of-phenotype synthetic records are down-weighted (G1 weighting). Calibration: isotonic followed by beta calibration, fitted on real development data only.

## TIMA-3 synthetic-data realism test

Ten blinded cardiac surgeons each classified 100 mixed real/synthetic pre-operative profiles (1,000 judgements in total):

| Metric | Value |
|---|---|
| Pooled accuracy | 49.5% (95% CI 46.4–52.6) |
| Cohen κ | −0.01 |
| AUROC | 0.49 |
| Exact binomial p vs 50% | 0.776 |

Synthetic patients were statistically indistinguishable from real ones by expert clinicians.

## Repository structure

```
github_update/
├── README.md
├── phase57c_run.py                              # Full training pipeline (ESS v3)
├── requirements.txt                             # Python dependencies
│
├── data/
│   ├── ZENODO_STUDY_development_4285.csv         # Development cohort (real patients)
│   ├── ZENODO_CONTROL_external_930.csv           # External validation cohort (real patients)
│   └── ZENODO_DATABASE_pre_intra_post_5215.csv   # Full registry (descriptive)
│
├── models/
│   ├── Model_S_v3_WINNER.pkl                      # Final trained model (LightGBM meta, M3)
│   └── feature_names_v3_G.json                    # Feature names and order
│
├── predictions/
│   └── control_predictions_WINNER.csv            # Model + comparator predictions on validation cohort
│
├── results/
│   ├── G_BEST_full_metrics.json                  # Global performance metrics
│   ├── architecture_comparison.csv               # M1 / M2 / M3 comparison
│   ├── delong_WINNER.csv                          # DeLong pairwise comparisons
│   ├── dca_WINNER.csv                             # Decision-curve analysis
│   └── patterns_WINNER.csv                        # Phenotype-level validation
│
├── tima3/
│   ├── TIMA3_responses_raw.csv                    # 1,000 individual rater responses
│   ├── TIMA3_statistics_.csv                      # Per-rater and pooled statistics
│   └── EHJ_TIMA3_10_CLINICIANS.xlsx               # Blinded clinician test workbook
│
└── calculator/
    └── index.html                                # Standalone web calculator (endosysscore.org)
```

## Quickstart

**1. Install dependencies**
```
pip install -r requirements.txt
```
Requires Python ≥ 3.10. Pinned versions: numpy 1.24.4, pandas 2.0.3, scikit-learn 1.3.2, xgboost 1.7.6, lightgbm 4.1.0, scipy 1.11.4, matplotlib 3.7.2, seaborn 0.12.2.

**2. Run the training pipeline**
```
python phase57c_run.py
```
Reproduces the ESS v3 pipeline: loads development + synthetic data, assigns phenotype features (A–F), trains the base learners and the LightGBM meta-learner with G1 weighting, applies isotonic + beta calibration, evaluates on the external validation cohort, and saves model, predictions and metrics.

**3. Use the calculator**
Open `calculator/index.html` in any browser (runs entirely client-side, no data transmitted), or use the live version at https://endosysscore.org.

## Comparator scores

Six established IE-specific surgical risk scores, applied with their published coefficients without recalibration: EuroSCORE II (Nashef 2012), EndoSCORE (Di Mauro 2017), RISK-E (Olmos 2017), AEPEI (Gatti 2016), APORTEI (Varela 2020), STS-IE (Gaca 2011).

**Conflict-of-interest disclosure:** M. Di Mauro, G. Actis Dato and S. Gelsomino are co-authors of EndoSCORE. EndoSCORE is used here only as an external comparator, applied with its originally published coefficients.

## Citation

Gelsomino S, Moula A, Actis Dato GM, Bidar E, Della Corte A, Merlo M, Ceravolo R, Salsano A, Santini F, Scrofani R, Russo C, Rinaldi M, De Vincentiis C, Gulizia M, Musazzi A, Tarzia V, De Bonis M, Nicolini F, Rosato F, Vendramin I, Pacini D, Benedetto U, Luciani GB, Lucà F, Troise G, Luzi G, Paparella D, Formica F, Mastroroberto P, Parise G, Parise O, Parolari A, Pollari F, Rao M, Vizzardi E, Minniti G, Savini C, Colli A, Lorusso R, Di Mauro M, on behalf of the Italian Group of Research for Outcome in Cardiac Surgery (GIROC). *EndoSysScore: phenotype-aware risk estimation for under-recognised high-risk subgroups in surgical infective endocarditis.* European Heart Journal (submitted), 2026. DOI (all versions): [10.5281/zenodo.20327974](https://doi.org/10.5281/zenodo.20327974).

Corresponding authors: Roberto Lorusso, Michele Di Mauro.

## License

MIT License — see `LICENSE`.

**Clinical disclaimer:** This software is a research tool and must not be the sole basis for clinical decisions. It has not been approved by any regulatory agency. Users are responsible for local clinical validation before any use beyond research.

© 2026 Sandro Gelsomino, Maastricht University Medical Centre.
