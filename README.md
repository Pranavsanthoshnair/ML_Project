# House Price Prediction Using Machine Learning
## Feature Selection Investigation — Team 04
| Name |
|------|
| Pranav S Nair |
| Ananthakrishnan K V |
| Anna Mariya Shibu |
| Aishwarya Pramod Nair |

**College:** St. Joseph's College of Engineering and Technology, Palai (Autonomous)  
**Course:** 24SJPCCST503 – Machine Learning  
**Academic Year:** 2026–2027

---

## Investigation Topic
**Feature Selection**

## Research Question
> **Does adding more features necessarily improve prediction?**

## Operational Question
> How does the *type* of added feature — informative, irrelevant, or redundant —
> affect predictive performance, generalization, stability, and model behaviour
> in house-price prediction?

---

## 1. Overview

This project is an **experimental ML investigation**, not a production prediction system.
We run six controlled experiments on the Ames Housing dataset, changing only the
feature configuration between experiments while keeping everything else fixed.
The goal is to produce evidence about how feature type — not just feature count —
affects model performance.

**Reasoning chain:** Question → Hypothesis → Experiment → Evidence → Analysis → Conclusion

---

## 2. Hypotheses

| ID | Hypothesis |
|---|---|
| **H1** | Adding informative features will generally improve predictive performance. |
| **H2** | Adding irrelevant features will not improve performance and may increase error. |
| **H3** | Adding redundant features will provide little additional predictive benefit. |
| **H4** | The effect of feature addition may differ between Ridge and Random Forest. |

> These are hypotheses to be tested — not conclusions. The final answer is based on experimental evidence.

---

## 3. Dataset

**Ames Housing Dataset** (Dean De Cock, 2011 / Kaggle)  
- 2,930 houses × 82 columns  
- Target: `SalePrice` (sale price in USD)  
- Mix of numerical and categorical features  
- File: `data/raw/train.csv`

---

## 4. Experimental Design

Six feature configurations, evaluated with the same train/test split and same models:

| Experiment | Feature Configuration | Raw Features | Purpose |
|---|---|---|---|
| **A** — Core 10 | 10 key informative features | 10 | Minimal meaningful baseline |
| **B** — Informative 20 | Core 10 + 10 more informative | 20 | Effect of more useful features |
| **C** — All Original | All 81 original Ames features | 81 | Full real-world feature set |
| **D** — Orig + Irrelevant | All original + 10 random noise | 91 | Effect of useless features |
| **E** — Orig + Redundant | All original + 5 near-duplicates | 86 | Effect of repeated information |
| **F** — Orig + Both | All original + irrelevant + redundant | 96 | Combined effect |

**Controlled variables:** Same dataset, same target, same 80/20 split (random_state=42),
same model hyperparameters, same preprocessing, same evaluation metrics.

---

## 5. Models

| Model | Settings | Why |
|---|---|---|
| **Ridge Regression** | `alpha=10.0` | Regularized linear model; L2 penalty absorbs noisy features |
| **Random Forest** | `n_estimators=300, random_state=42` | Non-linear ensemble; different mechanism for handling noise |

Testing both models reveals whether the effect of feature type is universal or model-dependent (H4).

---

## 6. Metrics

| Metric | Interpretation |
|---|---|
| **Test RMSE** | Primary metric — average prediction error in $. Lower = better. |
| **Test MAE** | Average absolute error in $. Less sensitive to outliers. Lower = better. |
| **Test R²** | Proportion of house price variance explained. Higher = better (max 1.0). |
| **Train RMSE** | Training accuracy — compared with Test RMSE to measure overfitting. |
| **Generalization Gap** | Test RMSE − Train RMSE. Smaller = less overfitting. |

---

## 7. Key Results

| Experiment | Ridge Test RMSE | RF Test RMSE | Ridge R² | RF R² |
|---|---|---|---|---|
| A — Core 10 | 39,549 | 31,658 | 0.805 | 0.875 |
| B — Informative 20 | 36,621 | 27,758 | 0.833 | 0.904 |
| C — All Original | 29,161 | 26,711 | 0.894 | 0.911 |
| D — Orig + Irrelevant | 29,082 | 27,560 | 0.895 | 0.905 |
| E — Orig + Redundant | 29,176 | 26,893 | 0.894 | 0.910 |
| F — Orig + Both | 29,087 | 27,632 | 0.894 | 0.905 |

**Pairwise delta summary:**

| Comparison | Ridge Δ Test RMSE | RF Δ Test RMSE | Effect |
|---|---|---|---|
| A → B (+10 informative) | −2,928 | −3,900 | Clear improvement — H1 ✅ |
| C → D (+10 irrelevant) | −78 (noise-level) | +849 | RF degraded — H2 ✅ |
| C → E (+5 redundant) | +15 | +181 | Negligible — H3 ✅ |
| C → F (+15 both) | −74 | +920 | RF degraded — H4 ✅ |

---

## 8. Main Findings

1. **Adding more features does not necessarily improve prediction.**
   Improvement occurs only when features carry new, genuine information.

2. **Informative features reliably improved both models** (H1 supported).
   Ridge RMSE fell 7.4%; RF RMSE fell 12.3% from A→B.

3. **Irrelevant features did not help either model** (H2 supported).
   Ridge was robust (−$78, noise-level). RF was clearly hurt (+$849).

4. **Redundant features provided negligible benefit** (H3 supported).
   Changes of +$15 (Ridge) and +$181 (RF) are practically insignificant.

5. **Model type matters** (H4 supported).
   Ridge's regularization absorbed noise; RF's random splits were diluted by it.

---

## 9. Limitations

- Irrelevant features are pure random noise — may understate harm of real uninformative features
- Redundant features are linearly constructed — real redundancy can be non-linear
- Single train/test split — small effects (Ridge C→D: −$78) need cross-validation to confirm
- Ames Housing only — findings may not generalise to other markets or domains
- Only two model types studied

---

## 10. Repository Structure

```
ML_Project/
├── README.md                        ← This file
├── requirements.txt                 ← Python dependencies
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── train.csv                ← Ames Housing dataset
│   └── README.md
│
├── notebooks/
│   └── Project_Code.ipynb           ← PRIMARY NOTEBOOK (30 markdown + 20 code cells)
│
├── docs/
│   ├── investigation.md             ← Complete research report (25 sections)
│   ├── methodology.md               ← Step-by-step methodology guide
│   ├── results_interpretation.md    ← Detailed interpretation of all results
│   ├── research_questions_and_viva.md ← Comprehensive viva preparation (A–J)
│   └── ai_usage.md                  ← Honest AI usage declaration
│
├── results/
│   ├── tables/
│   │   ├── main_results.csv         ← All 12 experiment results
│   │   ├── comparison_results.csv   ← Delta values for 4 comparisons
│   │   ├── feature_summary.csv      ← Feature configuration overview
│   │   └── redundancy_evidence.csv  ← Pearson correlations for redundant features
│   ├── figures/                     ← Generated plots (after running notebook)
│   ├── metrics/
│   └── analysis/
│
├── report/
│   └── README.md                    ← Place final report PDF here
│
└── presentation/
    └── README.md                    ← Place final presentation slides here
```

The notebook is the **primary implementation artifact**. It is self-contained —
all code, documentation, and analysis live in one file readable from top to bottom.

---

## 11. How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Dataset

Place `train.csv` (Ames Housing dataset) at:
```
data/raw/train.csv
```

Download from [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).

### Run the Notebook

**Launch Jupyter from the project root** (important for the data path):
```bash
cd c:\Users\PRANAV\ML_Project
jupyter notebook notebooks/Project_Code.ipynb
```

Then: **Kernel → Restart & Run All**

> The notebook will automatically find `data/raw/train.csv` using `os.getcwd()`.
> If it doesn't, edit `DATA_PATH` in Cell 2 to the full path.

### Save Results (Optional)

The final cell of the notebook saves all result CSVs and figures to `results/`.

---

## 12. Documentation Guide

| Document | What it covers | Who should read it |
|---|---|---|
| `docs/investigation.md` | Complete 25-section research report | Everyone |
| `docs/methodology.md` | Step-by-step methodology | Before viva |
| `docs/results_interpretation.md` | Detailed interpretation of every number | Before viva |
| `docs/research_questions_and_viva.md` | 100+ Q&A in 10 categories | Viva preparation |
| `docs/ai_usage.md` | AI usage declaration | Submission |

---

## 13. AI Usage

AI tools were used for code documentation, README writing, and research report drafting.
Team members verified all methodology, ran all experiments, reviewed all results, and
drew their own conclusions. See `docs/ai_usage.md` for full declaration.

---

## 14. References

1. De Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an End of
   Semester Regression Project. *Journal of Statistics Education*, 19(3).

2. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python.
   *Journal of Machine Learning Research*, 12, 2825–2830.

3. Hoerl, A. E., & Kennard, R. W. (1970). Ridge Regression: Biased Estimation for
   Nonorthogonal Problems. *Technometrics*, 12(1), 55–67.

4. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.

5. Guyon, I., & Elisseeff, A. (2003). An Introduction to Variable and Feature Selection.
   *Journal of Machine Learning Research*, 3, 1157–1182.
