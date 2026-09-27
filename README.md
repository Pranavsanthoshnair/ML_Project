# House Price Prediction Using Machine Learning

## Team 04

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

## 1. Project Overview

This project investigates a fundamental question in machine learning feature engineering:
**Does adding more features necessarily improve prediction?**

Using the Ames Housing dataset and two regression models, we run a series of controlled
experiments where the **only variable that changes** between conditions is the feature
configuration — the number and type of predictors given to the model. Everything else
(dataset, target, train/test split, model hyperparameters, evaluation metrics) stays fixed.

This is an **experimental ML investigation**, not simply a house-price prediction application.
The goal is to produce evidence about how different types of additional features — informative,
irrelevant, or redundant — affect predictive performance, generalization, stability, and model
complexity.

---

## 2. Research Question

> **Main Question:** Does adding more features necessarily improve prediction?

> **Operational Question:** How does the *type* of added feature — informative, irrelevant, or
> redundant — affect predictive performance, generalization, stability, and model complexity in
> house-price prediction?

---

## 3. Hypotheses

These are hypotheses to be tested, not expected results to be forced. The conclusion must be
based entirely on the actual experimental evidence.

**H1 — Informative features:**
Adding informative property features will generally improve predictive performance because they
provide additional, genuinely useful information about house prices.

**H2 — Irrelevant features:**
Adding features that contain no predictive information will not systematically improve
performance and may negatively affect generalization or stability.

**H3 — Redundant features:**
Adding highly redundant features will provide little additional predictive benefit because
their information is already represented by existing predictors.

**H4 — Model dependence:**
The effect of irrelevant and redundant features may differ between Ridge Regression and Random
Forest because the two models learn relationships differently.

---

## 4. Dataset

**Name:** Ames Housing Dataset  
**Source:** Dean De Cock (2011) — *"Ames, Iowa: Alternative to the Boston Housing Data as an
End of Semester Regression Project"*, Journal of Statistics Education.
Available on [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

**Target column:** `SalePrice` (house sale price in USD)  
**Size:** ~2,930 houses × 82 columns  

**Why this dataset is suitable:**
- It is a well-studied, real-world regression dataset.
- It contains a diverse mix of numerical and categorical features.
- It provides enough observations (~2,930) to support a controlled comparison across six
  different feature configurations.
- The wide variety of feature types (structural, condition, neighbourhood, sale-related) allows
  a natural grouping into informative, less-informative, and excluded categories.

**Important feature categories:**
- Structural/size: `Gr Liv Area`, `Total Bsmt SF`, `1st Flr SF`, `2nd Flr SF`, `Garage Area`
- Quality/condition: `Overall Qual`, `Overall Cond`, `Exter Qual`, `Kitchen Qual`
- Age-related: `Year Built`, `Year Remod/Add`, `Garage Yr Blt`
- Room counts: `Full Bath`, `Bedroom AbvGr`, `TotRms AbvGrd`, `Fireplaces`
- Categorical: `Neighborhood`, `Sale Type`, `House Style`, `Foundation`, and many more

**Limitations and caveats:**
- The dataset covers only Ames, Iowa (2006–2010). Findings may not generalise to other
  markets or time periods.
- Irrelevant features are **artificially generated** (random noise) rather than real-world
  uninformative variables.
- Redundant features are **artificially constructed** by applying minor transformations and
  noise to existing informative variables. Their redundancy is verified using Pearson
  correlation rather than assumed.
- The analysis is limited to two model types; other models may show different patterns.

---

## 5. Experimental Design

The experiment compares six controlled feature configurations. Everything except the feature
set is held constant.

| Experiment | Feature Configuration | Main Purpose |
|---|---|---|
| **A** — Core 10 | 10 most informative features | Minimal meaningful baseline |
| **B** — Informative 20 | 20 informative features (Core 10 + 10 additional) | Effect of adding genuinely useful information |
| **C** — All Original | All ~81 original Ames features | Full real-world feature set |
| **D** — Original + Irrelevant | All original + 10 random noise features | Effect of adding useless information |
| **E** — Original + Redundant | All original + 5 redundant features | Effect of adding repeated information |
| **F** — Original + Both | All original + 10 irrelevant + 5 redundant | Combined effect |

### Core 10 Informative Features
`Overall Qual`, `Gr Liv Area`, `Garage Cars`, `Total Bsmt SF`, `1st Flr SF`,
`Year Built`, `Year Remod/Add`, `Full Bath`, `TotRms AbvGrd`, `Garage Area`

### Additional 10 Informative Features (B = A + these)
`Overall Cond`, `Bedroom AbvGr`, `Fireplaces`, `Garage Yr Blt`, `Mas Vnr Area`,
`BsmtFin SF 1`, `Bsmt Unf SF`, `2nd Flr SF`, `Wood Deck SF`, `Open Porch SF`

### 10 Irrelevant Features (Experiments D and F)
`RandomFeature_1` through `RandomFeature_10` — independently sampled random normal variables
with no relationship to `SalePrice`. Generated with a fixed random seed for reproducibility.

### 5 Redundant Features (Experiments E and F)
`Redundant_GrLivArea`, `Redundant_OverallQual`, `Redundant_GarageArea`,
`Redundant_TotalBsmtSF`, `Redundant_YearBuilt`

Each redundant feature is a lightly transformed (small multiplier + Gaussian noise) version of
an existing informative feature. Pearson correlations with their originals are measured in the
notebook and reported in `results/tables/redundancy_evidence.csv`.

---

## 6. Models

### Ridge Regression (`Ridge(alpha=10.0)`)
A regularized linear regression model. Ridge penalises large coefficients and is therefore
better than ordinary least squares when features are correlated. It provides a useful linear
baseline and is particularly interesting for studying how regularization interacts with
irrelevant and redundant predictors.

### Random Forest Regressor (`n_estimators=300, random_state=42, n_jobs=-1`)
A tree-based ensemble that learns non-linear relationships by averaging predictions from
many decision trees. It naturally performs implicit feature selection. Including it allows
us to investigate whether the effect of irrelevant and redundant features differs for a
fundamentally different learning approach.

**Why these two models?**  
The pair covers both the linear and non-linear regime, and both are well-understood by
B.Tech students, making the results interpretable during a viva. Hyperparameters are fixed
across all six experiments so the only changing variable is the feature configuration.

---

## 7. Preprocessing

Preprocessing is performed **inside scikit-learn `Pipeline` objects** to prevent data leakage.
This means all transformations (imputers, scalers, encoders) are fitted **only** on the
training fold during cross-validation, and only on the training split during the main
experiment. Test data is never seen during fitting.

| Step | Numerical Features | Categorical Features |
|---|---|---|
| Missing values | Median imputation (`SimpleImputer`) | Most-frequent imputation (`SimpleImputer`) |
| Encoding | — | One-hot encoding (`OneHotEncoder`, `handle_unknown="ignore"`) |
| Scaling | `StandardScaler` (Ridge only) | — |

- **Why median for numerical?** Robust to outliers, which are present in `SalePrice` and
  several size-related features.
- **Why most-frequent for categorical?** Simple and effective for features with a dominant
  category (e.g., `Sale Condition`).
- **Why scale only for Ridge?** Linear models with regularization are sensitive to feature
  magnitude. Tree-based models (Random Forest) are scale-invariant, so scaling is omitted.
- **Why pipelines?** To guarantee that no test-set statistics leak into the model during
  training or cross-validation. This is a critical correctness requirement.

---

## 8. Evaluation Metrics

Each model/experiment combination is evaluated on both a single held-out test split and with
5-fold cross-validation.

| Metric | Interpretation |
|---|---|
| **Train RMSE** | Root Mean Squared Error on the training set — lower is better |
| **Test RMSE** | Root Mean Squared Error on the test set — lower is better; primary comparison metric |
| **Test MAE** | Mean Absolute Error on the test set — lower is better; less sensitive to outliers |
| **Test R²** | Proportion of variance explained — higher is better; 1.0 = perfect |
| **Generalization Gap** | Test RMSE − Train RMSE — smaller is better; large gap indicates overfitting |
| **CV RMSE Mean** | Average test RMSE across 5 folds — lower is better |
| **CV RMSE Std** | Standard deviation of CV RMSE — lower indicates more stable performance |

> **Quick reminder:**  
> Lower RMSE / MAE = better predictions.  
> Higher R² = more variance explained.  
> Smaller generalization gap = less overfitting.  
> Lower CV std = more stable across different data splits.

---

## 9. Results

Results are generated by running the notebook from top to bottom. After a full run, numerical
results are saved to:

- `results/tables/main_results.csv` — per-experiment, per-model metrics
- `results/tables/cv_results.csv` — 5-fold cross-validation metrics
- `results/tables/feature_summary.csv` — feature configuration overview
- `results/tables/redundancy_evidence.csv` — Pearson correlations for redundant features

A representative preview of the main results (from an actual notebook run):

| Experiment | Model | Raw Features | Test RMSE | Test R² | Gen. Gap |
|---|---|---|---|---|---|
| A_Core_10 | Ridge | 10 | 39,549 | 0.805 | 4,260 |
| A_Core_10 | Random Forest | 10 | 31,658 | 0.875 | 20,877 |
| B_Informative_20 | Ridge | 20 | 36,621 | 0.833 | 3,154 |
| B_Informative_20 | Random Forest | 20 | 27,758 | 0.904 | 17,731 |
| C_All_Original | Ridge | 81 | 29,161 | 0.894 | 6,210 |
| C_All_Original | Random Forest | 81 | 26,711 | 0.911 | 16,865 |
| D_Original_Plus_Irrelevant | Ridge | 91 | 29,082 | 0.895 | 6,177 |
| D_Original_Plus_Irrelevant | Random Forest | 91 | 27,560 | 0.905 | 17,504 |
| E_Original_Plus_Redundant | Ridge | 86 | 29,176 | 0.894 | 6,221 |
| E_Original_Plus_Redundant | Random Forest | 86 | 26,893 | 0.910 | 16,974 |
| F_Original_Plus_Both | Ridge | 96 | 29,087 | 0.894 | 6,182 |
| F_Original_Plus_Both | Random Forest | 96 | 27,632 | 0.905 | 17,437 |

> **Note:** The values above are from an actual run. Re-running the notebook will reproduce
> the same values (random seeds are fixed).

---

## 10. Visualizations

The following figures are generated by the notebook (Section 26 — Save Results):

| Figure | Description |
|---|---|
| `results/figures/feature_count_vs_rmse.png` | Raw feature count vs Test RMSE — directly addresses the main research question |
| `results/figures/configuration_vs_r2.png` | Feature configuration vs Test R² |
| `results/figures/train_vs_test_rmse.png` | Train RMSE vs Test RMSE — shows generalization gap |
| `results/figures/cv_rmse_comparison.png` | 5-fold CV RMSE mean ± std — shows stability |
| `results/figures/redundancy_correlation.png` | Pearson correlation between redundant features and their originals |

Run the notebook and then the final save cell to produce these files.

---

## 11. Key Findings

> **This section should be completed after running the notebook and analyzing the results.**

Based on the preliminary results visible in the notebook outputs, the team will analyze:

- Whether informative features A→B→C improved performance as expected (H1)
- Whether irrelevant features D worsened, maintained, or improved performance vs C (H2)
- Whether redundant features E showed negligible change vs C (H3)
- Whether the patterns differed between Ridge and Random Forest (H4)

*Do not write conclusions here until the team has reviewed and discussed the actual evidence.*

---

## 12. Limitations

The following limitations apply specifically to this investigation:

1. **Artificially generated irrelevant features.** Random normal variables may not accurately
   represent the types of uninformative features encountered in real-world datasets. Real
   uninformative features are often weakly correlated with the target or with other features,
   which can produce different effects from pure noise.

2. **Artificially constructed redundant features.** The redundant features are simple linear
   transformations of existing features with small added noise. More complex forms of
   redundancy (e.g., features that encode the same information non-linearly) are not studied.

3. **Dataset specificity.** The Ames Housing dataset covers a specific city, property types,
   and time period (2006–2010). Results may not generalise to other housing markets or to
   other regression domains.

4. **Limited model types.** Only Ridge Regression and Random Forest are studied. Other models
   (e.g., gradient boosting, SVR, neural networks) are likely to show different patterns,
   especially for irrelevant features.

5. **Fixed train/test split.** A single 80/20 split is used for the main comparison to ensure
   controlled conditions. 5-fold cross-validation is added as a supplementary stability check,
   but the main quantitative comparison is based on one split.

---

## 13. Reproducibility

### Requirements

| Software | Version |
|---|---|
| Python | 3.9+ (tested on 3.12.6) |
| numpy | ≥ 1.23 |
| pandas | ≥ 1.5 |
| scikit-learn | ≥ 1.1 |
| matplotlib | ≥ 3.5 |
| seaborn | ≥ 0.12 |
| jupyter | ≥ 1.0 |

Install all dependencies:

```bash
pip install -r requirements.txt
```

### Random Seeds

- `RANDOM_STATE = 42` — used for train/test split and Random Forest
- Irrelevant features generated with `np.random.RandomState(42)`
- Redundant features generated with `np.random.RandomState(42)`

All random seeds are set at the top of the notebook. Re-running the notebook will produce
identical results.

### How to Run

1. Clone or download this repository.

   ```bash
   git clone <repo-url>
   cd ML_Project
   ```

2. Install dependencies.

   ```bash
   pip install -r requirements.txt
   ```

3. Obtain the Ames Housing dataset.

   Download `train.csv` from
   [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   and place it at:

   ```
   data/raw/train.csv
   ```

4. Open the notebook.

   ```bash
   jupyter notebook notebooks/Team4_HousePrice_Feature_Investigation.ipynb
   ```

   or

   ```bash
   jupyter lab notebooks/Team4_HousePrice_Feature_Investigation.ipynb
   ```

5. Run all cells from top to bottom.

   In Jupyter: **Kernel → Restart & Run All**

6. *(Optional)* After the full run, execute the final cell (Section 26 — Save Results)
   to write tables and figures to `results/tables/` and `results/figures/`.

---

## 14. Project Structure

```
ML_Project/
│
├── README.md                  ← This file
├── requirements.txt           ← Python dependencies
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── train.csv          ← Place the dataset here (not committed)
│   └── README.md              ← Dataset description
│
├── notebooks/
│   └── Team4_HousePrice_Feature_Investigation.ipynb   ← Main notebook (primary implementation)
│
├── results/
│   ├── tables/                ← CSV result files (generated by notebook)
│   ├── figures/               ← PNG plots (generated by notebook)
│   └── README.md              ← Description of result files
│
├── report/
│   └── README.md              ← Place the final report PDF here
│
└── presentation/
    └── README.md              ← Place the final presentation slides here
```

The notebook is the **primary implementation artifact**. The `src/` folder was intentionally
omitted: splitting a beginner-friendly experimental notebook into multiple Python modules would
increase complexity without adding clarity. Every line of code lives in one place and can be
read and explained end-to-end.

---

## 15. AI Usage Declaration

AI tools (including large language model assistants) were used during this project for:

- Assistance with code structure, debugging, and explanation
- Documentation and README writing
- Formatting and proofreading

The team members:

- Verified the experimental methodology independently
- Ran and reviewed every code cell
- Interpreted the results and drew their own conclusions
- Are responsible for all claims made in the report and presentation

The AI did not run experiments, fabricate results, or make scientific conclusions on behalf
of the team. All numerical results come from actual Python execution within the notebook.

---

## 16. References

1. De Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an End of
   Semester Regression Project. *Journal of Statistics Education*, 19(3).

2. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python.
   *Journal of Machine Learning Research*, 12, 2825–2830.

3. Hoerl, A. E., & Kennard, R. W. (1970). Ridge Regression: Biased Estimation for
   Nonorthogonal Problems. *Technometrics*, 12(1), 55–67.

4. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.

5. Guyon, I., & Elisseeff, A. (2003). An Introduction to Variable and Feature Selection.
   *Journal of Machine Learning Research*, 3, 1157–1182.
