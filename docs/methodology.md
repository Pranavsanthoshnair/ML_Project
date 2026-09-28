# Methodology

## Team 04 — House Price Prediction: Feature Selection Investigation
**Course:** 24SJPCCST503 – Machine Learning  
**Institution:** St. Joseph's College of Engineering and Technology, Palai (Autonomous)

---

## Overview

This document explains the complete methodology of the investigation,
step by step, from dataset to final results.
It is designed to be studied before viva: every decision has a reason.

---

## Step 1: Dataset Selection

**Dataset:** Ames Housing (Ames, Iowa, 2006–2010)  
**Source:** Dean De Cock (2011) / Kaggle House Prices competition  
**File:** `data/raw/train.csv`  
**Size:** 2,930 rows × 82 columns

**Why Ames Housing?**
- Large enough (2,930 samples) to support 12 model/experiment combinations reliably
- Rich feature set (81 predictors) covering structural, quality, age, and location
- Mix of numerical and categorical columns — requires a full preprocessing pipeline
- Well-studied benchmark — results can be compared with published work

**Target variable:** `SalePrice` — the actual sale price of each house in USD.

---

## Step 2: Data Inspection and Missing Value Analysis

Before any modelling, we inspect the data:
- Shape: 2,930 rows, 82 columns
- Target column present: Yes
- Duplicate rows: 0
- Missing values: Many columns have missing values

**Missing value pattern:** In Ames Housing, most missing values are semantically
meaningful — they indicate *absence*, not unknown data.
- `Pool QC` missing = house has no pool (99.6% of houses)
- `Fireplace Qu` missing = house has no fireplace (48.5%)
- `Garage Cond/Qual/Type` missing = house has no garage (~5.4%)

These are handled automatically by the preprocessing pipeline (Step 6).

---

## Step 3: Separating Features and Target

```python
X = df.drop(columns=["SalePrice"])  # 81 predictor columns
y = df["SalePrice"]                 # target
# Drop Id if present (row identifier, not a property feature)
```

This gives: X shape (2930, 81), y shape (2930,).

---

## Step 4: Fixed Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
# Result: 2,344 training rows, 586 test rows
```

**Why fixed?** The same split is reused for all six experiments.
This is the critical experimental control: every feature configuration is
evaluated on the exact same 586 test houses.
Without this, performance differences could come from different test samples,
not from the features.

**Why 80/20?** Standard split giving enough training data (2,344) while keeping
a meaningful test set (586). With random_state=42, the split is deterministic
and reproducible.

---

## Step 5: Feature Construction

### 5a. Informative Features (Experiments A and B)
Hand-curated from domain knowledge:

**Core 10** (Experiment A): `Overall Qual`, `Gr Liv Area`, `Garage Cars`,
`Total Bsmt SF`, `1st Flr SF`, `Year Built`, `Year Remod/Add`, `Full Bath`,
`TotRms AbvGrd`, `Garage Area`

**Additional 10** (Experiment B, adds to Core 10): `Overall Cond`, `Bedroom AbvGr`,
`Fireplaces`, `Garage Yr Blt`, `Mas Vnr Area`, `BsmtFin SF 1`, `Bsmt Unf SF`,
`2nd Flr SF`, `Wood Deck SF`, `Open Porch SF`

All 20 are numerical features — no encoding needed in Experiments A and B.

### 5b. Irrelevant Features (Experiments D and F)
```python
rng = np.random.RandomState(42)
for i in range(10):
    X_experiment[f"RandomFeature_{i+1}"] = rng.normal(0, 1, len(X))
```
10 columns of standard normal random numbers with no relationship to SalePrice.

### 5c. Redundant Features (Experiments E and F)
```python
redundant_specs = {
    "Redundant_GrLivArea":    ("Gr Liv Area",    1.02, 10),
    "Redundant_OverallQual":  ("Overall Qual",   1.00, 0.20),
    "Redundant_GarageArea":   ("Garage Area",    1.01, 5),
    "Redundant_TotalBsmtSF":  ("Total Bsmt SF",  0.99, 10),
    "Redundant_YearBuilt":    ("Year Built",     1.00, 2),
}
# new_feature = original * multiplier + normal_noise(0, noise_sd)
```
5 near-perfect copies with Pearson correlations of 0.990–1.000 (verified).

---

## Step 6: Preprocessing Pipeline

**Design principle:** All preprocessing is inside scikit-learn Pipeline objects
to prevent data leakage.

```python
def create_preprocessor(X_data, scale_numeric=True):
    # Numerical: median imputation + optional StandardScaler
    # Categorical: most_frequent imputation + OneHotEncoder
    # Returns a ColumnTransformer
```

| Step | Numerical | Categorical |
|---|---|---|
| Missing values | Median imputation | Most-frequent imputation |
| Encoding | — | One-hot encoding |
| Scaling | StandardScaler (Ridge only) | — |

**Why median imputation?** Robust to outliers in house price data.

**Why most-frequent for categorical?** Simple, effective for high-frequency categories.

**Why OneHotEncoder?** Converts categorical values (e.g., 28 neighborhood names)
into binary columns without false ordinal assumptions.

**Why scale only for Ridge?** Ridge's regularization penalizes all coefficients
equally — different scales would create unfair penalties. Random Forest is
scale-invariant (splits on thresholds, not magnitudes).

**Why Pipeline?** `Pipeline.fit()` → fits preprocessors on training data only.
`Pipeline.predict()` → transforms test data with training statistics.
This prevents test-set information contaminating training.

---

## Step 7: Six Experimental Conditions

| ID | Experiment Name | Raw Features | What was added |
|---|---|---|---|
| A | A_Core_10 | 10 | None — baseline |
| B | B_Informative_20 | 20 | 10 additional informative |
| C | C_All_Original | 81 | All remaining original features |
| D | D_Original_Plus_Irrelevant | 91 | 10 random noise features |
| E | E_Original_Plus_Redundant | 86 | 5 redundant near-copies |
| F | F_Original_Plus_Both | 96 | 10 irrelevant + 5 redundant |

**Controlled variables (same for all experiments):**
- Dataset (Ames Housing, 2,930 rows)
- Target (`SalePrice`)
- Train/test split (same 2,344/586 row indices)
- Model hyperparameters (Ridge alpha=10, RF n_estimators=300)
- Preprocessing strategy (same Pipeline structure)
- Evaluation metrics (RMSE, MAE, R², Generalization Gap)

**Manipulated variable:** Feature set (which columns are given to the model)

---

## Step 8: Models

**Ridge Regression (`Ridge(alpha=10.0)`):**
- Linear model with L2 regularization penalty: loss + 10 × Σwᵢ²
- Shrinks coefficients, especially for noisy/correlated features
- Produces a single set of interpretable coefficients
- Scale-sensitive → requires StandardScaler

**Random Forest Regressor (`n_estimators=300, random_state=42`):**
- 300 decision trees, each on a bootstrap sample of training data
- At each split: considers a random subset of features (default: √n_features)
- Prediction = average over 300 trees → reduced variance
- Scale-invariant → no StandardScaler needed

---

## Step 9: Evaluation

For each of the 12 combinations (6 experiments × 2 models), the pipeline is:
1. Fitted on `X_train_exp` (training subset for this feature configuration)
2. Predicts on `X_train_exp` (training predictions) and `X_test_exp` (test predictions)

Five metrics computed:

| Metric | Formula | Direction |
|---|---|---|
| Train RMSE | √(mean((y_train − ŷ)²)) | Lower = better |
| Test RMSE | √(mean((y_test − ŷ)²)) | Lower = better (primary metric) |
| Test MAE | mean(|y_test − ŷ|) | Lower = better |
| Test R² | 1 − SS_res/SS_tot | Higher = better (max 1.0) |
| Generalization Gap | Test RMSE − Train RMSE | Smaller = less overfitting |

---

## Step 10: Pairwise Comparison

Four comparisons are made to isolate the effect of each feature type:

| Comparison | Effect being measured |
|---|---|
| A → B | Adding 10 informative features |
| C → D | Adding 10 irrelevant random features |
| C → E | Adding 5 redundant features |
| C → F | Adding both irrelevant and redundant features |

For each comparison: Δ Test RMSE = Test RMSE(modified) − Test RMSE(base)
- Negative Δ = improvement; Positive Δ = degradation; Near-zero = no effect

---

## Step 11: Results Interpretation

The final step is analysis: using the numerical results and visualizations
to answer each hypothesis and formulate the evidence-based conclusion.

See `docs/results_interpretation.md` for detailed interpretation
and `docs/investigation.md` for the complete research report.

---

## Summary of Methodological Decisions

| Decision | What was done | Why |
|---|---|---|
| Dataset | Ames Housing, 2,930 rows | Large, rich, well-studied |
| Split | Fixed 80/20, random_state=42 | Same split for all experiments = controlled comparison |
| Preprocessing | Inside Pipeline | Prevents data leakage |
| Irrelevant features | Random normal noise | Cleanest test of irrelevance |
| Redundant features | Linear transform + small noise | Verified by Pearson r ≥ 0.990 |
| Models | Ridge + Random Forest | Linear vs non-linear = tests model dependence |
| Hyperparameters | Fixed (Ridge α=10, RF n=300) | Only features vary |
| Metrics | RMSE, MAE, R², Gen Gap | Standard regression metrics covering accuracy and overfitting |
| Comparisons | Delta table (A→B, C→D, C→E, C→F) | Isolates effect of each feature type |
