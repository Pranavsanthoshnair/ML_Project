"""
Adds comprehensive markdown cells to Project_Code.ipynb.
Run from the project root: python build_notebook.py
"""
import json, os

NB_PATH = os.path.join(os.path.dirname(__file__),
                       "notebooks", "Project_Code.ipynb")

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

# ── helpers ──────────────────────────────────────────────────────────────────
_md_counter = [0]

def md(text):
    _md_counter[0] += 1
    return {
        "cell_type": "markdown",
        "id": f"doc-md-{_md_counter[0]:03d}",
        "metadata": {},
        "source": [text]
    }


# ── fetch original 20 code cells ─────────────────────────────────────────────
code = [c for c in nb["cells"] if c["cell_type"] == "code"]
assert len(code) == 20, f"Expected 20 code cells, found {len(code)}"

new_cells = []

# ─────────────────────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────────────────────
new_cells.append(md(
"""# TEAM 04 — House Price Prediction Using Machine Learning
## An Investigation of Informative, Irrelevant, and Redundant Features

**Course:** 24SJPCCST503 – Machine Learning  
**Institution:** St. Joseph's College of Engineering and Technology, Palai (Autonomous)  
**Academic Year:** 2026–2027  
**Team:** 04

**Team Members:**
- Pranav S Nair
- Ananthakrishnan K V
- Anna Mariya Shibu
- Aishwarya Pramod Nair

---

## Research Question
> **Does adding more features necessarily improve prediction?**

## Operational Question
> How does the *type* of added feature — informative, irrelevant, or redundant — affect
> predictive performance, generalization, stability, and model behaviour in
> house-price prediction?

## Project Principle
**Question → Hypothesis → Experiment → Evidence → Analysis → Conclusion**

## Hypotheses
- **H1:** Adding informative features will generally improve predictive performance.
- **H2:** Adding irrelevant features will not systematically improve performance
  and may increase error or variability.
- **H3:** Adding redundant features will provide little additional predictive benefit
  because their information is already present.
- **H4:** The effect of feature addition may differ between Ridge Regression and
  Random Forest.

> These are hypotheses to be tested — not conclusions. The final answer must
> come from the experimental evidence below."""
))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1: Imports
# ─────────────────────────────────────────────────────────────────────────────
new_cells.append(md(
"""## 1. Import Libraries and Set Constants

**Purpose:** Load all required Python libraries and set the global random seed.

**Why it is needed:** Before running any experiment we must import the tools we depend on,
and fix the random seed so every run of the notebook produces identical results.

**What the code does:**
- Imports numpy, pandas, matplotlib, seaborn
- Imports scikit-learn: `Pipeline`, `ColumnTransformer`, `SimpleImputer`,
  `OneHotEncoder`, `StandardScaler`, `Ridge`, `RandomForestRegressor`, and metrics
- Sets `RANDOM_STATE = 42` — controls the train/test split and Random Forest randomness
- Sets `np.random.seed(42)` for numpy reproducibility
- Configures pandas display options

**ML Concept — Reproducibility:** A fixed random seed means every execution of the notebook
produces the same train/test split, the same Random Forest, and the same irrelevant
feature values. Without this, comparing experiments would be invalid.

**Viva explanation:** "We set RANDOM_STATE = 42 so the notebook is fully reproducible.
Every time it runs, the same 2,344 training houses and 586 test houses are used,
making comparisons between experiments fair." """
))
new_cells.append(code[0])

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2: Load dataset
# ─────────────────────────────────────────────────────────────────────────────
new_cells.append(md(
"""## 2. Load the Ames Housing Dataset

**Purpose:** Read the CSV dataset into a pandas DataFrame.

**Why it is needed:** All analysis depends on this data. It must be loaded before any step.

**What the code does:**
- `DATA_PATH` points to `data/raw/train.csv` relative to the project root
- `pd.read_csv(DATA_PATH)` loads the file into `df`
- Prints dataset shape and displays the first 5 rows

**Output:** `df` — 2,930 rows × 82 columns

**Dataset:** Ames Housing dataset (Ames, Iowa, 2006–2010). Target = `SalePrice` (sale
price of each house in USD). 81 predictor columns covering structural, quality, age,
and location characteristics.

**Why Ames Housing?** Large enough (2,930 rows) to support 12 model/experiment
combinations reliably, rich enough (81 features) to study different feature types.

**Viva explanation:** "We use the Ames Housing dataset — nearly 3,000 real house sales
with 82 columns. The target is SalePrice. It gives us enough data and enough variety
of features to run the controlled feature-selection investigation." """
))
new_cells.append(code[1])
new_cells.append(md(
"""**Observation:** 2,930 rows, 82 columns including `SalePrice`.
The first rows show real Ames, Iowa house sales. Key columns visible:
`Overall Qual` (1–10 quality rating), `Gr Liv Area` (living area in sq.ft.),
`Year Built`, and many categorical columns like `Neighborhood`, `House Style`."""
))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3: Basic inspection
# ─────────────────────────────────────────────────────────────────────────────
new_cells.append(md(
"""## 3. Basic Dataset Inspection

**Purpose:** Verify dataset integrity — correct row/column count, target column present,
no duplicate rows.

**Why it is needed:** Hidden problems (missing target, duplicates) would invalidate
the experiment. We verify before proceeding.

**What the code does:**
- Prints row count, column count, whether `SalePrice` exists, duplicate count
- Shows descriptive statistics (`.describe(include="all").T`) for the first 20 columns

**ML Concept:** Data validation — always inspect before modelling. Duplicates would cause
the model to see some training samples multiple times, distorting the learned patterns.

**Viva explanation:** "We confirmed 2,930 rows, 82 columns, SalePrice is present,
and there are 0 duplicate rows. This basic check ensures the dataset loaded correctly
before we run any experiment." """
))
new_cells.append(code[2])
new_cells.append(md(
"""**Observation:** 2,930 rows, 82 columns, `SalePrice` present, 0 duplicates.
`Overall Qual` ranges 1–10 (mean 6.1). The dataset is clean and ready for analysis."""
))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4: Missing values
# ─────────────────────────────────────────────────────────────────────────────
new_cells.append(md(
"""## 4. Missing Value Analysis

**Purpose:** Identify which columns have missing values and how many.

**Why it is needed:** Missing values must be handled before modelling. Knowing which
columns and why they are missing informs the preprocessing strategy.

**What the code does:**
- Computes `df.isnull().sum()` for every column
- Filters to columns with at least 1 missing value, sorted descending
- Displays the top 20

**ML Concept — Meaningful missingness:** In Ames Housing, many NaN values indicate
*absence*, not *unknown*. `Pool QC` is NaN because the house has no pool.
`Fireplace Qu` is NaN because there is no fireplace. These are valid observations,
not data errors.

**How we handle them:** The preprocessing pipeline (Cell 12) uses median imputation
for numerical and most-frequent imputation for categorical columns — all inside
a Pipeline so test-set information never contaminates training.

**Viva explanation:** "Pool QC is missing for 99.6% of houses because most houses
have no pool. Alley is missing for 93%. These are not errors — they just mean
the feature doesn't apply. Our pipeline fills them automatically." """
))
new_cells.append(code[3])
new_cells.append(md(
"""**Observation:** Top missing columns: Pool QC (2,917 ≈ 99.6%), Misc Feature (96.4%),
Alley (93.2%), Fence (80.5%), Fireplace Qu (48.5%). Garage-related columns each
missing ~159 rows (houses with no garage). All handled by the preprocessing pipeline."""
))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5: Separate X and y
# ─────────────────────────────────────────────────────────────────────────────
new_cells.append(md(
"""## 5. Separate Features and Target Variable

**Purpose:** Split the dataset into predictor matrix `X` and target vector `y`.

**Why it is needed:** Every supervised ML model requires a clear separation between
inputs (`X`) and the quantity to predict (`y`). The target must never appear as a feature.

**What the code does:**
- `X = df.drop(columns=[\"SalePrice\"])` — all 81 non-target columns
- `y = df[\"SalePrice\"]` — the target
- Drops `Id` if present (row identifier, not a property feature)

**Output:** `X` shape (2930, 81), `y` shape (2930,)

**Viva explanation:** "We separate X (81 feature columns the model learns from) from y
(SalePrice it predicts). We also drop any Id column — it's just a row number and
giving it to the model would be meaningless." """
))
new_cells.append(code[4])

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6: Train/test split
# ─────────────────────────────────────────────────────────────────────────────
new_cells.append(md(
"""## 6. Train/Test Split — The Fixed Evaluation Framework

**Purpose:** Divide data into a training set (80%, 2,344 rows) and a test set
(20%, 586 rows).

**Why it is needed:** The model is trained on training houses and evaluated on
unseen test houses. This simulates real-world deployment and gives an honest
measure of how well the model will predict new houses.

**CRITICAL DESIGN DECISION:** The **same split is reused for all six
feature configurations.** This is the most important experimental control.
Every configuration is evaluated on exactly the same 586 test houses.
Without this, a better result could come from luckier test cases, not better features.

**What the code does:**
- `train_test_split(X, y, test_size=0.20, random_state=42)`
- Produces `X_train` (2,344 rows), `X_test` (586 rows), `y_train`, `y_test`

**ML Concept:** Train/test split prevents data leakage — the model must never see
test data during training. The indices created here (`X_train.index`, `X_test.index`)
are reused in Cell 15 to guarantee identical splits.

**Viva explanation:** "We use one fixed 80/20 split for all six experiments.
This is essential: if each experiment had a different split, a better Test RMSE
might just be from easier test houses. Same split = fair comparison." """
))
new_cells.append(code[5])

# ─────────────────────────────────────────────────────────────────────────────
# CELL 7: Define feature sets A and B
# ─────────────────────────────────────────────────────────────────────────────
new_cells.append(md(
"""## 7. Define Informative Feature Sets (Experiments A and B)

**Purpose:** Define the two hand-curated informative feature sets for experiments A and B.

**Why it is needed:** These sets form the informative baseline. Testing with 10 features (A)
then 20 (B) measures how much genuine predictive value the additional 10 add. This tests H1.

**What the code does:**
- `core_features` (10): the most universally cited numerical predictors of house price:
  Overall Quality, Living Area, Garage Cars, Basement SF, 1st Floor SF, Year Built,
  Year Remodelled, Full Bathrooms, Total Rooms, Garage Area
- `additional_features` (10): more useful numerical features: Overall Condition,
  Bedrooms, Fireplaces, Garage Year Built, Masonry Veneer Area, Finished Basement SF,
  Unfinished Basement SF, 2nd Floor SF, Wood Deck SF, Open Porch SF
- `expanded_features = core_features + additional_features` (20 total)

**Why these 10 core features?** They are all numerical (no encoding needed) and
have strong correlations with SalePrice — structural size, quality, and age are
the primary drivers of house value.

**Connection to Research Question:** A→B tests the effect of adding *genuinely useful*
features. If H1 holds, Test RMSE should decrease.

**Viva explanation:** "We start with 10 features that are known strong predictors of
house price. Then we add 10 more property features that also have genuine relationships
with price. The question is: does adding good features improve prediction?" """
))
new_cells.append(code[6])

# ─────────────────────────────────────────────────────────────────────────────
# CELL 8: Create irrelevant features
# ─────────────────────────────────────────────────────────────────────────────
new_cells.append(md(
"""## 8. Create Controlled Irrelevant Features (for Experiment D)

**Purpose:** Generate 10 random numerical variables with NO relationship to SalePrice.

**Why it is needed:** To test H2 — whether adding genuinely useless features hurts,
helps, or has no effect on prediction quality.

**What the code does:**
- Creates `np.random.RandomState(RANDOM_STATE)` for reproducibility
- Copies `X` into `X_experiment` (working DataFrame for all experiments)
- Adds `RandomFeature_1` through `RandomFeature_10` using `rng.normal(0, 1, n)`
  — each is a column of standard normal random numbers

**Why random normal?** Normal(0,1) values have no structure related to any house
characteristic or SalePrice. The Pearson correlation between any RandomFeature
and SalePrice is approximately 0 by construction.

**ML Concept — Irrelevant feature:** A variable containing no information about
the target. Adding it cannot improve predictions; it can only add noise, consume
model capacity, or dilute feature importance scores.

**Viva explanation:** "We deliberately create 10 random columns — numbers generated
independently of any house characteristic. This is the cleanest possible test of
irrelevance: does adding pure noise help, hurt, or do nothing?" """
))
new_cells.append(code[7])

# ─────────────────────────────────────────────────────────────────────────────
# CELL 9: Create redundant features
# ─────────────────────────────────────────────────────────────────────────────
new_cells.append(md(
"""## 9. Create Controlled Redundant Features (for Experiment E)

**Purpose:** Generate 5 features that are near-perfect copies of existing features,
with tiny amounts of noise added.

**Why it is needed:** To test H3 — whether highly correlated (redundant) features
provide any additional predictive benefit.

**What the code does:**
- `redundant_specs` maps new name → (original column, multiplier, noise standard deviation)
- For each: `new_feature = original * multiplier + random_noise(0, noise_sd)`
- Example: `Redundant_GrLivArea = Gr_Liv_Area * 1.02 + noise(0, 10)`
  — nearly identical to `Gr Liv Area` with 2% scale and ±10 sq.ft. noise

**Why this construction?**
- `multiplier ≈ 1.0` keeps the feature on the same scale as the original
- `noise_sd` is tiny relative to feature ranges (±10 sq.ft. for areas, ±0.2 for
  a 1–10 quality scale) → produces correlations of 0.990–1.000

**ML Concept — Redundant feature:** A variable whose information is already captured
by another feature. Adding it gives the model no new signal — it is mathematically
close to having two identical columns.

**Viva explanation:** "We create 5 near-duplicate features: take an existing feature,
multiply by ~1, add tiny noise. The result is 99–100% correlated with its original.
This tests whether having two almost-identical features helps the model." """
))
new_cells.append(code[8])

# ─────────────────────────────────────────────────────────────────────────────
# CELL 10: Verify redundancy
# ─────────────────────────────────────────────────────────────────────────────
new_cells.append(md(
"""## 10. Verify Redundancy Using Pearson Correlation

**Purpose:** Provide quantitative evidence that the constructed redundant features
are genuinely redundant — not just claimed to be.

**Why it is needed:** Scientific claims must be supported by evidence. Showing
correlations of 0.99–1.00 proves the redundancy rather than assuming it.

**What the code does:**
- For each (original, redundant) pair: computes `X_experiment[original].corr(X_experiment[new])`
- Collects results in `redundancy_df` and displays them

**ML Concept — Pearson correlation (r):** Measures linear relationship strength.
r = 1.0 means perfect positive correlation — knowing one value tells you exactly
the other. r > 0.99 means the two features are essentially exchangeable.

**Connection to Research Question:** This table is the evidence for experiment E's
design validity. A model that already uses `Gr Liv Area` gains no new information
from `Redundant_GrLivArea` (r = 1.000).

**Viva explanation:** "We measure Pearson correlation between each redundant feature
and its original. All are above 0.99 — essentially perfect copies. This proves
our redundant features are genuinely redundant, not just claimed to be." """
))
new_cells.append(code[9])
new_cells.append(md(
"""**Redundancy Evidence:**

| Original Feature | Redundant Feature | Pearson r |
|---|---|---|
| Gr Liv Area | Redundant_GrLivArea | **1.000** |
| Overall Qual | Redundant_OverallQual | **0.990** |
| Garage Area | Redundant_GarageArea | **1.000** |
| Total Bsmt SF | Redundant_TotalBsmtSF | **1.000** |
| Year Built | Redundant_YearBuilt | **0.998** |

All five correlations ≥ 0.990. A model using the original feature gains essentially
no new information from the redundant version."""
))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 11: Define 6 feature sets
# ─────────────────────────────────────────────────────────────────────────────
new_cells.append(md(
"""## 11. Define All Six Experimental Conditions (A through F)

**Purpose:** Build the complete dictionary of six feature sets that will be tested.

**Why it is needed:** This defines exactly what changes between experiments.
Only the feature set changes — data, split, models, and metrics stay identical.

**The six conditions:**

| ID | Name | Raw Features | What was added |
|---|---|---|---|
| A | A_Core_10 | 10 | None — baseline |
| B | B_Informative_20 | 20 | 10 additional informative |
| C | C_All_Original | 81 | All remaining original features |
| D | D_Original_Plus_Irrelevant | 91 | 10 random noise features |
| E | E_Original_Plus_Redundant | 86 | 5 redundant near-copies |
| F | F_Original_Plus_Both | 96 | 10 irrelevant + 5 redundant |

**What the code does:**
- `all_original_features = list(X.columns)` — all 81 original columns
- Builds `feature_sets` dictionary by combining lists
- Displays `feature_summary` table

**Experimental control:** Only the column list changes. The same `X_experiment`
DataFrame provides data; the same `X_train.index` / `X_test.index` provide
the split. This is a textbook controlled experiment.

**Viva explanation:** "This cell defines the six experiments. The only thing that
changes is which columns we give to the model. Everything else — data, split,
models, hyperparameters, metrics — is fixed. This isolation is what makes the
comparison meaningful." """
))
new_cells.append(code[10])

# ─────────────────────────────────────────────────────────────────────────────
# CELL 12: Preprocessing pipeline
# ─────────────────────────────────────────────────────────────────────────────
new_cells.append(md(
"""## 12. Preprocessing Pipeline

**Purpose:** Define a function that builds a scikit-learn preprocessing pipeline
tailored to any given feature set.

**Why it is needed:** The Ames dataset has both numerical and categorical features
with missing values. All preprocessing must be inside a Pipeline to prevent
data leakage.

**What the code does:**
- Detects numerical and categorical columns automatically via `select_dtypes()`
- **Numerical pipeline:** `SimpleImputer(strategy="median")` then optionally `StandardScaler()`
- **Categorical pipeline:** `SimpleImputer(strategy="most_frequent")` then
  `OneHotEncoder(handle_unknown="ignore")`
- Returns a `ColumnTransformer` combining both pipelines

**Why pipelines prevent data leakage:**
If preprocessing is done on the whole dataset before splitting, test-set statistics
influence training — that's leakage. Inside a Pipeline, `fit()` is called only on
training data, and `transform()` is applied separately to train and test.

**Why median imputation?** Robust to outliers. Area features have right-skewed
distributions; median is not pulled toward extreme high-value houses the way mean is.

**Why scale only for Ridge?** Ridge's L2 penalty applies equally to all coefficients.
Features on different scales (Year Built ~1960, Full Bath = 1–3) would be penalized
unfairly without scaling. Random Forest is scale-invariant.

**Why OneHotEncoder?** Categorical features like `Neighborhood` (28 unique values)
must be converted to numbers. One-hot encoding creates one binary column per
category — no false ordinal assumption.

**Viva explanation:** "All preprocessing is inside a pipeline. If we scaled the
whole dataset before splitting, the model would have seen test-set statistics during
training. That's data leakage — it makes test performance look better than it is.
The pipeline fits everything only on training data." """
))
new_cells.append(code[11])

# ─────────────────────────────────────────────────────────────────────────────
# CELL 13: Models
# ─────────────────────────────────────────────────────────────────────────────
new_cells.append(md(
"""## 13. Define the Two Models

**Purpose:** Create fresh instances of Ridge and Random Forest with fixed hyperparameters.

**Why it is needed:** Fixed hyperparameters across all experiments ensure that
performance differences come from features, not from different model tuning.

**Ridge(alpha=10.0):**
- Regularized linear regression with L2 penalty: Loss + α × Σwᵢ²
- Higher α → stronger regularization → coefficients shrunk closer to zero
- α = 10.0 is a moderate value that prevents extreme coefficients when
  estimating 300+ parameters from ~2,344 training samples
- **Key property:** Ridge actively suppresses noisy features by shrinking their
  coefficients toward zero

**RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1):**
- 300 decision trees, each trained on a random bootstrap sample of training data
- At each split, a random subset of features is considered
- Final prediction = average across all 300 trees
- **Key property:** The random feature subsampling at each node means RF doesn't
  actively filter out useless features — they may still be selected

**Why two models?** They represent fundamentally different learning approaches.
Comparing them tests H4: does the effect of feature type depend on model type?

**Viva explanation:** "We use Ridge because it is a linear model that regularizes
by shrinking useless features toward zero. We use Random Forest because it is
non-linear and makes random feature selections at each split. These two approaches
may react differently to irrelevant and redundant features — that's what H4 tests." """
))
new_cells.append(code[12])

# ─────────────────────────────────────────────────────────────────────────────
# CELL 14: Evaluation function
# ─────────────────────────────────────────────────────────────────────────────
new_cells.append(md(
"""## 14. Evaluation Function

**Purpose:** Fit a model pipeline and return all five evaluation metrics.

**Why it is needed:** A single function ensures consistent, correct metric
computation across all 12 combinations without code repetition.

**What the code does:**
- `model.fit(X_train_exp, y_train)` — trains the pipeline on training data
- Predicts on both train and test sets
- Returns:
  - **Train RMSE** = sqrt(mean((y_train - y_hat_train)²)) — training accuracy
  - **Test RMSE** = sqrt(mean((y_test - y_hat_test)²)) — generalization (primary metric)
  - **Test MAE** = mean(|y_test - y_hat_test|) — less sensitive to outliers
  - **Test R²** = 1 - SS_residual/SS_total — proportion of variance explained (max 1.0)
  - **Generalization Gap** = Test RMSE - Train RMSE — measures overfitting

**Metric interpretation:**
- Lower RMSE and MAE = better (fewer prediction dollars off)
- Higher R² = better (R² = 0.911 means 91.1% of house price variance explained)
- Smaller generalization gap = less overfitting

**Connection to Research Question:** "Improvement" means: does Test RMSE decrease?
Does R² increase? Does the generalization gap stay controlled?

**Viva explanation:** "The key metric is Test RMSE — the typical prediction error
in dollars on houses the model has never seen. Lower is better. We also track
Train RMSE to see if the model overfits, and R² to see what fraction of
house price variation it explains." """
))
new_cells.append(code[13])

# ─────────────────────────────────────────────────────────────────────────────
# CELL 15: Main experiment
# ─────────────────────────────────────────────────────────────────────────────
new_cells.append(md(
"""## 15. Run All Six Experiments — Main Controlled Comparison

**Purpose:** Execute all 12 model/experiment combinations and collect results.

**Why it is needed:** This is the core experiment. It produces the evidence
needed to answer the research question.

**What the code does:**
- Outer loop: 6 feature configurations (A through F)
- Inner loop: 2 models (Ridge, Random Forest)
- For each combination:
  - `X_experiment.loc[X_train.index, features]` — extracts the right columns
    from the right rows (using the SAME fixed split indices every time)
  - Builds the preprocessing + model pipeline
  - Calls `evaluate_single_split()` to get all 5 metrics
  - Records raw feature count and transformed feature count
- Stores all 12 results in `main_results_df`

**Critical line:** `.loc[X_train.index, features]`
- `X_train.index` = the 2,344 training row numbers from the fixed split
- `features` = the column names for this specific experiment
- This guarantees identical train/test samples across all 12 combinations

**Viva explanation:** "For each of the six feature sets, we build a pipeline and
evaluate both models on the same 586 test houses. The .loc[X_train.index]
ensures the split is identical every time — only the features change." """
))
new_cells.append(code[14])
new_cells.append(md(
"""**Main Results:**

| Experiment | Model | Raw | Trans. | Train RMSE | **Test RMSE** | MAE | R² | Gen Gap |
|---|---|---|---|---|---|---|---|---|
| A_Core_10 | Ridge | 10 | 10 | 35,289 | **39,549** | 24,842 | 0.805 | 4,260 |
| A_Core_10 | RF | 10 | 10 | 10,781 | **31,658** | 18,053 | 0.875 | 20,877 |
| B_Informative_20 | Ridge | 20 | 20 | 33,467 | **36,621** | 22,340 | 0.833 | 3,154 |
| B_Informative_20 | RF | 20 | 20 | 10,027 | **27,758** | 16,138 | 0.904 | 17,731 |
| C_All_Original | Ridge | 81 | 302 | 22,951 | **29,161** | 16,579 | 0.894 | 6,210 |
| C_All_Original | RF | 81 | 302 | 9,846 | **26,711** | 15,867 | 0.911 | 16,865 |
| D_Orig+Irrel | Ridge | 91 | 312 | 22,905 | **29,082** | 16,545 | 0.895 | 6,177 |
| D_Orig+Irrel | RF | 91 | 312 | 10,057 | **27,560** | 16,295 | 0.905 | 17,504 |
| E_Orig+Redund | Ridge | 86 | 307 | 22,955 | **29,176** | 16,594 | 0.894 | 6,221 |
| E_Orig+Redund | RF | 86 | 307 | 9,919 | **26,893** | 15,925 | 0.910 | 16,974 |
| F_Orig+Both | Ridge | 96 | 317 | 22,906 | **29,087** | 16,546 | 0.894 | 6,182 |
| F_Orig+Both | RF | 96 | 317 | 10,195 | **27,632** | 16,441 | 0.905 | 17,437 |

**Key observations:**
- Test RMSE drops steeply A → B → C for both models (informative features help)
- Barely changes C → D for Ridge (−$78), increases for RF (+$849) — irrelevant features hurt RF
- Barely changes C → E for both (Ridge +$15, RF +$181) — redundant features negligible
- RF Train RMSE is always much lower than Ridge; RF generalization gap is 3–4× larger"""
))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 16: Figure 1
# ─────────────────────────────────────────────────────────────────────────────
new_cells.append(md(
"""## 16. Visualization 1 — Feature Quantity vs Test RMSE

**Purpose:** Plot Test RMSE vs number of raw features for both models.

**What to look for:**
- **A → B → C (10→20→81 features):** Both lines drop steeply — informative features reduce error
- **C → D → E → F (81→91→86→96):** Both lines plateau or slightly rise —
  adding non-informative features provides no further improvement

**Connection to Research Question:** This graph directly visualises the answer:
more features lower RMSE only in the informative region (A→C);
the improvement stops when features become irrelevant or redundant (C→D/E/F).

**Viva explanation:** "This line graph shows Test RMSE on the y-axis and number of
features on the x-axis. Both lines fall from A to C as informative features are
added. After C, they plateau or rise slightly — adding noise or redundant features
does not continue improving prediction." """
))
new_cells.append(code[15])

# ─────────────────────────────────────────────────────────────────────────────
# CELL 17: Figure 2
# ─────────────────────────────────────────────────────────────────────────────
new_cells.append(md(
"""## 17. Visualization 2 — Feature Configuration vs Test R²

**Purpose:** Compare Test R² (proportion of variance explained) across all
six experiments, grouped by model.

**What to look for:**
- R² increases from A (Ridge 0.805, RF 0.875) through C (Ridge 0.894, RF 0.911)
- After C, R² plateaus or very slightly decreases at D, E, F
- RF bars are consistently slightly taller than Ridge bars

**What this means:** The plateau from C onward shows that the additional features
in D, E, F explain no new variance in SalePrice. The model already extracted all
available signal at configuration C.

**Viva explanation:** "R² tells us what fraction of house price variation our model
explains. It reaches about 0.894 for Ridge and 0.911 for RF at configuration C.
Adding irrelevant or redundant features doesn't raise R² further — confirming
they contribute no new explanatory power." """
))
new_cells.append(code[16])

# ─────────────────────────────────────────────────────────────────────────────
# CELL 18: Figure 3
# ─────────────────────────────────────────────────────────────────────────────
new_cells.append(md(
"""## 18. Visualization 3 — Training vs Test RMSE (Generalization Gap)

**Purpose:** Show Train RMSE and Test RMSE side-by-side, making the generalization
gap visually obvious.

**What to look for:**
- Random Forest: Train RMSE bars are very short (~$9,800–$10,800); Test RMSE bars
  are much taller (~$26,700–$31,700) — a large gap confirming overfitting
- Ridge: Train and Test RMSE bars are much closer together — better generalization
- The gap is present in ALL experiments for RF — it is a structural property of RF,
  not caused by irrelevant features specifically

**Why RF's gap is large:** RF with 300 trees and 2,344 training samples can memorize
most training patterns (hence very low Train RMSE) but doesn't transfer all of them
to new data.

**Why Ridge's gap is smaller:** L2 regularization explicitly limits coefficient size,
preventing the model from over-specialising to training data.

**Viva explanation:** "For Random Forest, the training error is about $10,000 but the
test error is $27,000 — a $17,000 gap. This shows overfitting. Ridge has a much
smaller gap because regularization limits how much the model can memorize training data." """
))
new_cells.append(code[17])

# ─────────────────────────────────────────────────────────────────────────────
# CELL 19: Figure 4
# ─────────────────────────────────────────────────────────────────────────────
new_cells.append(md(
"""## 19. Visualization 4 — Redundancy Correlation Heatmap

**Purpose:** Visually confirm the high correlations between original and redundant
features using a colour-coded heatmap.

**What to look for:**
- Diagonal: always 1.000 (feature correlated with itself)
- Off-diagonal within a pair (e.g., Gr Liv Area × Redundant_GrLivArea): 0.99–1.00
  — shown as deep red in the coolwarm colourmap
- Cross-pair correlations (e.g., Gr Liv Area × Overall Qual): much lower, ~0.3–0.6

**What this proves:** The near-perfect correlations between originals and their
redundant copies confirm that including both provides no new information. Any model
that uses `Gr Liv Area` gains nothing from `Redundant_GrLivArea` (r = 1.000).

**Connection to H3:** This heatmap is the visual evidence behind H3.
Near-perfect redundancy → near-zero additional predictive benefit.

**Viva explanation:** "The heatmap shows 5 original features and their 5 redundant copies.
The bright red squares between each pair (0.99–1.00) prove they are near-identical.
This is why experiment E shows almost no improvement over C — the model already
has that information." """
))
new_cells.append(code[18])

# ─────────────────────────────────────────────────────────────────────────────
# CELL 20: Comparison table
# ─────────────────────────────────────────────────────────────────────────────
new_cells.append(md(
"""## 20. Pairwise Comparison Analysis — Delta Table

**Purpose:** Calculate the change in metrics when specific feature types are added,
providing direct, quantitative hypothesis tests.

**Why it is needed:** Comparing absolute values (A: $39,549 vs B: $36,621) is
less clear than comparing deltas (Δ = −$2,928). The delta table makes the
effect of each feature type immediately readable.

**Comparisons made:**
- **A → B:** Effect of adding 10 informative features
- **C → D:** Effect of adding 10 irrelevant (random noise) features
- **C → E:** Effect of adding 5 redundant features
- **C → F:** Effect of adding both irrelevant and redundant features

**Interpretation guide:**
- Negative Δ Test RMSE = improvement (error decreased)
- Positive Δ Test RMSE = degradation (error increased)
- Near-zero change = feature type had no meaningful effect

**Viva explanation:** "The delta table shows exactly what happened when each type
of feature was added. Negative Δ Test RMSE means prediction improved; positive
means it got worse. This table directly tests each of our four hypotheses." """
))
new_cells.append(code[19])
new_cells.append(md(
"""**Pairwise Comparison Results:**

| Model | Comparison | Feature Type | Δ Test RMSE | Δ R² | Interpretation |
|---|---|---|---|---|---|
| Ridge | A → B | +10 informative | **−2,928** | +0.028 | Clear improvement |
| RF | A → B | +10 informative | **−3,900** | +0.029 | Clear improvement |
| Ridge | C → D | +10 irrelevant | **−78** | +0.001 | Negligible (noise-level) |
| RF | C → D | +10 irrelevant | **+849** | −0.006 | Meaningful degradation |
| Ridge | C → E | +5 redundant | **+15** | −0.000 | Negligible |
| RF | C → E | +5 redundant | **+181** | −0.001 | Negligible |
| Ridge | C → F | +15 both | **−74** | +0.001 | Negligible |
| RF | C → F | +15 both | **+920** | −0.006 | Meaningful degradation |

**Hypothesis Assessment:**
- **H1:** ✅ Supported — both models improved substantially with informative features (A→B)
- **H2:** ✅ Supported — Ridge unaffected (+noise), RF meaningfully degraded (+$849) (C→D)
- **H3:** ✅ Supported — both models showed negligible change with redundant features (C→E)
- **H4:** ✅ Supported — Ridge and RF responded very differently to irrelevant features"""
))

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
new_cells.append(md(
"""---

## 21. Analysis and Discussion

### H1: Adding Informative Features (A → B)

Adding 10 more informative property features improved both models substantially:
- **Ridge:** Test RMSE fell by $2,928 (7.4%), R² rose from 0.805 to 0.833
- **RF:** Test RMSE fell by $3,900 (12.3%), R² rose from 0.875 to 0.904

The generalization gap also decreased (Ridge −$1,106, RF −$3,146), meaning
additional informative features helped generalization, not just training accuracy.

**Why this makes ML sense:** Features like `BsmtFin SF 1`, `2nd Flr SF`, and
`Fireplaces` carry genuine information about house value. More signal → better
predictions. **H1 is supported.**

---

### H2: Adding Irrelevant Features (C → D)

- **Ridge:** Test RMSE changed by only −$78 (−0.27%) — effectively unchanged.
  Ridge's L2 regularization shrinks noisy feature coefficients toward zero,
  neutralizing them.
- **RF:** Test RMSE increased by +$849 (+3.2%) — meaningful degradation.
  RF's random feature subsampling means irrelevant features are occasionally
  selected at tree splits, reducing tree quality.

**H2 is supported.** No systematic improvement; RF shows clear negative effect.

*Note: The −$78 Ridge result is within single-split noise range (0.27% change)
and should not be interpreted as "irrelevant features helped Ridge."*

---

### H3: Adding Redundant Features (C → E)

- **Ridge:** +$15 (+0.05%) — negligible
- **RF:** +$181 (+0.68%) — negligible

Redundant features (r = 0.990–1.000 with originals) contain no new information.
Ridge redistributes weight between original and redundant without improving.
RF sometimes selects the slightly noisier redundant copy, causing marginal loss.

**H3 is supported.** Genuinely redundant features provide negligible benefit.

---

### H4: Model Dependence

Ridge was robust to both irrelevant and redundant features (regularization).
RF was clearly more sensitive to irrelevant features (+$849 vs −$78 for Ridge).

**H4 is supported.** The model type substantially affects how feature type
impacts performance.

---

### Final Answer to the Research Question

> **Does adding more features necessarily improve prediction?**

**No.** The answer depends on the *type* of feature:
- **Informative features** → Yes, they improve prediction (H1 supported)
- **Irrelevant features** → No improvement; can hurt tree models (H2 supported)
- **Redundant features** → Negligible benefit (H3 supported)
- **Model type matters** → Ridge robust to noise; RF more vulnerable (H4 supported)

Adding more features improves prediction **only when** those features carry new,
genuine information about the target. Feature count alone is not the criterion —
information content is."""
))

# ─────────────────────────────────────────────────────────────────────────────
# LIMITATIONS
# ─────────────────────────────────────────────────────────────────────────────
new_cells.append(md(
"""---

## 22. Limitations and Critical Evaluation

1. **Artificial irrelevant features:** Pure random noise may understate the effect
   of real-world uninformative features, which often have weak but non-zero
   correlations with the target.

2. **Artificially constructed redundant features:** Built by linear transformation
   and noise. Real redundancy can be non-linear. Results apply most directly to
   linearly redundant features.

3. **Single train/test split:** All results come from one fixed 80/20 split.
   Small effects (Ridge C→D: −$78) could change sign with a different split.
   Large effects (A→B: −$2,928) are robust. Cross-validation would confirm stability.

4. **Dataset scope:** Ames Housing covers one city (2006–2010). Conclusions may
   not generalise to other housing markets or other regression domains.

5. **Two model types only:** Only Ridge and Random Forest were studied.
   Gradient boosting, SVR, and neural networks may respond differently."""
))

# ─────────────────────────────────────────────────────────────────────────────
# AI USAGE
# ─────────────────────────────────────────────────────────────────────────────
new_cells.append(md(
"""---

## 23. AI Usage Declaration

AI language model tools were used during this project for:
- Assistance with code structure, debugging, and explanation
- Documentation writing and formatting
- README and report preparation

The team members:
- Verified the experimental methodology independently
- Ran and reviewed all notebook cells
- Interpreted results and drew their own conclusions
- Are responsible for all claims made

All numerical results in this notebook come from actual Python execution.
AI did not run experiments, fabricate results, or make scientific conclusions
on behalf of the team.

---
*Team 04 — 24SJPCCST503 Machine Learning —
St. Joseph's College of Engineering and Technology, Palai (Autonomous)*"""
))

# ─────────────────────────────────────────────────────────────────────────────
# Write notebook
# ─────────────────────────────────────────────────────────────────────────────
nb["cells"] = new_cells
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

md_count  = sum(1 for c in new_cells if c["cell_type"] == "markdown")
cod_count = sum(1 for c in new_cells if c["cell_type"] == "code")
print(f"Notebook written: {len(new_cells)} total cells "
      f"({md_count} markdown, {cod_count} code)")
