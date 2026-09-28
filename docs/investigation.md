# Investigation Report: Does Adding More Features Necessarily Improve Prediction?

**Team:** 04  
**Course:** 24SJPCCST503 – Machine Learning  
**Institution:** St. Joseph's College of Engineering and Technology, Palai (Autonomous)  
**Academic Year:** 2026–2027  
**Team Members:** Pranav S Nair, Ananthakrishnan K V, Anna Mariya Shibu, Aishwarya Pramod Nair

---

## 1. Investigation Overview

This document presents the complete research report for Team 04's machine learning investigation. The project is not a house-price prediction application — it is a controlled experiment that uses house-price prediction as the domain to study a fundamental question in feature engineering: **does adding more features to a model necessarily make it better?**

The experiment systematically varies the type and quantity of features given to two regression models while holding everything else constant. By comparing results across six controlled configurations, the team produces evidence about how informative, irrelevant, and redundant features each affect predictive performance, generalization, and model behaviour.

The methodology follows the pattern: **Question → Hypothesis → Experiment → Evidence → Analysis → Conclusion.**

---

## 2. Official Research Question

> **Does adding more features necessarily improve prediction?**

This is a yes/no question that requires evidence. The answer is not assumed in advance — it must come from the experimental data.

---

## 3. Operational Research Question

> How does the *type* of added feature — informative, irrelevant, or redundant — affect predictive performance, generalization, stability, and model behaviour in house-price prediction?

The operational question breaks the main question into testable components. Instead of just asking "more or fewer features?", it asks about the specific mechanism: does it matter *what kind* of additional feature is added? This is what the six experimental conditions are designed to test.

---

## 4. Hypotheses

The following four hypotheses were formulated before running the experiments. They represent reasoned predictions based on ML theory, not guaranteed outcomes.

**H1 — Informative Features**  
Adding informative property features will generally improve predictive performance because they provide additional, genuinely useful signal about house prices that the model can learn from.

*Reasoning:* More relevant features reduce the uncertainty in predicting SalePrice. A feature like `Fireplaces` or `BsmtFin SF 1` carries real information about the property that is not fully captured by the core 10 features.

**H2 — Irrelevant Features**  
Adding features that contain no predictive information will not systematically improve performance and may negatively affect generalization or increase error.

*Reasoning:* Irrelevant features (pure random noise) carry no signal. A well-regularized linear model should largely ignore them, but a tree-based model must split on them sometimes, potentially wasting decision capacity and reducing accuracy.

**H3 — Redundant Features**  
Adding highly redundant features (near-duplicate versions of existing informative features) will provide little additional predictive benefit because their information is already represented by the original predictors.

*Reasoning:* If the model already has access to `Gr Liv Area`, a feature that is 99.9% correlated with it adds almost no new information. The model already has the signal.

**H4 — Model Dependence**  
The effect of irrelevant and redundant features may differ between Ridge Regression and Random Forest because the two models handle feature redundancy and noise differently.

*Reasoning:* Ridge uses L2 regularization, which penalizes coefficients and tends to shrink the contribution of noisy or redundant features toward zero. Random Forest uses random feature subsets at each split, so irrelevant features may occasionally be selected as the best split, diluting the model's capacity to find good splits on informative features.

> These are hypotheses to be tested — not conclusions. The final answer must come from the experimental evidence.

---

## 5. Relevant ML Theory

### 5.1 Feature Selection and Feature Types

In supervised machine learning, a feature is a measurable property of the data that the model uses to make predictions. Not all features are equally useful.

- **Informative features** have a statistical relationship with the target variable. Adding them gives the model new, useful signal. Example: `Overall Qual` has a strong positive correlation with `SalePrice`.
- **Irrelevant features** have no meaningful relationship with the target. They are noise. In this investigation, irrelevant features are explicitly constructed as random normal variables with no connection to `SalePrice`.
- **Redundant features** carry information that is already present in other features. Their correlation with existing predictors is very high (r ≥ 0.99 in this investigation). Adding them does not introduce new information — it duplicates existing information.

### 5.2 The Curse of Dimensionality

When the number of features grows relative to the number of training samples, models must estimate more parameters from the same amount of data. This can lead to overfitting: the model fits the noise in training data rather than the true pattern, and performance on unseen test data deteriorates. In this investigation, the dataset has 2930 rows. Adding 10 irrelevant features increases the raw feature count from 81 to 91, and the transformed (post one-hot encoding) count from 302 to 312.

### 5.3 Overfitting and Generalization

Overfitting occurs when a model learns patterns that are specific to the training data and do not generalize to new data. It is diagnosed by comparing training error and test error:

- If **Train RMSE << Test RMSE**, the model is overfitting: it performs much better on seen data than unseen data.
- The **generalization gap** (Test RMSE − Train RMSE) quantifies overfitting. A larger gap indicates worse generalization.

In this investigation, Random Forest consistently shows a larger generalization gap than Ridge, because it can memorize training data more aggressively.

### 5.4 Evaluation Metrics

**RMSE (Root Mean Squared Error):** The square root of the average squared prediction error. Lower is better. It penalizes large errors more than small ones due to the squaring. Used as the primary comparison metric here because SalePrice has a wide range and outliers exist.

**MAE (Mean Absolute Error):** The average of absolute prediction errors. Lower is better. Less sensitive to large outliers than RMSE. Used as a supplementary metric.

**R² (Coefficient of Determination):** The proportion of variance in SalePrice that the model explains. Range: 0 to 1 (higher is better; 1.0 = perfect). R² = 0.90 means the model explains 90% of the variation in house prices.

**Generalization Gap:** Test RMSE minus Train RMSE. Measures how much worse the model is on unseen data compared to training data. Smaller gap = better generalization.

### 5.5 Ridge Regression

Ridge Regression is a regularized form of linear regression that adds a penalty term (L2 penalty) to the loss function:

Loss = Σ(y - ŷ)² + α × Σ(β²)

where α (alpha) controls the strength of regularization. In this investigation, α = 10.0.

The L2 penalty discourages large coefficients. This means:
- Noisy or irrelevant features tend to get small (near-zero) coefficients — Ridge shrinks them toward zero.
- Redundant features tend to share their coefficients — Ridge distributes the weight evenly among correlated predictors rather than assigning all weight to one.

This makes Ridge **robust to irrelevant and redundant features**, though not completely immune.

### 5.6 Random Forest

Random Forest is an ensemble method that builds many decision trees, each trained on a random subset of training samples (bagging) and a random subset of features at each split. Predictions are the average of all individual tree predictions.

Key properties relevant to this investigation:
- **Implicit feature selection:** At each node, only a random subset of features is considered. Features with more predictive power are selected more often.
- **Non-linear:** Can capture interactions and non-linear relationships that Ridge cannot.
- **Sensitive to irrelevant features through dilution:** If irrelevant features are included, they can occasionally be selected as the best split, especially in shallow trees. This is called feature dilution and can slightly reduce performance.
- **Larger generalization gap:** Random Forest tends to overfit training data more than Ridge (shown clearly in this investigation's generalization gap column).

In this investigation: n_estimators=300, random_state=42, n_jobs=-1.

### 5.7 Preprocessing and Data Leakage

**Data leakage** occurs when information from the test set (or future data) is used to influence the model's training. It produces overly optimistic performance estimates that do not reflect real-world performance.

In this investigation, preprocessing is performed **inside scikit-learn Pipeline objects** to prevent leakage. This means:
- The imputer's median values are learned from the **training data only**.
- The scaler's mean and standard deviation are computed from the **training data only**.
- The one-hot encoder's category lists are determined from the **training data only**.

If preprocessing were done *before* the train/test split, the imputer would have seen the test set's distribution, which would be leakage.

### 5.8 One-Hot Encoding and Feature Expansion

Categorical features with k unique categories are converted to k−1 binary (0/1) columns by one-hot encoding. This is why the "transformed features" count in the results is much higher than the raw feature count.

Example: The `Neighborhood` column has 28 unique neighborhoods → 28 binary columns. This is why 81 raw features become 302 transformed features after one-hot encoding.

---

## 6. Dataset

### 6.1 Overview

**Name:** Ames Housing Dataset  
**Source:** Dean De Cock (2011), "Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project", Journal of Statistics Education.  
**Size:** 2,930 rows × 82 columns  
**Target column:** `SalePrice` (house sale price in USD)  
**Train/test split:** 2,344 training samples, 586 test samples (80/20, random_state=42)

### 6.2 Feature Categories

The dataset contains a rich mix of property characteristics:
- **Size features:** `Gr Liv Area`, `Total Bsmt SF`, `1st Flr SF`, `2nd Flr SF`, `Garage Area`, `Lot Area`
- **Quality features:** `Overall Qual` (1–10 scale), `Overall Cond`, `Exter Qual`, `Kitchen Qual`, `Bsmt Qual`
- **Age features:** `Year Built`, `Year Remod/Add`, `Garage Yr Blt`
- **Room counts:** `Full Bath`, `Bedroom AbvGr`, `TotRms AbvGrd`, `Fireplaces`, `Garage Cars`
- **Categorical:** `Neighborhood` (28 values), `House Style`, `Foundation`, `Sale Type`, `MS Zoning`, and many more

### 6.3 Missing Values

The dataset has significant missing values in several columns. The top missing-value columns are:

| Column | Missing Values |
|---|---|
| Pool QC | 2,917 |
| Misc Feature | 2,824 |
| Alley | 2,732 |
| Fence | 2,358 |
| Mas Vnr Type | 1,775 |
| Fireplace Qu | 1,422 |
| Lot Frontage | 490 |
| Garage cols | ~159 |
| Bsmt cols | ~80 |

Most of these missing values are **not data entry errors** — they indicate the *absence* of a feature. For example, `Pool QC` is missing for 2,917 houses because those houses don't have a pool. The preprocessing pipeline handles these with median imputation (numerical) and most-frequent imputation (categorical).

### 6.4 Suitability for This Investigation

The Ames Housing dataset is well-suited for this experiment because:
1. It has 2,930 rows — enough data to support stable comparisons across 12 model/experiment combinations.
2. It contains a realistic mix of numerical and categorical features, making the preprocessing pipeline (with one-hot encoding) non-trivial and realistic.
3. Its features have genuine, real-world meaning, so the distinction between informative and irrelevant is clear and interpretable.
4. It is well-studied, allowing comparison of the team's Ridge and RF performance to known benchmarks.

### 6.5 Dataset Limitations

- **Geographic scope:** Ames, Iowa only (2006–2010). Results may not generalize to other housing markets.
- **Temporal scope:** Sales from 2006–2010 only. Housing market conditions have changed significantly since then.
- **Artificially added features:** The irrelevant and redundant features in experiments D, E, F are constructed by the team — they are not genuine features of the dataset.

---

## 7. Experimental Design

### 7.1 What Changed

Only the **feature configuration** changes between experiments. Specifically:
- Which columns are included in the feature matrix X
- Whether artificially constructed irrelevant or redundant columns are appended

### 7.2 What Was Controlled

Everything else is kept strictly constant:
- **Dataset:** Same 2,930 rows
- **Target variable:** SalePrice
- **Train/test split:** Same 80/20 split with random_state=42 (2,344 train, 586 test)
- **Model architectures:** Ridge(alpha=10.0), RandomForestRegressor(n_estimators=300, random_state=42)
- **Preprocessing logic:** Same pipeline structure, same imputation strategies
- **Evaluation metrics:** RMSE, MAE, R², Generalization Gap
- **Random seeds:** RANDOM_STATE=42 throughout

### 7.3 Why the Fixed Split Matters

Using the **same train/test split** across all six experiments is critical for a fair comparison. If each experiment used a different split, differences in results could be due to chance variation in which houses ended up in train vs. test — not due to the feature configuration. With a fixed split, any difference in Test RMSE between experiments A and D is attributable only to the different feature set.

### 7.4 The Six Experimental Conditions

| Experiment | Raw Features | Configuration | Purpose |
|---|---|---|---|
| A_Core_10 | 10 | 10 core informative features | Minimal meaningful baseline |
| B_Informative_20 | 20 | Core 10 + 10 additional informative | Effect of adding useful information |
| C_All_Original | 81 | All original Ames features | Full real-world feature set |
| D_Original_Plus_Irrelevant | 91 | All 81 + 10 random noise features | Effect of adding useless information |
| E_Original_Plus_Redundant | 86 | All 81 + 5 redundant features | Effect of adding repeated information |
| F_Original_Plus_Both | 96 | All 81 + 10 irrelevant + 5 redundant | Combined effect |

---

## 8. Feature Configurations

### 8.1 Core 10 Features (Experiment A)

These 10 features were selected because they are among the strongest predictors of SalePrice based on domain knowledge and correlation analysis:

`Overall Qual`, `Gr Liv Area`, `Garage Cars`, `Total Bsmt SF`, `1st Flr SF`, `Year Built`, `Year Remod/Add`, `Full Bath`, `TotRms AbvGrd`, `Garage Area`

They represent: overall quality rating, living area size, garage capacity, basement size, main floor area, construction year, renovation year, bathroom count, room count, and garage size.

### 8.2 Additional 10 Features (Experiment B = A + these)

`Overall Cond`, `Bedroom AbvGr`, `Fireplaces`, `Garage Yr Blt`, `Mas Vnr Area`, `BsmtFin SF 1`, `Bsmt Unf SF`, `2nd Flr SF`, `Wood Deck SF`, `Open Porch SF`

These are secondary informative features — they do have a genuine relationship with SalePrice, but their individual predictive power is slightly weaker than the core 10.

### 8.3 All Original Features (Experiment C)

All 81 columns from the original Ames Housing dataset (after removing the target SalePrice and the `Order`/`PID` identifier columns). After one-hot encoding of categorical features, this expands to 302 transformed features.

### 8.4 Irrelevant Features (Experiments D and F)

10 features named `RandomFeature_1` through `RandomFeature_10`, each drawn independently from a standard normal distribution (mean=0, std=1) using np.random.RandomState(42). These have no mathematical or real-world relationship with SalePrice.

### 8.5 Redundant Features (Experiments E and F)

5 features that are near-perfect linear copies of existing informative features, with tiny amounts of Gaussian noise added:

| Redundant Feature | Original | Multiplier | Noise SD | Pearson r |
|---|---|---|---|---|
| Redundant_GrLivArea | Gr Liv Area | 1.02 | 10 | 1.000 |
| Redundant_OverallQual | Overall Qual | 1.00 | 0.20 | 0.990 |
| Redundant_GarageArea | Garage Area | 1.01 | 5 | 1.000 |
| Redundant_TotalBsmtSF | Total Bsmt SF | 0.99 | 10 | 1.000 |
| Redundant_YearBuilt | Year Built | 1.00 | 2 | 0.998 |

All five redundant features have Pearson correlations ≥ 0.990 with their originals, confirming that they carry essentially the same information.

---

## 9. Informative Features Analysis (A → B)

### 9.1 Results

| Model | Test RMSE (A) | Test RMSE (B) | Δ Test RMSE | Δ Test R² | Δ Gen Gap |
|---|---|---|---|---|---|
| Ridge | 39,548.984 | 36,621.425 | **−2,927.559** | +0.028 | −1,106.086 |
| Random Forest | 31,658.027 | 27,758.095 | **−3,899.932** | +0.029 | −3,145.939 |

### 9.2 Interpretation

Both models improved substantially when the additional 10 informative features were added. Ridge's Test RMSE dropped by approximately $2,928, and Random Forest's dropped by approximately $3,900. R² improved by 0.028–0.029 for both models.

Importantly, the generalization gap *decreased* for both models (Ridge: −$1,106, RF: −$3,146). This is an important signal: adding more informative features not only improved accuracy but also reduced overfitting. This is because more informative features help the model focus on the true underlying patterns rather than fitting noise in the limited feature set.

### 9.3 Hypothesis Assessment

**H1 is supported.** Adding informative features improved performance measurably for both model types. The improvement was consistent across all four metrics (RMSE ↓, MAE ↓, R² ↑, Gen Gap ↓).

---

## 10. Irrelevant Features Analysis (C → D)

### 10.1 Results

| Model | Test RMSE (C) | Test RMSE (D) | Δ Test RMSE | Δ Test R² | Δ Gen Gap |
|---|---|---|---|---|---|
| Ridge | 29,160.886 | 29,082.412 | **−78.474** | +0.001 | −32.793 |
| Random Forest | 26,711.359 | 27,560.391 | **+849.033** | −0.006 | +638.369 |

### 10.2 Interpretation

The two models responded differently to the addition of 10 irrelevant random features:

**Ridge:** Test RMSE decreased by $78.47 — a negligible change (0.27% improvement). The R² increase of 0.001 is statistically meaningless. Ridge essentially ignored the random features due to L2 regularization. The regularization penalty forced the coefficients for random features toward zero, so they contributed almost nothing to predictions. The tiny apparent improvement is likely random variation from the single train/test split rather than a real effect.

**Random Forest:** Test RMSE *increased* by $849.03 — a noticeable degradation (~3.2%). The generalization gap also increased by $638. RF was hurt by the irrelevant features because random trees occasionally selected random features as the best split node (due to the random feature subsetting in each tree), wasting decision capacity that could have been used on genuinely informative features. This is the feature dilution effect.

### 10.3 Unexpected Finding

Ridge showed a marginal *improvement* (Δ = −78.47) when irrelevant features were added. This is unexpected based on theory — we would expect neutral or slight degradation. The most likely explanation is that with a single train/test split, small numerical differences can arise from chance variation. The effect is so small (< 0.3%) that it should not be interpreted as a genuine benefit of adding irrelevant features. It is within the noise level of a single-split evaluation.

### 10.4 Hypothesis Assessment

**H2 is supported.** Adding irrelevant features provided no systematic improvement. Ridge was essentially unaffected (Δ ≈ −78, noise-level). RF showed clear degradation (+849). The evidence confirms that irrelevant features do not help and can hurt, especially for tree-based models.

---

## 11. Redundant Features Analysis (C → E)

### 11.1 Results

| Model | Test RMSE (C) | Test RMSE (E) | Δ Test RMSE | Δ Test R² | Δ Gen Gap |
|---|---|---|---|---|---|
| Ridge | 29,160.886 | 29,175.642 | **+14.757** | −0.000 | +11.004 |
| Random Forest | 26,711.359 | 26,892.592 | **+181.234** | −0.001 | +108.781 |

### 11.2 Interpretation

Both models became marginally worse when 5 redundant features were added. The changes are small: Ridge RMSE increased by $14.76 (0.05%), and RF RMSE increased by $181.23 (0.68%). In both cases, R² changed by −0.000 to −0.001, which is negligible.

The redundant features had Pearson correlations of 0.990–1.000 with their originals. This means they carry essentially no new information. Ridge handled them well: L2 regularization distributes the coefficient weight among correlated predictors without large individual effects. RF also handled them reasonably: because the redundant features are nearly identical to existing features, they can substitute for the originals in some tree splits with minimal effect on predictions.

The tiny degradation in both models likely comes from the slight noise added to the redundant features (e.g., Redundant_GrLivArea = Gr Liv Area × 1.02 + N(0,10)). This noise is small but real.

### 11.3 Hypothesis Assessment

**H3 is supported.** Adding highly redundant features provides negligible additional benefit. Both models showed only minimal change (+14 to +181 RMSE increase), which is consistent with "no meaningful benefit." The near-zero correlations to the original features make the result predictable.

---

## 12. Combined Features Analysis (C → F)

### 12.1 Results

| Model | Test RMSE (C) | Test RMSE (F) | Δ Test RMSE | Δ Test R² | Δ Gen Gap |
|---|---|---|---|---|---|
| Ridge | 29,160.886 | 29,087.210 | **−73.676** | +0.001 | −28.335 |
| Random Forest | 26,711.359 | 27,631.613 | **+920.254** | −0.006 | +571.673 |

### 12.2 Interpretation

Experiment F adds both 10 irrelevant and 5 redundant features (15 extra features total) to all 81 original features. The combined effect is essentially the sum of the D and E individual effects:

- Ridge: −73.68 ≈ −78.47 (D alone) + 14.76 (E alone) = −63.71 (approximately additive, with some interaction)
- RF: +920.25 ≈ +849.03 (D alone) + 181.23 (E alone) = +1,030.26 (slightly less than additive, may show some saturation)

The pattern confirms that the dominant driver in experiment F is the irrelevant features (experiment D's effect dominates over experiment E's effect). Ridge's marginal apparent improvement disappears when the interpretation is that it is noise-level variation. RF's degradation is real and meaningful.

---

## 13. Model Behaviour Comparison

### 13.1 Absolute Performance

At baseline (Experiment C, All Original Features), Random Forest outperforms Ridge:
- Ridge: Test RMSE = 29,160.886, R² = 0.894
- RF: Test RMSE = 26,711.359, R² = 0.911

RF's non-linear learning captures complex interactions between features (e.g., the interaction between `Neighborhood` and `Overall Qual`) that Ridge cannot express.

### 13.2 Generalization Gap

Random Forest consistently shows a much larger generalization gap than Ridge:
- Ridge generalization gaps: 3,154 to 6,221 (Train RMSE is 22,905 to 35,289)
- RF generalization gaps: 16,865 to 20,877 (Train RMSE is 9,846 to 10,781)

This reflects RF's tendency to memorize training data (low Train RMSE) while still generalizing reasonably well to test data (moderate Test RMSE). RF with 300 trees is a high-capacity model.

### 13.3 Robustness to Feature Noise

Ridge is more robust than RF to irrelevant features:
- Ridge Δ Test RMSE (C→D): −78 (effectively neutral)
- RF Δ Test RMSE (C→D): +849 (clear degradation)

This is because Ridge's L2 regularization penalizes the coefficients for random features toward zero, while RF's random feature selection mechanism can inadvertently select them.

### 13.4 H4 Assessment

**H4 is supported.** The two models behaved differently in response to feature type. Ridge was robust to both irrelevant and redundant features, while RF showed more sensitivity to irrelevant features. This difference is explained by the regularization mechanism in Ridge vs. the random feature selection in RF.

---

## 14. Metrics Explanation

| Metric | Formula | Interpretation | Good value |
|---|---|---|---|
| Train RMSE | √(mean((y_train − ŷ_train)²)) | Training accuracy | Lower |
| Test RMSE | √(mean((y_test − ŷ_test)²)) | Generalization accuracy | Lower |
| Test MAE | mean(|y_test − ŷ_test|) | Average absolute error | Lower |
| Test R² | 1 − SS_res/SS_tot | Proportion of variance explained | Higher (max 1.0) |
| Generalization Gap | Test RMSE − Train RMSE | Degree of overfitting | Smaller |

For this dataset (SalePrice values ranging from ~$13,000 to $755,000, median ~$163,000):
- An RMSE of $26,711 means the model's typical prediction error is about ±$27,000
- An R² of 0.911 means the model explains 91.1% of the variation in house prices
- A generalization gap of $16,865 (RF) vs $6,210 (Ridge) shows Ridge generalizes more consistently

---

## 15. Results

### 15.1 Main Results Table

| Experiment | Model | Raw Features | Transformed Features | Train RMSE | Test RMSE | Test MAE | Test R² | Gen Gap |
|---|---|---|---|---|---|---|---|---|
| A_Core_10 | Ridge | 10 | 10 | 35,288.604 | 39,548.984 | 24,841.636 | 0.805 | 4,260.380 |
| A_Core_10 | Random Forest | 10 | 10 | 10,780.842 | 31,658.027 | 18,053.324 | 0.875 | 20,877.185 |
| B_Informative_20 | Ridge | 20 | 20 | 33,467.131 | 36,621.425 | 22,340.125 | 0.833 | 3,154.294 |
| B_Informative_20 | Random Forest | 20 | 20 | 10,026.849 | 27,758.095 | 16,138.034 | 0.904 | 17,731.246 |
| C_All_Original | Ridge | 81 | 302 | 22,951.019 | 29,160.886 | 16,579.318 | 0.894 | 6,209.867 |
| C_All_Original | Random Forest | 81 | 302 | 9,846.204 | 26,711.359 | 15,867.277 | 0.911 | 16,865.155 |
| D_Original_Plus_Irrelevant | Ridge | 91 | 312 | 22,905.338 | 29,082.412 | 16,544.567 | 0.895 | 6,177.074 |
| D_Original_Plus_Irrelevant | Random Forest | 91 | 312 | 10,056.867 | 27,560.391 | 16,294.933 | 0.905 | 17,503.524 |
| E_Original_Plus_Redundant | Ridge | 86 | 307 | 22,954.772 | 29,175.642 | 16,594.005 | 0.894 | 6,220.871 |
| E_Original_Plus_Redundant | Random Forest | 86 | 307 | 9,918.657 | 26,892.592 | 15,925.157 | 0.910 | 16,973.935 |
| F_Original_Plus_Both | Ridge | 96 | 317 | 22,905.678 | 29,087.210 | 16,546.178 | 0.894 | 6,181.532 |
| F_Original_Plus_Both | Random Forest | 96 | 317 | 10,194.785 | 27,631.613 | 16,441.177 | 0.905 | 17,436.828 |

### 15.2 Pairwise Comparison Table (Delta Values)

| Model | Comparison | Feature Change | Δ Test RMSE | Δ Test MAE | Δ Test R² | Δ Gen Gap |
|---|---|---|---|---|---|---|
| Ridge | A → B | Additional informative | −2,927.559 | −2,501.511 | +0.028 | −1,106.086 |
| RF | A → B | Additional informative | −3,899.932 | −1,915.289 | +0.029 | −3,145.939 |
| Ridge | C → D | Additional irrelevant | −78.474 | −34.751 | +0.001 | −32.793 |
| RF | C → D | Additional irrelevant | +849.033 | +427.656 | −0.006 | +638.369 |
| Ridge | C → E | Additional redundant | +14.757 | +14.687 | −0.000 | +11.004 |
| RF | C → E | Additional redundant | +181.234 | +57.880 | −0.001 | +108.781 |
| Ridge | C → F | Irrelevant + redundant | −73.676 | −33.140 | +0.001 | −28.335 |
| RF | C → F | Irrelevant + redundant | +920.254 | +573.900 | −0.006 | +571.673 |

---

## 16. Figures Description

The notebook generates four figures, saved to `results/figures/`.

### Figure 1: Feature Count vs Test RMSE (`feature_count_vs_rmse.png`)
A line plot with Raw_Features on the x-axis and Test_RMSE on the y-axis, with separate lines for Ridge and Random Forest.

**What to observe:** The line drops steeply from A (10 features) to C (81 features) for both models, showing that genuine informative features reduce error. After C, the line plateaus or slightly increases for D, E, F — showing that adding non-informative features yields no further improvement and can slightly degrade performance.

**Pattern:** A steady decline for informative features (A→B→C), then a plateau or slight uptick for D/E/F.

### Figure 2: Configuration vs R² (`configuration_vs_r2.png`)
A grouped bar chart with Experiment on the x-axis and Test R² on the y-axis, bars grouped by model.

**What to observe:** R² increases from A through C (0.805 → 0.894 for Ridge; 0.875 → 0.911 for RF), then stays flat or slightly decreases at D, E, F. The pattern matches RMSE: improvement only comes from informative features.

### Figure 3: Train vs Test RMSE (`train_vs_test_rmse.png`)
A grouped bar chart showing both Train RMSE and Test RMSE side-by-side for each experiment.

**What to observe:** Random Forest has a much larger gap between Train and Test RMSE than Ridge. This visualizes the generalization gap directly. RF's Train RMSE is approximately 9,846–10,195 while its Test RMSE is 26,711–31,658. Ridge's Train RMSE is 22,905–35,289 while its Test RMSE is 29,082–39,549.

### Figure 4: Redundancy Correlation Heatmap (`redundancy_heatmap.png`)
A seaborn heatmap showing Pearson correlations between the 5 original features and their 5 redundant counterparts (10 features total).

**What to observe:** Off-diagonal correlations between originals and their redundants (e.g., Gr Liv Area vs Redundant_GrLivArea) are 0.99–1.00 (shown in deep red). This visually confirms that the redundant features are near-perfect copies of their originals.

---

## 17. Comparison Analysis

### 17.1 A → B: Adding 10 Informative Features

**What changed:** 10 additional property features that genuinely relate to house price were added to the core 10.

**What happened:** Both models improved significantly. Ridge Test RMSE fell by $2,927.56 and RF Test RMSE fell by $3,899.93. Both models' generalization gaps also decreased.

**Why:** The additional features contain new signal. `BsmtFin SF 1` (finished basement area), `2nd Flr SF` (second floor area), and `Fireplaces` all have real correlations with SalePrice. Adding them allows the model to make more accurate predictions. The reduction in generalization gap is also notable — more informative features give the model less need to fit noise.

### 17.2 C → D: Adding 10 Irrelevant Random Features

**What changed:** 10 columns of random normal noise were appended to all 81 original features.

**What happened:** Ridge barely changed (−$78, noise-level). RF degraded by +$849 and its generalization gap increased by $638.

**Why:** Ridge's L2 penalty effectively zeros out the coefficients of features with no predictive power. RF's random feature subsets occasionally select random features as the best split, diluting model capacity. The different responses confirm H4: model type matters.

### 17.3 C → E: Adding 5 Redundant Features

**What changed:** 5 near-duplicate features (r ≥ 0.990) of existing features were added.

**What happened:** Both models degraded very slightly. Ridge: +$14.76 (0.05%). RF: +$181.23 (0.68%). Generalization gaps increased slightly.

**Why:** The redundant features provide essentially no new information (their Pearson correlations to originals are 0.990–1.000). The tiny degradation comes from the small amount of noise deliberately added to construct the redundant features. Ridge redistributes weights among correlated predictors without large effects. RF occasionally uses the slightly noisier redundant feature instead of the original, causing marginal accuracy loss.

### 17.4 C → F: Adding Both Irrelevant and Redundant

**What changed:** Both 10 irrelevant and 5 redundant features were added together.

**What happened:** Ridge: −$73.68 (noise-level, consistent with C→D pattern). RF: +$920.25 degradation (the dominant effect is from the irrelevant features).

**Why:** The combined effect is close to the sum of the individual effects. Irrelevant features dominate because they introduce more noise than redundant features. RF is more sensitive to this.

---

## 18. Generalization Analysis

The generalization gap (Test RMSE − Train RMSE) reveals how well each model generalizes:

| Experiment | Ridge Gap | RF Gap |
|---|---|---|
| A_Core_10 | 4,260 | 20,877 |
| B_Informative_20 | 3,154 | 17,731 |
| C_All_Original | 6,210 | 16,865 |
| D_Original_Plus_Irrelevant | 6,177 | 17,504 |
| E_Original_Plus_Redundant | 6,221 | 16,974 |
| F_Original_Plus_Both | 6,182 | 17,437 |

**Ridge:** Gap is moderate (3,154–6,221). The model fits training data reasonably but doesn't drastically overfit. The gap *increased* from B to C because adding all 81 features (including many weakly predictive categorical ones) allows the model to overfit more.

**Random Forest:** Gap is large (16,865–20,877). RF memorizes training data very effectively (Train RMSE ≈ 9,846–10,781), but its Test RMSE remains much higher. This is inherent to RF's high capacity. With 300 trees and 2,344 training samples, each tree can memorize training patterns.

**Key observation:** RF's generalization gap is roughly 3–4× larger than Ridge's in every experiment. Yet RF still achieves better Test RMSE in most experiments. This means RF's excellent learning capacity overcomes its higher overfitting tendency — it extracts enough signal from the features to still beat the Ridge model on test data, despite memorizing training data much more.

---

## 19. Cross-Validation Note

**Cross-validation was not performed in this investigation.** The investigation uses a fixed 80/20 train/test split as the primary evaluation method. All reported results (RMSE, MAE, R², Generalization Gap) come from this single split with random_state=42.

Cross-validation (e.g., 5-fold or 10-fold) would provide more stable estimates of model performance by averaging over multiple train/test splits. It would also give a standard deviation of performance, showing how much the results vary with different data subsets.

**Why it was not used:** The primary goal of this investigation is to compare six feature configurations under identical conditions. A fixed split ensures that the comparison is perfectly controlled — all 12 model/experiment combinations are evaluated on the exact same test set. Cross-validation would add robustness but is not necessary for the controlled comparison design.

**Recommended extension:** Adding 5-fold cross-validation would strengthen the conclusions by confirming that the observed differences (especially the small ones, like Ridge C→D Δ = −78) are stable across different data subsets. This is a straightforward extension of the current code.

---

## 20. Complexity: Raw vs Transformed Features

One-hot encoding expands categorical features into multiple binary columns, significantly increasing the feature count after preprocessing:

| Experiment | Raw Features | Transformed Features | Expansion Factor |
|---|---|---|---|
| A_Core_10 | 10 | 10 | 1.0× (all numeric) |
| B_Informative_20 | 20 | 20 | 1.0× (all numeric) |
| C_All_Original | 81 | 302 | 3.7× |
| D_Original_Plus_Irrelevant | 91 | 312 | 3.4× |
| E_Original_Plus_Redundant | 86 | 307 | 3.6× |
| F_Original_Plus_Both | 96 | 317 | 3.3× |

Experiments A and B use only numerical features from the core lists — no categorical features, no one-hot encoding. This is why their raw and transformed counts are identical at 10 and 20.

From C onward, the ~71 categorical columns in the full Ames dataset (neighborhood names, zoning codes, material types, etc.) expand to ~221 one-hot binary columns. This explains the jump from 81 raw to 302 transformed features.

The Ridge model must estimate 302 coefficients (one per transformed feature). The regularization α=10.0 is critical at this scale — without regularization, estimating 302 coefficients from 2,344 samples with many correlated features would produce unstable estimates.

---

## 21. Critical Evaluation

### What the investigation does well:
1. **Controlled experimental design:** Only the feature configuration varies. This is the correct way to isolate the effect of feature type.
2. **Two very different model types:** Testing both a regularized linear model (Ridge) and a tree ensemble (RF) reveals that the answer is model-dependent, not just feature-dependent.
3. **Verified redundancy:** The redundant features were verified to be genuinely redundant using Pearson correlation (r = 0.990–1.000), not just assumed.
4. **Honest reporting:** The results include cases where the model improved (Ridge C→D: −78), even when theory predicts neutral. The investigation does not hide unexpected results.
5. **Clear distinction between noise-level and meaningful effects:** +$14 change is interpreted as negligible; +$849 is interpreted as meaningful.

### What the investigation could do better:
1. **Multiple random seeds for feature generation:** The irrelevant and redundant features were generated with a single random seed. Different seeds could produce slightly different results.
2. **Cross-validation for stability:** A single split is sufficient for the controlled comparison but doesn't confirm result stability.
3. **Feature selection methods:** The investigation doesn't study whether automatic feature selection (e.g., recursive feature elimination) would outperform or approach the manually curated core 10.
4. **Effect size reporting:** Statistical significance of the observed deltas is not tested. For small effects like Ridge C→D (Δ = −78), it is unclear whether this is genuine or random variation.

---

## 22. Unexpected and Contradictory Findings

### 22.1 Ridge Improved with Irrelevant Features (C → D: Δ = −78.474)

**The unexpected finding:** Adding 10 random noise features to Ridge caused a marginal *decrease* in Test RMSE (−$78, or 0.27%). Theory predicts neutral or slight degradation.

**Explanation:** This result is most likely within the noise range of a single train/test split rather than a genuine effect. An RMSE difference of $78 on a baseline of $29,161 is a 0.27% change — well within the expected variation from a single split. If the experiment were repeated with a different random seed for the train/test split, this tiny effect would likely flip sign (and become a slight increase).

A secondary explanation is that Ridge's regularization actually achieves a slight ensemble-averaging effect: with more features (even noisy ones), the model's optimization landscape changes slightly, and the regularized solution might land at a marginally better local minimum on this particular data split. This is a minor theoretical possibility but not a robust, reproducible effect.

**Conclusion:** Do not interpret the Ridge C→D Δ = −78 as evidence that "irrelevant features help Ridge." It is a noise-level artifact of the single-split evaluation. The correct interpretation is "Ridge was robust to irrelevant features (no meaningful change)."

### 22.2 RF Generalization Gap is Large Despite Good Test Performance

**The finding:** RF's generalization gap ranges from $16,865 to $20,877 — roughly 3–4× larger than Ridge. Yet RF still achieves better Test RMSE than Ridge in most experiments.

**Explanation:** This is not a contradiction — it reflects the difference between absolute and relative overfitting. RF's Train RMSE is very low (~$9,846–$10,781) because 300 trees can memorize most training patterns. But its Test RMSE ($26,711–$31,658) is still better than Ridge's Test RMSE ($29,082–$39,549) in most cases. The key insight: a model can overfit *and* generalize well at the same time, as long as the model's capacity allows it to capture both genuine signal and training-specific noise.

---

## 23. Final Answer to the Research Question

> **Does adding more features necessarily improve prediction?**

**No.** Adding more features does not necessarily improve prediction. The answer depends critically on the *type* of feature added:

1. **Adding informative features improves prediction.** When 10 additional informative features were added (A→B), both Ridge (RMSE −$2,928) and RF (RMSE −$3,900) improved substantially. H1 is supported.

2. **Adding irrelevant features does not improve prediction.** Ridge was unaffected (noise-level Δ = −$78). RF was degraded (+$849). H2 is supported.

3. **Adding redundant features provides negligible benefit.** Both models showed minimal change (Ridge: +$15, RF: +$181). H3 is supported.

4. **The model type affects the magnitude of the impact.** Ridge (regularized linear) was more robust to non-informative features. RF (tree ensemble) was more sensitive to irrelevant features. H4 is supported.

**The complete answer:** More features help only when those features carry genuinely new and useful information. When features are noise (irrelevant) or duplicates of existing information (redundant), adding them provides no benefit and can actively hurt performance, especially for tree-based models.

---

## 24. Confidence in Conclusion

The conclusions are based on a single 80/20 train/test split. The large effects (H1: Ridge −$2,928, RF −$3,900) are very likely to be genuine, as they are large enough to not be explained by random split variation. The small effects (H3: Ridge +$15, RF +$181) are meaningful in direction (they show non-improvement) but less reliable in exact magnitude.

The unexpected Ridge C→D finding (−$78) is the weakest result — it is within noise range and should not be treated as a reliable finding. Cross-validation would be needed to confirm it.

Overall confidence level:
- H1 (informative features help): **High** — large, consistent effect across both models
- H2 (irrelevant features don't help): **High for RF** (large +$849 effect), **moderate for Ridge** (noise-level effect)
- H3 (redundant features don't help): **Moderate** — direction is consistent (slight degradation), but magnitude is small
- H4 (model dependence): **High** — clear difference in how Ridge vs RF responds to irrelevant features

---

## 25. AI Usage

AI language model tools were used during this project for:
- Assistance with code structure, debugging, and explanation
- Documentation writing and formatting
- README and report preparation

The team members:
- Verified the experimental methodology independently
- Ran and reviewed all notebook cells
- Interpreted results and drew their own conclusions
- Are responsible for all claims made in this document

All numerical results in this document come from actual Python execution in the notebook. AI did not run experiments, fabricate results, or make scientific conclusions on behalf of the team.

---

*End of Investigation Report*
