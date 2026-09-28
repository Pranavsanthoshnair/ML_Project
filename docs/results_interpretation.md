# Results Interpretation

## Team 04 — House Price Prediction: Feature Selection Investigation
**Course:** 24SJPCCST503 – Machine Learning  
**Institution:** St. Joseph's College of Engineering and Technology, Palai (Autonomous)

> All numbers in this document come directly from notebook execution.
> Nothing is invented or estimated.

---

## The Actual Results

### Main Results Table

| Experiment | Model | Raw Feats | Trans. Feats | Train RMSE | Test RMSE | Test MAE | Test R² | Gen Gap |
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

### Pairwise Comparison Table (Delta Values)

| Model | Comparison | Feature Change | Δ Test RMSE | Δ Test MAE | Δ Test R² | Δ Gen Gap |
|---|---|---|---|---|---|---|
| Ridge | A → B | +10 informative | −2,927.559 | −2,501.511 | +0.028 | −1,106.086 |
| RF | A → B | +10 informative | −3,899.932 | −1,915.289 | +0.029 | −3,145.939 |
| Ridge | C → D | +10 irrelevant | −78.474 | −34.751 | +0.001 | −32.793 |
| RF | C → D | +10 irrelevant | +849.033 | +427.656 | −0.006 | +638.369 |
| Ridge | C → E | +5 redundant | +14.757 | +14.687 | −0.000 | +11.004 |
| RF | C → E | +5 redundant | +181.234 | +57.880 | −0.001 | +108.781 |
| Ridge | C → F | +15 both | −73.676 | −33.140 | +0.001 | −28.335 |
| RF | C → F | +15 both | +920.254 | +573.900 | −0.006 | +571.673 |

---

## Experiment A — Core 10 Features

### Ridge: Test RMSE = 39,548.984 | R² = 0.805
With only 10 numerical features, Ridge predicts SalePrice with an average error
of about **$39,549**. The model explains **80.5%** of house price variance.
The generalization gap is small (**$4,260**) because regularization and the simple
feature set prevent overfitting. This is the Ridge baseline.

### Random Forest: Test RMSE = 31,658.027 | R² = 0.875
RF achieves a much lower Test RMSE (**$31,658**) and higher R² (**0.875**) than Ridge
even with only 10 features. However, its generalization gap is massive (**$20,877**):
Train RMSE = $10,781 vs Test RMSE = $31,658. This is a fundamental property of RF —
it memorizes training data very effectively but generalizes less perfectly.

**Insight for experiment A:** With the same 10 features, RF already outperforms Ridge
on Test RMSE by $7,890. RF extracts non-linear relationships that Ridge cannot.

---

## Experiment B — 20 Informative Features

### Ridge: Test RMSE = 36,621.425 | R² = 0.833 | Δ from A = −2,927.559
Adding 10 more informative features reduced Ridge Test RMSE by **$2,928 (7.4%)**.
R² rose from 0.805 to 0.833. The generalization gap actually *decreased* from $4,260
to $3,154, meaning the additional features helped Ridge generalize better.

**Why?** Features like `2nd Flr SF`, `BsmtFin SF 1`, and `Fireplaces` carry genuine
information about house value. More signal → better predictions. The reduced gap shows
these features add consistent information across the train/test split.

### Random Forest: Test RMSE = 27,758.095 | R² = 0.904 | Δ from A = −3,899.932
RF improved even more strongly: **$3,900 (12.3%)** Test RMSE reduction.
R² rose from 0.875 to 0.904. The generalization gap fell from $20,877 to $17,731.

**H1 assessment at A→B:** ✅ **H1 is supported.** Both models improved substantially
when genuinely informative features were added.

---

## Experiment C — All 81 Original Features

### Ridge: Test RMSE = 29,160.886 | R² = 0.894 | Δ from B = −7,460.539
The largest improvement occurs at the B→C transition for Ridge.
Going from 20 to 81 features drops Test RMSE by another **$7,461** and brings R²
to **0.894**. However, the generalization gap *increases* from $3,154 to $6,210.

**Why does the gap increase?** The extra 61 features include many categorical variables
(Neighborhood, Sale Type, etc.) which expand to **302 transformed features** after
one-hot encoding. Ridge can fit the training data more specifically with 302 features,
but the improvement in test performance is smaller proportionally — mild overfitting.

### Random Forest: Test RMSE = 26,711.359 | R² = 0.911 | Δ from B = −1,046.736
RF improves more modestly at B→C: only **$1,047** improvement. RF's tree-based
structure already captures non-linear patterns effectively with 20 features.
The generalization gap falls to $16,865.

**Experiment C is the reference baseline for D, E, and F comparisons.**

---

## Experiment D — All Original + 10 Irrelevant Features

### Ridge: Test RMSE = 29,082.412 | R² = 0.895 | Δ from C = −78.474
Adding 10 random noise features to Ridge produced a change of only **−$78 (−0.27%)**.
This is a noise-level result — effectively zero meaningful change.

**Why?** Ridge's L2 regularization shrinks the coefficients of features with no
predictive power toward zero. The 10 random features receive near-zero weights,
contributing essentially nothing. The −$78 result is within single-split variability
and should **not** be interpreted as "irrelevant features helped."

### Random Forest: Test RMSE = 27,560.391 | R² = 0.905 | Δ from C = +849.033
RF degraded by **+$849 (3.2%)**. R² fell from 0.911 to 0.905.
The generalization gap increased by $638.

**Why?** At each tree node, RF randomly selects a subset of features to consider.
With 10 noise features added to 91 total (11% noise), some splits use a noise feature
as the best split — reducing tree accuracy. Unlike Ridge, RF has no mechanism to
assign zero weight to useless features during training.

**H2 assessment at C→D:** ✅ **H2 is supported.** Ridge was unaffected (−$78, noise-level).
RF showed meaningful degradation (+$849). No systematic improvement from irrelevant features.

---

## Experiment E — All Original + 5 Redundant Features

### Ridge: Test RMSE = 29,175.642 | R² = 0.894 | Δ from C = +14.757
Adding 5 near-perfect copies of existing features caused a change of only **+$15 (0.05%)**.
This is negligible.

**Why?** The redundant features have Pearson correlations of 0.990–1.000 with their
originals. Ridge redistributes weight between original and redundant without improving
predictions — the model sees two almost-identical columns and splits credit between them.
The tiny noise in the redundant features contributes marginally to the $15 increase.

### Random Forest: Test RMSE = 26,892.592 | R² = 0.910 | Δ from C = +181.234
RF showed a slightly larger but still small change: **+$181 (0.68%)**.

**Why?** RF occasionally selects a redundant feature (which is slightly noisier than
its original) at a tree split instead of the cleaner original. This marginal noise
selection causes the small degradation.

**H3 assessment at C→E:** ✅ **H3 is supported.** Both models showed negligible change.
Verified redundancy (r ≥ 0.990) produces near-zero additional predictive benefit.

---

## Experiment F — All Original + Both Irrelevant and Redundant

### Ridge: Test RMSE = 29,087.210 | R² = 0.894 | Δ from C = −73.676
Ridge again shows a noise-level change (−$74, −0.25%). The combined effect
is consistent with the individual effects: Ridge neutralizes both irrelevant
and redundant features through regularization.

### Random Forest: Test RMSE = 27,631.613 | R² = 0.905 | Δ from C = +920.254
RF degraded by **+$920 (3.4%)** — the dominant effect is from the irrelevant features
(consistent with C→D: +$849). The redundant features add modest additional degradation.

**C→F assessment:** The combined effect is dominated by the irrelevant features
for RF. Ridge remains essentially unaffected. Results are consistent with the
individual C→D and C→E findings.

---

## Generalization Gap Analysis

| Experiment | Ridge Gap | RF Gap | Ridge Gap Change from C | RF Gap Change from C |
|---|---|---|---|---|
| A_Core_10 | 4,260 | 20,877 | — | — |
| B_Informative_20 | 3,154 | 17,731 | — | — |
| C_All_Original | 6,210 | 16,865 | (reference) | (reference) |
| D_Original+Irrel | 6,177 | 17,504 | −33 | +638 |
| E_Original+Redund | 6,221 | 16,974 | +11 | +109 |
| F_Original+Both | 6,182 | 17,437 | −28 | +572 |

**Key observations:**

1. **RF gap is always 2.5–4.9× larger than Ridge** — this is structural, not caused
   by any feature configuration. RF with 300 trees memorizes training data much more
   than Ridge with regularization.

2. **Ridge gap stays stable C→D→E→F** (range: $6,177–$6,221, variation of $44).
   Irrelevant/redundant features do not meaningfully change Ridge's overfitting.

3. **RF gap increases C→D (+638) and C→F (+572)** — irrelevant features specifically
   increase RF's overfitting. This is because RF's training performance is unchanged
   (still memorizes training) while test performance degrades.

4. **A and B have different gap patterns:** Ridge gap falls from A (4,260) to B (3,154)
   — informative features help Ridge generalize better. RF gap also falls from A (20,877)
   to B (17,731). Both models generalize better with more informative features.

---

## Model Behaviour Comparison (in context of feature selection)

| Behaviour | Ridge | Random Forest |
|---|---|---|
| Response to informative features | Improved substantially | Improved substantially |
| Response to irrelevant features | Essentially unchanged (−$78) | Clearly degraded (+$849) |
| Response to redundant features | Negligible (+$15) | Negligible (+$181) |
| Generalization gap | Small, stable (~$3,154–$6,210) | Large, variable (~$16,865–$20,877) |
| Best Test RMSE achieved | $29,082 (Config D) | $26,711 (Config C) |
| Regularization mechanism | L2 penalty shrinks noisy coefficients | Random feature subsets (no explicit filtering) |

**Why does Ridge handle irrelevant features better?**
L2 regularization's penalty term pushes small/noisy feature coefficients toward zero.
With 10 noise features, Ridge distributes essentially zero weight to them.

**Why does RF struggle more with irrelevant features?**
RF's node-splitting considers a random subset of features at each split (~√91 ≈ 9 features).
With 10 of 91 features being pure noise (~11%), each split has an 11% chance of the
randomly selected subset containing only noise features — reducing split quality.

**H4 assessment:** ✅ **H4 is supported.** Ridge and RF responded very differently
to the same irrelevant features, confirming that model architecture matters.

---

## Complexity: Raw vs Transformed Feature Counts

| Experiment | Raw Features | Transformed Features | Expansion |
|---|---|---|---|
| A_Core_10 | 10 | 10 | 1.0× (all numerical, no encoding) |
| B_Informative_20 | 20 | 20 | 1.0× (all numerical, no encoding) |
| C_All_Original | 81 | 302 | 3.7× |
| D_Original+Irrel | 91 | 312 | 3.4× |
| E_Original+Redund | 86 | 307 | 3.6× |
| F_Original+Both | 96 | 317 | 3.3× |

The jump from 81 raw to 302 transformed in Experiment C happens because the original
Ames dataset contains ~40 categorical columns. One-hot encoding expands each categorical
column into one binary column per unique value:
- `Neighborhood` (28 unique values) → 28 binary columns
- `Sale Type` (9 values) → 9 binary columns
- etc.

Experiments A and B use only the selected numerical features from the core/expanded
lists — no categorical features, so no one-hot expansion.

The 10 random noise features in D are numerical → add exactly 10 to the count.
The 5 redundant features are numerical → add exactly 5 to the count.

---

## Final Synthesis

### What the results collectively show

1. **Informative features reliably improve both models.** A→B→C shows consistent
   improvement across all metrics for both Ridge and RF. Feature count matters
   *when features carry new information.*

2. **Irrelevant features are model-dependent in their harm.** Ridge is robust
   (regularization absorbs noise); RF is vulnerable (random splits diluted by noise).
   The difference is $78 vs $849 — a factor of ~11.

3. **Redundant features provide negligible benefit regardless of model.** The tiny
   degradation (+$15 Ridge, +$181 RF) is consistent with the near-perfect correlations
   (0.990–1.000) — no new information means no new performance.

4. **The answer to the research question is conditional:**
   More features improve prediction → only when those features are genuinely informative.
   More features do not improve prediction → when features are irrelevant or redundant.

5. **Model choice matters for resilience to bad features,** but not for the direction
   of the effect: neither model benefited from irrelevant or redundant features.
