# Viva Defence Guide — Team 04
## Feature Selection: Does adding more features necessarily improve prediction?

**Course:** 24SJPCCST503 – Machine Learning  
**Institution:** St. Joseph's College of Engineering and Technology, Palai (Autonomous)  
**Academic Year:** 2026–2027

> One document. Everything a faculty member can ask. Every answer uses our actual results.
> Study this before the viva.

---

## OUR CONCLUSIONS (with evidence)

### C1 — Informative features improve prediction ✅ H1 Supported

**A → B (10 → 20 features)**

| Model | Before (A) | After (B) | Change |
|---|---|---|---|
| Ridge | 39,549 | 36,621 | −2,928 (−7.4%) |
| RF | 31,658 | 27,758 | −3,900 (−12.3%) |

Both models improved on all metrics (RMSE, MAE, R², generalization gap).

**Why:** The additional features (`BsmtFin SF 1`, `2nd Flr SF`, `Fireplaces`, etc.) have genuine relationships with house price. More real signal → better predictions.

---

### C2 — Irrelevant features do NOT improve prediction ✅ H2 Supported

**C → D (81 → 91 features, +10 random noise)**

| Model | Before (C) | After (D) | Change |
|---|---|---|---|
| Ridge | 29,161 | 29,082 | −78 (−0.3%, noise-level) |
| RF | 26,711 | 27,560 | +849 (+3.2%) |

Ridge was unaffected. RF clearly degraded.

**Why:** Ridge's L2 regularization shrinks noise feature coefficients to near zero — they contribute nothing. RF randomly samples features at each split, so noise features occasionally get selected, reducing tree quality.

---

### C3 — Redundant features provide negligible benefit ✅ H3 Supported

**C → E (81 → 86 features, +5 near-copies)**

| Model | Before (C) | After (E) | Change |
|---|---|---|---|
| Ridge | 29,161 | 29,176 | +15 (+0.05%) |
| RF | 26,711 | 26,893 | +181 (+0.68%) |

Verified correlations: Gr Liv Area r=1.000, Overall Qual r=0.990, Garage Area r=1.000, Total Bsmt SF r=1.000, Year Built r=0.998.

**Why:** r ≥ 0.990 means the feature is a near-perfect copy. A model that already has `Gr Liv Area` gains nothing from `Redundant_GrLivArea` — same information.

---

### C4 — Model type determines resilience to bad features ✅ H4 Supported

Same irrelevant features added (C→D):
- Ridge: −$78 → **robust**
- RF: +$849 → **hurt**

**Why:** L2 regularization (Ridge) vs random feature subsampling (RF) are fundamentally different mechanisms. Ridge penalizes useless coefficients to zero. RF has no such filter.

---

### FINAL ANSWER to the Research Question

> **Does adding more features necessarily improve prediction?**

**No. Only informative features improve prediction.**

| Feature Type | Effect | Why |
|---|---|---|
| Informative | ✅ Clearly improved | New signal for the model |
| Irrelevant | ❌ No improvement, hurt RF | No signal; dilutes RF splits |
| Redundant | ❌ Negligible change | Duplicate information |

---

## EVERY QUESTION THE FACULTY CAN ASK

Questions are drawn from the guideline (viva categories: Understanding, Interpretation, Reasoning, What-if).

---

### UNDERSTANDING QUESTIONS

**Q: What exactly did your team investigate?**  
We investigated whether adding more features to a machine learning model always improves its predictions. We tested three types of additional features — informative, irrelevant, and redundant — on two models (Ridge Regression and Random Forest) using the Ames Housing dataset.

**Q: What is your research question?**  
"Does adding more features necessarily improve prediction?"

**Q: What were your hypotheses?**  
H1: Informative features will improve prediction.  
H2: Irrelevant features will not improve and may hurt prediction.  
H3: Redundant features will provide negligible benefit.  
H4: The effect will differ between Ridge and Random Forest.

**Q: What is feature selection?**  
Feature selection is the process of choosing which variables (features) to give to a machine learning model. Not all available features are useful — some carry no information, some duplicate existing ones. Good feature selection improves prediction accuracy, reduces overfitting, and speeds up training.

**Q: What is an informative feature?**  
A feature that has a genuine relationship with the target variable. For example, `Overall Qual` (quality rating) and `Gr Liv Area` (living area) are informative because houses with higher quality and more space genuinely sell for more.

**Q: What is an irrelevant feature?**  
A feature with no relationship to the target. We created 10 irrelevant features using `np.random.normal(0, 1)` — pure random numbers. Their correlation with `SalePrice` is approximately zero by construction.

**Q: What is a redundant feature?**  
A feature that contains information already present in another feature. We created redundant features by taking existing features, multiplying by ~1.0, and adding tiny noise. The result has Pearson r = 0.990–1.000 with the original — near-identical information.

**Q: What ML concept is central to your investigation?**  
Feature selection and its interaction with model regularization. The core concept is that more features only help when they carry new, genuine information. This connects to the curse of dimensionality, regularization theory, and the bias-variance tradeoff.

---

### INTERPRETATION QUESTIONS (Graphs and Results)

**Q: What does the Feature Count vs Test RMSE graph show?**  
The x-axis shows the number of raw features (10, 20, 81, 86, 91, 96). The y-axis shows Test RMSE. Both lines (Ridge and RF) fall steeply from A (10 features) to C (81 features), then plateau or rise slightly at D, E, F. This directly answers the research question visually: improvement stops when features stop being informative.

**Q: What does the Train vs Test RMSE graph show?**  
It shows both training and test RMSE side by side for each experiment. The key observation is that Random Forest has a much larger gap between Train RMSE (~$9,800–$10,800) and Test RMSE (~$26,700–$31,700) compared to Ridge. This confirms RF overfits more than Ridge. Neither model's gap dramatically changes across D, E, F — the gap is a structural property, not caused by irrelevant features.

**Q: What does the R² bar chart show?**  
R² increases from A through C for both models (Ridge: 0.805 → 0.894, RF: 0.875 → 0.911), then plateaus at D, E, F. This confirms the informative feature improvement (A→C) and shows that irrelevant/redundant features add no new explanatory power.

**Q: What does the redundancy heatmap show?**  
It shows Pearson correlations between the 5 original features and their 5 redundant copies. The cells between each original-redundant pair are deep red (0.99–1.00), confirming near-perfect correlation. This is the visual evidence that the redundant features are genuinely redundant, not just claimed to be.

**Q: What does the comparison delta table show?**  
It shows the change in Test RMSE, MAE, R², and generalization gap for each of our four comparisons (A→B, C→D, C→E, C→F), for each model. Negative delta = improvement. The table lets us directly read the effect of each feature type: A→B shows large improvements, C→D shows Ridge neutral and RF degraded, C→E shows both nearly unchanged.

**Q: Why did performance improve from A to C but stop improving after C?**  
A to C adds genuinely informative features — each adds new signal about house price. After C (all 81 original features), the only additions are artificial noise (D), near-duplicate copies (E), or both (F). These add no new signal, so performance stops improving. The model has already extracted all available useful information.

**Q: Why is Random Forest's generalization gap so large?**  
RF with 300 trees can memorize most training patterns — hence Train RMSE ≈ $9,846. But it cannot perfectly transfer those patterns to unseen test data — hence Test RMSE ≈ $26,711. The gap ($16,865) reflects overfitting. Ridge's smaller gap ($6,210) is due to L2 regularization, which prevents extreme coefficient values and limits memorization.

---

### REASONING QUESTIONS

**Q: Why did you choose the Ames Housing dataset?**  
Because it is large enough (2,930 houses) for reliable comparisons across 12 model/experiment combinations, has 81 predictor features giving us room to test different feature configurations, and mixes numerical and categorical features — a realistic setting. It is also a well-studied benchmark, making it credible.

**Q: Why did you choose Ridge and Random Forest?**  
They represent fundamentally different learning approaches: Ridge is linear and regularized; RF is non-linear and ensemble-based. Testing both lets us determine whether the effect of feature type is universal or model-dependent (H4). If we had used only one model, we couldn't have detected the difference between Ridge's robustness and RF's sensitivity to noise features.

**Q: Why did you keep the same train/test split for all experiments?**  
Because we are comparing six feature configurations. If each experiment used a different split, a better result might come from getting easier test houses, not from better features. Using the same 586 test houses for every experiment means the only variable changing is the feature set — a controlled experiment.

**Q: Why did you use RMSE as the primary metric?**  
RMSE measures prediction error in the same units as SalePrice (dollars), making it directly interpretable. It penalizes large errors more than MAE, which matters in house pricing — mispredicting a $500,000 house by $100,000 is worse than mispredicting by $10,000. We also use MAE, R², and generalization gap as complementary metrics.

**Q: Why did you use pipelines for preprocessing?**  
To prevent data leakage. If we fitted the scaler or imputer on the full dataset (including test rows) before splitting, the model would have seen test-set statistics during training. Inside a Pipeline, `fit()` happens only on training data, and `transform()` is applied separately to train and test. This gives honest, unbiased test performance.

**Q: Why alpha=10 for Ridge?**  
It is a moderate regularization strength that prevents the model from overfitting when estimating ~302 coefficients from 2,344 training samples. Lower alpha would allow coefficients to grow large and overfit. Higher alpha would under-fit. We fixed it across all experiments so the only variable is the feature set.

**Q: Why 300 trees for Random Forest?**  
300 trees provides a stable, well-averaged ensemble. More trees reduce variance (predictions stabilize). With 2,344 training samples and 81+ features, 300 trees is computationally reasonable and gives stable results. We fixed it for the same reason as Ridge's alpha — to isolate the feature effect.

**Q: Why did you verify the redundant features using Pearson correlation?**  
Because we claimed the features are redundant — that claim must be backed by evidence, not assumption. Pearson r measures linear correlation. All five pairs showed r ≥ 0.990, confirming near-perfect linear redundancy. Without this verification, H3's premise would be unproven.

---

### WHAT-IF QUESTIONS

**Q: What if you had added 50 irrelevant features instead of 10?**  
Ridge: Still largely robust, but with 50 noise features out of ~131 total (~38%), even α=10 might not fully zero all of them. Small degradation possible.  
RF: Much larger degradation. With ~38% of features being noise, nearly half of every random feature subset at a node would be useless. Test RMSE degradation likely well above $849.

**Q: What if you had removed an important feature (e.g., Overall Qual)?**  
`Overall Qual` is the single most correlated feature with SalePrice. Removing it from the core 10 would significantly hurt experiment A and B. We can see this indirectly: going from C (81 features, includes Overall Qual) to A (10 features, only core features including Overall Qual) already raised Test RMSE from ~$29,161 to ~$39,549. Removing Overall Qual from A would push it even higher.

**Q: What if you had used a higher alpha (e.g., alpha=100) for Ridge?**  
Stronger regularization would make Ridge even more robust to irrelevant features — all noise coefficients would be pushed even closer to zero. However, it might slightly hurt performance on the informative experiments (A, B, C) because regularization would also shrink genuinely useful coefficients. There is a tradeoff: more robustness to noise vs slightly reduced ability to use good features.

**Q: What if you had used cross-validation instead of one train/test split?**  
The large effects (H1: −$2,928, H2-RF: +$849) would still be clearly visible and would be confirmed as consistent across folds. The small effects (H2-Ridge: −$78, H3: +$15) would be better resolved — CV would show whether they are consistent or random. CV would also give standard deviation of performance, showing stability. Our main directional conclusions would not change.

**Q: What if the redundant features were perfectly correlated (r = 1.000) with no noise?**  
The effect would be even smaller — pure duplicates add literally zero new information. With r = 1.000 and no noise, Ridge would distribute weight perfectly symmetrically and performance would be identical to C. RF might still use either one interchangeably at splits. The tiny degradation we observed (Ridge +$15, RF +$181) comes from the small noise in our redundant features — remove the noise and the degradation would essentially vanish.

**Q: What if preprocessing was done before the train/test split?**  
This would cause data leakage. The scaler would compute mean and standard deviation from all 2,930 rows including the 586 test rows. The model would effectively have seen test-set statistics during training. Test RMSE would appear better than it truly is because the scaling was already informed by the test data. Reported improvements might be artifacts of leakage rather than genuine model improvements.

**Q: What if the dataset had only 200 samples instead of 2,930?**  
With fewer samples, results would be much more variable — a single different split could produce very different Test RMSE values. The large effects (A→B) might still be detectable, but the small effects (C→D Ridge: −$78) would be completely lost in noise. RF would overfit more severely with fewer training samples. Cross-validation would become essential rather than optional.

**Q: What if all 81 features were irrelevant?**  
Both models would perform at or near the baseline of predicting the mean SalePrice for every house (R² ≈ 0, RMSE ≈ standard deviation of SalePrice). Ridge would shrink all coefficients toward zero, effectively making every prediction the intercept (mean). RF would split on random features, producing random-looking trees with no systematic relationship to price. This represents the worst possible feature set — no information at all.

---

### CRITICAL THINKING QUESTIONS

**Q: Your results show Ridge "improved" with irrelevant features (−$78). Doesn't that disprove H2?**  
No. −$78 on a baseline of $29,161 is 0.27% — noise-level on a single split. H2 states irrelevant features will not *systematically* improve performance. A 0.27% fluctuation on one split is not systematic. With a different random seed, this would likely be +$78. The RF result (+$849) is the meaningful finding for H2. The correct interpretation is "Ridge was unaffected" not "Ridge improved."

**Q: RF has a huge generalization gap but still beats Ridge on Test RMSE. Is that contradictory?**  
Not at all. A model can overfit AND generalize better than a simpler model if its high capacity captures enough real signal. RF's non-linear patterns (e.g., the interaction between size and quality) are more valuable than the noise it memorizes. Ridge misses those non-linear patterns entirely. The net result: RF wins on Test RMSE despite a larger gap.

**Q: Can you generalize your conclusions beyond Ames Housing?**  
Partially. The directional findings (irrelevant features hurt RF, not Ridge; redundant features negligible) are consistent with ML theory and should hold broadly. The specific numbers (e.g., exactly +$849 for RF with 10 noise features) are dataset-specific. We explicitly state this as Limitation 3: dataset scope.

**Q: Is one train/test split sufficient?**  
For the large effects, yes — an effect of $849 or $2,928 would not reverse with a different split. For the small effects (−$78, +$15), no — cross-validation would be needed to confirm direction and stability. We acknowledge this as Limitation 3.

**Q: Could the results be different if you used different irrelevant features (different random seed)?**  
Slightly, yes. Different random seeds produce different noise values. However, the overall pattern should be stable: Ridge will always be largely unaffected (regularization doesn't depend on the specific noise values), and RF will always degrade by a similar amount (the dilution of ~11% noise features holds regardless of which specific noise values are used). This is another argument for cross-validation with multiple seeds.

---

### ASSESSMENT-SPECIFIC QUESTIONS (from marking criteria)

**Q: What is your research question and why does it matter?** (2 marks — Research Question & Hypothesis)  
"Does adding more features necessarily improve prediction?" This matters because a common intuition is that more data always helps. Our investigation shows this is false — it depends on whether the additional features carry new information. This insight guides practical feature engineering decisions.

**Q: How was your experiment designed?** (3 marks — Experimental Design)  
We compared six feature configurations (A through F) using the same Ames Housing dataset, same 80/20 train/test split (random_state=42), same two models (Ridge, RF), same hyperparameters, and same evaluation metrics. Only the feature set changed between experiments. This controlled design isolates the effect of feature type.

**Q: What evidence did you produce?** (3 marks — Results & Evidence)  
12 rows of quantitative results (6 experiments × 2 models), 4 comparison delta values, 4 figures (feature count vs RMSE, R² bars, train/test RMSE, redundancy heatmap), 4 CSV tables, and Pearson correlation evidence for redundancy (r = 0.990–1.000).

**Q: Why did the results occur? Explain the ML reasoning.** (4 marks — Critical Analysis & ML Reasoning)  
(See conclusions C1–C4 above — each explains WHAT happened and WHY using L2 regularization theory, random feature subsampling mechanics, and information theory. This is the highest-weighted component.)

---

## QUICK FACTS TO MEMORISE

| Fact | Value |
|---|---|
| Dataset | Ames Housing — 2,930 rows, 82 columns |
| Target | SalePrice (house sale price in USD) |
| Train/test split | 80/20, random_state=42, 2,344 train / 586 test |
| Models | Ridge (α=10), Random Forest (n=300, random_state=42) |
| Redundancy correlations | All ≥ 0.990 (1.000, 0.990, 1.000, 1.000, 0.998) |
| Best Ridge Test RMSE | $29,082 (Experiment D) |
| Best RF Test RMSE | $26,711 (Experiment C) |
| Biggest improvement | A→B RF: −$3,900 (−12.3%) |
| Biggest degradation | C→D RF: +$849 (+3.2%) |
| Ridge gap (C) | $6,210 |
| RF gap (C) | $16,865 |
| Hypotheses supported | H1 ✅, H2 ✅, H3 ✅, H4 ✅ |

---

## THREE-SENTENCE SUMMARY FOR VIVA OPENING

"We investigated whether adding more features always improves machine learning predictions by testing six controlled feature configurations on the Ames Housing dataset. Our results show that informative features consistently improved both models, while irrelevant features degraded Random Forest by 3.2% without affecting Ridge, and redundant features had negligible effect on either model. The conclusion is that feature count alone does not determine prediction quality — information content matters, and model architecture determines resilience to useless features."
