# Team 04 — Conclusions and Counter-Arguments

**Investigation Topic:** Feature Selection  
**Research Question:** Does adding more features necessarily improve prediction?  
**Course:** 24SJPCCST503 – Machine Learning  
**Institution:** St. Joseph's College of Engineering and Technology, Palai (Autonomous)

---

> This document answers the three questions the guideline demands:
> **What happened? Why did it happen? How confidently can we make this conclusion?**
>
> It also prepares you for every counter-argument and critical question
> a faculty member could raise during the viva.

---

## PART 1 — CONCLUSIONS

---

### Conclusion 1: Adding more features does NOT necessarily improve prediction.

**What happened?**  
Test RMSE improved from A (10 features) → B (20 features) → C (81 features),
but stopped improving — and in some cases slightly worsened — when we moved to
D (+ 10 irrelevant), E (+ 5 redundant), or F (+ both).

**Evidence:**
- Ridge Test RMSE: A = 39,549 → B = 36,621 → C = 29,161 → D = 29,082 → E = 29,176 → F = 29,087
- RF Test RMSE:    A = 31,658 → B = 27,758 → C = 26,711 → D = 27,560 → E = 26,893 → F = 27,632

The improvement was real and large going A→B→C (informative features).
It stopped and slightly reversed at D, E, F (non-informative features).

**Why it happened:**  
Features improve a model only when they carry new, genuine information about the target.
Once a model has access to all informative predictors (C), adding noise (D) or
near-duplicates (E) does not provide new signal — it only adds complexity.

**Confidence:** High. The effect is large (RMSE drops of $2,928–$10,388) and consistent
across both models and all metrics.

---

### Conclusion 2: Adding informative features reliably improves prediction (H1 supported).

**What happened?**  
Adding 10 more informative features (A→B):
- Ridge: Test RMSE fell by **$2,928 (7.4%)**, R² rose from 0.805 to 0.833
- RF: Test RMSE fell by **$3,900 (12.3%)**, R² rose from 0.875 to 0.904

Adding all original features (B→C):
- Ridge: Test RMSE fell by a further **$7,460 (20.4%)**
- RF: Test RMSE fell by a further **$1,047 (3.8%)**

**Why it happened:**  
The additional features (`BsmtFin SF 1`, `2nd Flr SF`, `Fireplaces`, `Bedroom AbvGr`, etc.)
have genuine correlations with SalePrice. Giving the model more real information allows
it to make better predictions. This is the basic principle of supervised learning:
more relevant signal → better predictions.

**Confidence:** High. The improvement was large, consistent across both models,
and consistent across all metrics (RMSE, MAE, R², generalization gap).

---

### Conclusion 3: Adding irrelevant features does not improve prediction and can hurt tree models (H2 supported).

**What happened?**  
Adding 10 random noise features (C→D):
- Ridge: Test RMSE changed by **−$78 (−0.27%)** — noise-level, effectively zero
- RF: Test RMSE increased by **+$849 (+3.2%)** — meaningful degradation

**Why it happened:**  
Ridge's L2 regularization shrinks the coefficients of noisy, uninformative features
toward zero, effectively neutralizing them. The model assigns near-zero weight to
`RandomFeature_1` through `RandomFeature_10`.

Random Forest's node-splitting randomly samples a subset of features at each decision
node (~√91 ≈ 9 features). With 10 of 91 features being pure noise (~11%), some splits
accidentally select noise features as the best split, reducing tree quality.

**Confidence:** High for RF (large +$849 effect). Moderate for Ridge
(−$78 is noise-level on a single split — consistent with regularization absorbing noise,
but too small to confirm directionally on one split alone).

---

### Conclusion 4: Adding redundant features provides negligible predictive benefit (H3 supported).

**What happened?**  
Adding 5 near-perfect copies (r = 0.990–1.000) of existing features (C→E):
- Ridge: Test RMSE changed by **+$15 (+0.05%)** — negligible
- RF: Test RMSE changed by **+$181 (+0.68%)** — negligible

**Why it happened:**  
A feature with Pearson r = 1.000 to an existing feature is mathematically
interchangeable with that feature — it contains no new information.
Ridge simply distributes weight between original and copy without improving predictions.
RF occasionally uses the slightly noisier copy instead of the clean original,
causing marginal accuracy loss.

**Evidence of genuine redundancy:**
| Original | Redundant Copy | r |
|---|---|---|
| Gr Liv Area | Redundant_GrLivArea | 1.000 |
| Overall Qual | Redundant_OverallQual | 0.990 |
| Garage Area | Redundant_GarageArea | 1.000 |
| Total Bsmt SF | Redundant_TotalBsmtSF | 1.000 |
| Year Built | Redundant_YearBuilt | 0.998 |

**Confidence:** Moderate. Direction is consistent (slight degradation for both models),
but the magnitude is small enough that cross-validation would be needed to confirm
it is not random variation.

---

### Conclusion 5: The effect of feature type depends on model type (H4 supported).

**What happened?**  
Both models responded the same way to informative features (both improved).
But for irrelevant features (C→D):
- Ridge Δ = **−$78** (unchanged)
- RF Δ = **+$849** (degraded)

**Why it happened:**  
Ridge and RF use fundamentally different learning mechanisms:
- Ridge: explicit L2 penalty → noisy features get shrunk toward zero
- RF: random feature subsampling → noisy features can be selected at splits

This model dependence is an important practical insight: regularized linear models
are more robust to irrelevant features than tree ensembles.

**Confidence:** High. The magnitude difference ($78 vs $849) is too large to be
explained by random variation.

---

### Overall Answer to the Research Question

> **"Does adding more features necessarily improve prediction?"**

**No — and the reason depends on the type of feature:**

| Feature type | Effect | Why |
|---|---|---|
| Informative | Clear improvement | New genuine signal → better predictions |
| Irrelevant | No improvement / hurt RF | No signal → noise dilutes tree splits |
| Redundant | Negligible change | Duplicate information → no new signal |

And the model matters: Ridge is more robust to non-informative features than
Random Forest, due to regularization.

---

## PART 2 — COUNTER-ARGUMENTS AND CRITICAL QUESTIONS

These are challenges a faculty member could raise. Each has a prepared answer.

---

### Counter 1: "Ridge slightly improved with irrelevant features (−$78). Doesn't that contradict H2?"

**Counter-argument:** If irrelevant features shouldn't help, why did Ridge improve slightly?

**Our answer:**  
An $78 change on a baseline of $29,161 is a 0.27% change — well within the noise
range of a single train/test split. If we ran the same experiment with 10 different
random seeds for the split, this tiny effect would sometimes be +$78 and sometimes
−$78. It is not a reproducible, directional finding.

H2 states irrelevant features will *not systematically improve* performance.
A 0.27% random fluctuation is not systematic improvement. The correct interpretation
is: "Ridge was robust to irrelevant features (no meaningful change)."

The RF result (+$849) is the meaningful finding for H2 — large enough to be clearly
directional.

---

### Counter 2: "You only used one train/test split. How do you know the results are reliable?"

**Counter-argument:** A different split might give completely different results.

**Our answer:**  
This is a genuine limitation. We acknowledge it.

For large effects (A→B: −$2,928 Ridge, −$3,900 RF), the results are almost certainly
reliable — an effect of nearly 10–12% would not reverse with a different split.

For small effects (Ridge C→D: −$78, Ridge C→E: +$15), we cannot confirm direction
without cross-validation. These are treated as "no meaningful change," not as
directional findings.

Cross-validation with 5–10 folds would confirm stability. We identify this explicitly
as a limitation and a recommended extension.

---

### Counter 3: "Your irrelevant features were pure random noise. Real uninformative features aren't like that."

**Counter-argument:** In practice, uninformative features often have weak correlations
with the target (r ≈ 0.05–0.15), not zero. Your experiment might understate the
real-world harm.

**Our answer:**  
This is correct and we acknowledge it as a limitation. Pure random normal variables
(r ≈ 0 with target) represent the most extreme form of irrelevance. Real-world
"useless" features often have small but non-zero correlations, which may actually
cause more harm to RF (because the model might assign them meaningful split importance)
and less harm to Ridge (because regularization might not fully zero them out).

Our results therefore provide a lower bound on the harm of irrelevant features for RF.
The true harm in practice is likely as large or larger.

---

### Counter 4: "Your redundant features were artificially constructed. Does this apply to real redundancy?"

**Counter-argument:** Real multicollinearity is more complex than a linear copy with noise.

**Our answer:**  
Correct. Our redundant features are deliberately simple: `original × multiplier + noise`.
This produces near-perfect linear correlations (0.990–1.000), which is the strongest
possible form of linear redundancy.

Real datasets may have non-linear redundancy (e.g., `total_area` and `floor_area`
are related but not perfectly linearly). Our experiment applies most directly to
highly linearly correlated features. For non-linearly redundant features, the effects
on Ridge and RF might differ from our findings.

This is acknowledged as Limitation 2 in the investigation.

---

### Counter 5: "Random Forest has a massive generalization gap (up to $20,877). Isn't that overfitting?"

**Counter-argument:** If RF is overfitting so badly, why does it still outperform Ridge on Test RMSE?

**Our answer:**  
RF overfits in an absolute sense: Train RMSE ≈ $9,846–$10,781, Test RMSE ≈ $26,711–$31,658.
The gap is large ($16,865–$20,877).

But a model can overfit AND still generalize better than a less flexible model.
RF's high capacity lets it capture non-linear patterns (e.g., the interaction between
`Overall Qual` and `Gr Liv Area`) that Ridge's linear form cannot. Even though RF
also memorizes some training noise, the genuine patterns it captures are strong enough
to produce better test predictions than Ridge.

Think of it this way: RF's advantage from capturing non-linearity outweighs its
disadvantage from memorizing some noise.

---

### Counter 6: "Why didn't you use more models?"

**Counter-argument:** With only two models, how do you know if the findings are general?

**Our answer:**  
We deliberately chose Ridge and Random Forest because they represent two fundamentally
different learning approaches — one regularized linear, one non-linear tree ensemble.
This contrast is sufficient to demonstrate H4 (model dependence).

Adding more models (gradient boosting, SVR, neural network) would extend the
investigation but was outside the scope of what was required. The course guideline
for Team 04 requires comparing a meaningful feature set with irrelevant and redundant
features — it does not specify a minimum number of models.

That said, different models would likely show intermediate behaviour: gradient boosting
(which also uses trees but adds regularization) would probably be less sensitive to
irrelevant features than RF but more sensitive than Ridge.

---

### Counter 7: "Experiment C uses 302 transformed features. Did the model have enough training data?"

**Counter-argument:** With 302 features and only 2,344 training samples, isn't Ridge
underdetermined?

**Our answer:**  
The feature-to-sample ratio is 302:2,344 ≈ 1:7.8. This is manageable but not trivially
safe for unregularized regression. Without Ridge's L2 penalty (α=10), estimating 302
correlated coefficients from 2,344 samples would be unstable.

The α=10 regularization is precisely what makes this feasible. By shrinking all
coefficients, Ridge stabilizes the solution even when features outnumber what an
unregularized model could safely handle. The fact that Experiment C achieves
Test R² = 0.894 with 302 features confirms that the regularization is working correctly.

---

### Counter 8: "Why did you choose Ames Housing? Couldn't this be dataset-specific?"

**Counter-argument:** Your conclusions might only apply to house price prediction in Ames, Iowa.

**Our answer:**  
Correct — this is explicitly stated as Limitation 3. The Ames dataset covers one
city and one time period (2006–2010). Whether Ridge remains equally robust to
irrelevant features on, for example, a medical diagnosis dataset is unknown.

However, the *mechanisms* we observed (regularization absorbing noise, tree splits
diluted by noise) are general ML principles, not dataset-specific. The Ames dataset
provides empirical evidence for these principles; the principles themselves hold broadly.

We chose Ames Housing because it is large enough (2,930 rows) for reliable
comparisons, rich enough (81 features) to support the six configurations, and
well-studied enough to be credible as an investigation benchmark.

---

### Counter 9: "What would happen if you added 100 irrelevant features instead of 10?"

**Counter-argument (what-if):** Would the effect scale?

**Our answer:**  
For Random Forest: Yes, the harm would likely scale. With 100 irrelevant features
out of 181 total (~55%), more than half of all random feature subsets at each node
would contain mostly noise. The RF's split quality would degrade substantially.
We'd expect a Test RMSE degradation well beyond the +$849 we observed.

For Ridge: Still likely robust, but α=10 might not be sufficient. With 100 noise
features, the regularization would need to zero out far more coefficients. A higher
alpha might be needed to maintain the same robustness. With α=10, we might start
seeing a small genuine degradation.

This is a concrete, testable prediction based on the mechanisms we observed.

---

### Counter 10: "What if you removed an important feature instead of adding features?"

**Counter-argument (what-if):** Your investigation only added features — what about removal?

**Our answer:**  
The investigation was specifically designed to answer "does adding more features
necessarily improve prediction?" — an addition question, not a removal question.

However, the evidence from A (10 features) vs C (81 features) already tells us
about removal: going from C down to A, Ridge Test RMSE rises from $29,161 to $39,549
(+36%). Removing the 71 informative features the model had in C clearly hurts.

The symmetric relationship holds: adding informative features helps, removing them
hurts. Adding non-informative features is neutral/harmful; removing them would be
neutral/slightly beneficial.

---

## PART 3 — THREE-QUESTION SUMMARY

As required by the course guideline (Section 17):

### What happened?

Adding informative features (A→B→C) consistently improved prediction for both Ridge
and Random Forest across all metrics. Test RMSE fell by $10,388 (Ridge) and $4,947 (RF)
from the 10-feature baseline to the full 81-feature set.

Adding irrelevant features (C→D) produced no meaningful change for Ridge (−$78, 0.27%)
but clearly degraded RF (+$849, 3.2%).

Adding redundant features (C→E) produced negligible change for both models
(+$15 Ridge, +$181 RF).

Adding both together (C→F) followed the irrelevant feature pattern: Ridge unaffected,
RF degraded (+$920).

### Why did it happen?

- **Informative features helped** because they carry genuine signal about SalePrice.
  More relevant information → better model.
- **Irrelevant features hurt RF** because RF's random node-splitting cannot distinguish
  useful from useless features — noise dilutes the split quality.
- **Irrelevant features didn't hurt Ridge** because L2 regularization shrinks noisy
  feature coefficients toward zero, effectively filtering them out.
- **Redundant features did nothing** because a feature perfectly correlated with
  an existing feature contains no new information. The model already has that signal.

### How confidently can we make this conclusion?

| Conclusion | Confidence | Reason |
|---|---|---|
| H1 (informative help) | **High** | Large effects (−7–12%), consistent across both models and all metrics |
| H2 (irrelevant hurt RF) | **High** | +$849 for RF is large and clearly directional |
| H2 (Ridge robust) | **Moderate** | −$78 is noise-level on one split; needs CV to confirm |
| H3 (redundant negligible) | **Moderate** | Direction consistent but magnitude small; needs CV |
| H4 (model dependence) | **High** | $78 vs $849 gap is too large to be random |

**Overall:** The main conclusion — more features only help when they are informative —
is supported with high confidence. The model-specific details require cross-validation
for full confirmation.

---

*Team 04 — 24SJPCCST503 Machine Learning —
St. Joseph's College of Engineering and Technology, Palai (Autonomous)*
