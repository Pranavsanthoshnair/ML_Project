# Viva Preparation Guide — Team 04 ML Investigation

**Course:** 24SJPCCST503 – Machine Learning  
**Institution:** St. Joseph's College of Engineering and Technology, Palai (Autonomous)

---

## Section A: Core Understanding

**Q1. What is this project about?**  
This project is a controlled ML experiment that investigates whether adding more features to a machine learning model necessarily improves its prediction accuracy. We use house price prediction on the Ames Housing dataset as the domain, and compare six carefully designed feature configurations across two regression models — Ridge Regression and Random Forest.

**Q2. What is your research question?**  
"Does adding more features necessarily improve prediction?" The operational form is: "How does the *type* of added feature — informative, irrelevant, or redundant — affect predictive performance, generalization, stability, and model behaviour in house-price prediction?"

**Q3. Is this an application project or an investigation project?**  
It is an investigation project. The goal is not to build the best possible house-price predictor. The goal is to produce evidence about a general ML question — how feature type affects model performance. House prices are the vehicle for the experiment, not the end goal.

**Q4. What is the difference between informative, irrelevant, and redundant features?**  
- **Informative:** Has a real statistical relationship with the target. Example: `Overall Qual` is strongly correlated with `SalePrice`.  
- **Irrelevant:** Has no relationship with the target. In our experiment, these are random normal variables generated from np.random.RandomState(42) with no connection to house prices.  
- **Redundant:** Carries information already present in other features. In our experiment, redundant features are near-perfect copies of existing features with Pearson correlation r ≥ 0.990.

**Q5. What is your main finding?**  
No — adding more features does NOT necessarily improve prediction. Informative features helped both models. Irrelevant features hurt Random Forest (+$849 RMSE increase) but not Ridge. Redundant features provided negligible benefit for both models.

**Q6. Why is this question important in machine learning?**  
In practice, data scientists often have access to many features and must decide which to include. More features means more complexity, more computation, and more risk of overfitting. This investigation shows that the quality of features matters more than the quantity. A model with 10 carefully chosen informative features (RMSE ≈ $31,658 for RF) performs much better than a model with 81 features that include noise — understanding why guides real-world feature selection decisions.

**Q7. What was your methodology?**  
We followed the pipeline: Research Question → Hypotheses → Experimental Design → Data Preparation → Feature Engineering → Model Training → Evaluation → Comparison → Conclusion. All six experiments used the same dataset, same train/test split, same model hyperparameters, and same evaluation metrics. Only the feature set changed.

**Q8. What are the four hypotheses and what did the evidence show?**  
- H1 (informative helps): Supported — Ridge RMSE improved −$2,928; RF improved −$3,900 when going A→B.  
- H2 (irrelevant doesn't help): Supported — Ridge essentially unchanged (−$78, noise-level); RF degraded +$849 when going C→D.  
- H3 (redundant provides little benefit): Supported — Ridge +$15, RF +$181 when going C→E — both negligible.  
- H4 (model dependence): Supported — Ridge was robust to noise features; RF was more sensitive.

**Q9. Why did you choose Ridge and Random Forest specifically?**  
They represent two fundamentally different learning approaches: Ridge is a regularized linear model that makes assumptions about linearity and uses L2 penalty to handle correlated features. Random Forest is a non-linear tree ensemble that learns complex interactions through bagging and random feature subsets. Using both allows us to see whether the effect of feature type depends on the learning algorithm — and it does (H4).

**Q10. What does "controlled experiment" mean in this context?**  
It means we change only one variable (the feature configuration) between experiments, while keeping everything else fixed (same dataset, same split, same models, same hyperparameters, same evaluation). This way, any difference in results can be attributed to the feature change, not to other factors.

---

## Section B: Experimental Design

**Q1. How many experiments did you run?**  
Six feature configurations × two models = 12 model/experiment combinations. Each combination was evaluated using the same train/test split.

**Q2. Why did you use a fixed train/test split instead of cross-validation?**  
A fixed split ensures that all 12 combinations are evaluated on the *exact same test set*. This makes comparisons perfectly controlled — any difference in Test RMSE between, say, experiment A and experiment D, is due only to the different feature set, not to different test data. Cross-validation would be useful for measuring stability but would complicate the controlled comparison.

**Q3. What is the train/test split ratio and why?**  
80/20 — 2,344 training samples and 586 test samples. 80/20 is a common choice in ML: it gives the model enough training data to learn well, while reserving enough test data for a reliable performance estimate. With 2,930 total rows, this gives reasonable sample sizes for both sets.

**Q4. What is random_state=42 and why does it matter?**  
`random_state=42` is the seed for the random number generator that determines which rows go into the training set and which go into the test set. Setting it to 42 (or any fixed value) makes the split reproducible — every time the notebook is run, the exact same rows are in train and test. This is essential: if the split were random each run, results would differ and comparisons would be unreliable.

**Q5. What does "only the feature set changes" mean technically?**  
In the main experiment loop (Cell 15), the code uses `X_experiment.loc[X_train.index, features]` to extract training data, and `X_experiment.loc[X_test.index, features]` for test data. Using `.loc[X_train.index]` ensures that the same rows (as determined by the original train/test split) are used in every experiment. Only the columns (`features`) change.

**Q6. How did you ensure no data leakage?**  
All preprocessing (imputation, scaling, one-hot encoding) is performed inside a scikit-learn `Pipeline`. When `pipeline.fit()` is called on training data, the imputer learns medians from the training set only, the scaler learns mean/std from training set only, and the encoder learns categories from training set only. Test data is only used in `pipeline.predict()`, where these pre-learned transformations are applied.

**Q7. How did you construct the irrelevant features?**  
Using `np.random.RandomState(42).normal(0, 1, n_samples)` — standard normal samples with mean 0 and standard deviation 1. Ten such columns were created (`RandomFeature_1` through `RandomFeature_10`). They have no mathematical relationship to SalePrice and their expected Pearson correlation with SalePrice is 0.

**Q8. How did you construct the redundant features?**  
Each redundant feature is: `original_feature × multiplier + random_noise`. For example, `Redundant_GrLivArea = Gr Liv Area × 1.02 + N(0, 10)`. The multiplier is close to 1.0 (0.99–1.02) and the noise standard deviation is tiny relative to the feature's range. The result has Pearson correlation r ≥ 0.990 with the original, confirming it is genuinely redundant.

**Q9. Why did you verify redundancy using Pearson correlation?**  
To provide evidence, not assumption. Instead of saying "these features are redundant because we constructed them that way," we measured the actual Pearson correlation between each redundant feature and its original. Correlations of 1.000, 0.990, 1.000, 1.000, and 0.998 provide objective confirmation that the features are indeed highly redundant.

**Q10. Why did you use 10 irrelevant features and 5 redundant features?**  
10 irrelevant features provides a meaningful but not overwhelming addition to the 81 original features (~12% increase in raw count). 5 redundant features covers the most important informative features (the ones with strongest signal) to test whether their near-duplication hurts. The exact numbers are somewhat arbitrary — the key is that the addition is large enough to potentially affect results, while being small enough that the comparison remains interpretable.

---

## Section C: Dataset Questions

**Q1. What is the Ames Housing dataset?**  
A real-world dataset of residential house sales in Ames, Iowa, USA, covering 2006–2010. Published by Dean De Cock in 2011 as an alternative to the older Boston Housing dataset. It contains 2,930 rows (one per sale) and 82 columns including the target `SalePrice` in US dollars.

**Q2. What is the target variable and what does it represent?**  
`SalePrice` — the sale price of the house in US dollars. It ranges from approximately $13,000 to $755,000 with a median around $163,000. It is a continuous numerical variable, making this a regression (not classification) problem.

**Q3. What types of features does the dataset have?**  
Both numerical and categorical. Numerical features include size measurements (sq ft), year values, and room counts. Categorical features include quality ratings (Ex/Gd/TA/Fa/Po), neighborhood names (28 unique values), roof types, foundation types, and many others.

**Q4. Why does the dataset have so many missing values?**  
Most missing values represent the *absence* of a feature, not a data entry error. For example, `Pool QC` (pool quality) is missing for 2,917 of 2,930 houses because those houses don't have a pool. Similarly, `Alley` is missing for 2,732 houses because they don't have alley access. The preprocessing pipeline handles these using median imputation for numerical features and most-frequent imputation for categorical features.

**Q5. Why is this dataset good for your investigation?**  
It has enough rows (2,930) for reliable comparisons, it has a realistic mix of feature types making preprocessing non-trivial, its features have genuine real-world meaning (unlike synthetic datasets), and it is well-studied so performance benchmarks exist for reference.

**Q6. What are the limitations of using this dataset?**  
Geographic limitation (Ames, Iowa only), temporal limitation (2006–2010 only), and the artificially constructed irrelevant/redundant features are not "real" uninformative features — they are synthetic. Real-world uninformative features often have weak correlations with other features, which can produce different effects than pure random noise.

---

## Section D: Feature Questions

**Q1. Why did you choose these specific 10 core features?**  
They are among the strongest predictors of SalePrice based on domain knowledge and correlation analysis. `Overall Qual` is a 1–10 quality rating assigned by an assessor. `Gr Liv Area` is the total above-ground living area. `Garage Cars` indicates the garage's capacity. These features capture the most important aspects of a house's value: quality, size, and condition.

**Q2. What happens after one-hot encoding of the full feature set?**  
81 raw features become 302 transformed features. This is because categorical columns with many unique values expand into many binary columns. For example, `Neighborhood` with 28 unique values becomes 28 binary columns. The jump from raw to transformed count is approximately 3.7× for experiment C.

**Q3. Why are experiments A and B not affected by one-hot encoding expansion?**  
Because all 10 core features and all 20 expanded features are numerical (integers or floats). None of them are categorical strings. Therefore, one-hot encoding adds no extra columns — raw count equals transformed count (10 and 20 respectively).

**Q4. If a redundant feature has r = 1.000 with its original, does it add literally zero information?**  
In theory, yes — r = 1.000 means the features are perfectly linearly related (one is an exact linear transformation of the other). In practice, the correlation is computed from 2,930 samples and may round to 1.000 at 3 decimal places even if there is a tiny difference. The noise added during construction (e.g., N(0, 10) for Redundant_GrLivArea) is real — the feature is not literally identical — but the noise is so small relative to the feature range that the correlation rounds to 1.000.

**Q5. What is the difference between redundant features in this investigation and multicollinearity in real datasets?**  
Our redundant features are deliberately engineered to have near-perfect correlations (0.990–1.000) with their originals. In real datasets, multicollinearity is less extreme — features like `1st Flr SF` and `Gr Liv Area` might have correlations of 0.7–0.9. The effects would be similar in direction but less extreme in magnitude.

**Q6. Could a feature selection algorithm have found better features than your core 10?**  
Possibly. The core 10 were selected using domain knowledge and expected correlation with SalePrice. An automated method like Recursive Feature Elimination with Ridge or feature importance from Random Forest might produce a slightly different set of 10. However, this investigation is not about finding optimal features — it uses specific, interpretable feature sets to control the experiment cleanly.

**Q7. Why do irrelevant features sometimes show tiny improvements (Ridge C→D: Δ = −78)?**  
This is a noise-level artifact of a single train/test split. An RMSE difference of $78 on a baseline of $29,161 is only 0.27%. Different random splits would likely produce different signs for this tiny effect. It should not be interpreted as "irrelevant features help Ridge" — the correct interpretation is "Ridge was robust to irrelevant features."

**Q8. What would happen if you added 100 irrelevant features instead of 10?**  
The effect would likely be larger for both models, but especially for Random Forest. With 100 irrelevant features added to 81 originals, the probability that any given tree node selects an irrelevant feature as the best split increases significantly, leading to more feature dilution and likely a more noticeable performance degradation. Ridge would still largely absorb them through regularization, though with 100 irrelevant features the regularization might face more pressure.

---

## Section E: Model Questions

**Q1. What is Ridge Regression and why does it have an alpha parameter?**  
Ridge Regression is linear regression with an added L2 regularization penalty. The loss function is: Σ(y − ŷ)² + α × Σ(β²). The alpha parameter controls how strongly coefficients are penalized. Higher alpha → coefficients shrink more toward zero → less overfitting but potentially more bias. Alpha = 10.0 in this investigation was chosen to provide meaningful regularization on the 302-feature transformed space.

**Q2. What is Random Forest and how does it differ from a single decision tree?**  
Random Forest builds many decision trees (300 in this investigation), each trained on a random bootstrap sample of the training data and using a random subset of features at each split. Predictions are the average of all 300 trees. A single tree would overfit badly — the ensemble of 300 trees smooths out individual tree errors through averaging, producing much better generalization.

**Q3. Why does Random Forest have a much larger generalization gap than Ridge?**  
Random Forest with 300 trees is a very high-capacity model. It can memorize the training data almost perfectly — Train RMSE of approximately $9,846 to $10,781. This produces a large gap to the Test RMSE of approximately $26,711 to $31,658. Ridge's lower capacity (linear model) means it cannot memorize training data to the same extent — Train RMSE of $22,905 to $35,289. Ridge's gap is smaller (3,154–6,221) because it cannot overfit as aggressively.

**Q4. Why did Random Forest outperform Ridge on Test RMSE despite having a larger generalization gap?**  
RF learns non-linear relationships between features and SalePrice that Ridge cannot capture with a linear model. For example, the interaction between `Neighborhood` and `Overall Qual` is non-linear. RF's capacity to capture these complex patterns gives it an advantage on test data even though it overfits training data more. The key insight: high capacity with some overfitting can still beat low capacity with less overfitting, if the signal is strong enough.

**Q5. What does n_estimators=300 mean for Random Forest?**  
It means the ensemble consists of 300 individual decision trees. More trees generally produce more stable and accurate predictions (up to a point), but also require more computation time. 300 is a reasonable choice for this investigation — enough trees for stable predictions without being computationally excessive.

**Q6. What is n_jobs=-1 in RandomForestRegressor?**  
It tells scikit-learn to use all available CPU cores when training the 300 trees in parallel. Since the 300 trees are independent of each other, they can be trained simultaneously, significantly speeding up training.

**Q7. Why use StandardScaler for Ridge but not for Random Forest?**  
Ridge regression is a linear model with L2 regularization. The L2 penalty is sensitive to the scale of features — a feature measured in thousands (like `SalePrice`) would dominate a feature measured in units (like `Full Bath`) without scaling. StandardScaler standardizes all features to mean=0, std=1, making the regularization apply evenly. Random Forest uses decision tree splits that compare feature values within each feature's own scale — it doesn't matter whether `Gr Liv Area` is in square feet (1,000s) or scaled (0–3). Tree splits are scale-invariant.

**Q8. Could you have used other models, like Gradient Boosting or Neural Networks?**  
Yes, and that would be an interesting extension. Gradient Boosting (e.g., XGBoost) would likely show behavior similar to Random Forest — sensitivity to irrelevant features, though potentially less so due to its regularized tree-building. Neural networks would likely be more sensitive to irrelevant features in a small dataset (2,930 rows), as they have very high capacity. The choice of Ridge and RF provides a clear contrast between regularized linear and non-linear ensemble approaches.

---

## Section F: Metric Questions

**Q1. What is RMSE and why is it the primary metric here?**  
Root Mean Squared Error = √(mean((y_actual − y_predicted)²)). It represents the typical magnitude of prediction errors in the same units as the target (here, USD). It is the primary metric because it penalizes large errors more than small ones (due to squaring), which matters for house prices — being off by $100,000 on a prediction is much worse than being off by $10,000. For this investigation, a Test RMSE of $26,711 means the model's typical prediction error is about ±$26,711.

**Q2. What is MAE and how does it differ from RMSE?**  
Mean Absolute Error = mean(|y_actual − y_predicted|). It takes the average of absolute (not squared) errors. It is less sensitive to large outliers than RMSE. If a few houses have very unusual sale prices, RMSE would be pulled up more than MAE. In this investigation, MAE (approximately $15,867 to $24,842) is always lower than RMSE because RMSE gives extra weight to the larger errors.

**Q3. What does R² = 0.911 mean?**  
An R² of 0.911 means the model explains 91.1% of the variance in SalePrice. The remaining 8.9% is unexplained (attributed to factors the model doesn't capture). R² ranges from 0 (model explains nothing — equivalent to just predicting the mean price) to 1.0 (perfect prediction). R² = 0.911 is a good result for a real-world regression problem.

**Q4. Can R² be negative?**  
Yes, in theory. R² is negative when the model performs worse than simply predicting the mean for all samples. This happens if the model is completely wrong. In this investigation, all R² values are between 0.805 and 0.911 — all well above zero.

**Q5. What is the Generalization Gap and what does it measure?**  
Generalization Gap = Test RMSE − Train RMSE. It measures how much worse the model is on unseen data compared to training data. A large gap indicates overfitting: the model learned the training data too specifically and doesn't transfer well to new data. Example: RF in experiment A has Train RMSE = $10,781 and Test RMSE = $31,658, giving a gap of $20,877. This large gap shows that RF memorizes training patterns more than Ridge (which has a gap of only $4,260 in the same experiment).

**Q6. If RMSE is lower for experiment B than A, does that definitely prove informative features help?**  
For this specific train/test split — yes, we can say that informative features improved performance on this evaluation. However, to be completely confident, we would want cross-validation (multiple different splits) to confirm the effect is consistent and not just lucky for this particular split. The large magnitude of the improvement (−$2,928 for Ridge, −$3,900 for RF) makes it very likely to be a genuine effect.

**Q7. Why do we look at multiple metrics (RMSE, MAE, R², Gen Gap) instead of just one?**  
Each metric captures a different aspect of performance. RMSE tells us the typical error size with emphasis on outliers. MAE tells us the average error without outlier emphasis. R² tells us how well the model explains variance. Generalization Gap tells us about overfitting. A model that looks great on one metric might have issues revealed by another. For example, a model with very low Train RMSE but very high Generalization Gap is overfitting — you wouldn't see this from Test RMSE alone.

**Q8. In the comparison table, what does Δ Test RMSE = −2927.559 mean (Ridge A→B)?**  
It means that by adding the 10 additional informative features (going from experiment A to B), Ridge's Test RMSE decreased by $2,927.56. Negative Δ Test RMSE = improvement (lower RMSE is better). The formula is: Δ = (Test RMSE in B) − (Test RMSE in A) = 36,621.425 − 39,548.984 = −2,927.559.

---

## Section G: Graph Interpretation

### Figure 1: Feature Count vs Test RMSE

**Q: What does the line plot of Feature Count vs Test RMSE show?**  
The x-axis shows the number of raw features (10, 20, 81, 86, 91, 96) and the y-axis shows Test RMSE. Both Ridge and Random Forest lines drop steeply from left to right as we go from 10 to 81 features — this is the effect of adding informative features (experiments A, B, C). After 81 features, the lines flatten out or rise slightly for experiments D (91), E (86), F (96) — this shows that irrelevant and redundant features don't extend the improvement.

**Q: What is the main conclusion from this figure?**  
More features reduce error only when they are informative. After reaching the full set of genuine features (C, 81 features), adding noise or redundant information provides no further benefit. The figure directly visualizes the answer to the research question.

### Figure 2: Configuration vs R²

**Q: What does the R² bar chart show?**  
Grouped bars for each of the 6 experiments, with one bar per model. The bars are tallest for C, D, E, F (high R² around 0.894–0.911) and shorter for A and B (0.805–0.904). Random Forest bars are consistently slightly taller than Ridge bars.

**Q: Why are the D, E, F bars not taller than C?**  
Because the additional features in D, E, F (irrelevant and redundant) don't explain any new variance in SalePrice. They add noise or duplicate existing information, so R² stays essentially the same or very slightly decreases.

### Figure 3: Train vs Test RMSE

**Q: What is the key visual pattern in the Train vs Test RMSE chart?**  
For Random Forest, there is a large gap between the Train RMSE bar (very short, ~$9,800–$10,200) and the Test RMSE bar (much taller, ~$26,700–$31,700). For Ridge, the gap is much smaller — Train RMSE (~$22,900–$35,300) and Test RMSE (~$29,100–$39,500) are closer together.

**Q: What does this large gap tell us about Random Forest?**  
It confirms that Random Forest overfits the training data. The model memorizes training patterns very well (low Train RMSE) but doesn't transfer them perfectly to new data (higher Test RMSE). The large generalization gap is an inherent property of high-capacity ensemble models trained on moderate-sized datasets.

### Figure 4: Redundancy Correlation Heatmap

**Q: What does the heatmap show?**  
It shows Pearson correlations between the 5 original features (Gr Liv Area, Overall Qual, Garage Area, Total Bsmt SF, Year Built) and their 5 redundant counterparts. The off-diagonal cells between an original and its redundant version show correlations of 0.99–1.00 (displayed as deep red in the coolwarm colormap).

**Q: How does this heatmap support H3?**  
It provides visual and quantitative evidence that the redundant features are genuinely redundant — they have near-perfect linear relationships with their originals. This confirms that any model already using the original feature is gaining essentially no new information by also including the redundant version, which explains why H3 is supported (negligible performance change).

---

## Section H: What-if Questions

**Q1. What if you had used only 1 feature?**  
Performance would be much worse. The single best feature is probably `Overall Qual` or `Gr Liv Area`. A single feature cannot capture the complexity of house pricing — many factors (size, quality, location, age, condition) all matter. RMSE would likely be well above $50,000.

**Q2. What if you had added 100 irrelevant features instead of 10?**  
Ridge would still be largely robust due to L2 regularization, but might show a small degradation with 100 noisy features. Random Forest would likely degrade more substantially — with 100 irrelevant features out of 181 total (~55%), the probability of selecting an irrelevant feature at each tree node becomes very high, causing significant feature dilution. The degradation for RF might be $2,000–$5,000 in Test RMSE based on the trend we observe.

**Q3. What if you had used a higher alpha for Ridge (e.g., alpha=100)?**  
Stronger regularization would make Ridge even more robust to irrelevant and redundant features, but might slightly hurt performance on informative features by over-shrinking their coefficients. The C→D delta for Ridge would likely be even closer to zero. There could be a small performance cost on the main experiments (A through F Test RMSE might increase slightly).

**Q4. What if you had used cross-validation instead of a single train/test split?**  
The main directional conclusions would likely remain the same. The large effects (A→B: −$2,928 Ridge, −$3,900 RF) would still be clearly significant. The small effects (C→D Ridge: −$78) would be better resolved — cross-validation would show whether the −$78 effect is consistent across folds or just a single-split artifact. CV would also provide a standard deviation, showing how stable results are across different data subsets.

**Q5. What if you had not used pipelines and instead scaled the data before splitting?**  
This would introduce data leakage. The scaler would have computed mean and standard deviation from the entire dataset (including test rows), so the test set's distribution would have influenced the scaling. Test performance would appear better than it truly is, because the model effectively "saw" the test data's statistics during preprocessing. This is a common mistake in ML projects.

**Q6. What if SalePrice was categorical (e.g., "cheap", "medium", "expensive")?**  
Then this would be a classification problem, not regression. We would use different models (logistic regression, random forest classifier) and different metrics (accuracy, F1-score, confusion matrix) instead of RMSE, MAE, and R². The experimental design (testing feature configurations) could remain the same.

**Q7. What if you used a neural network instead of Ridge/RF?**  
Neural networks are high-capacity, non-linear models. With only 2,930 samples and no cross-validation regularization, a neural network might overfit badly — the generalization gap could be very large. With proper regularization (dropout, weight decay), it might approach RF performance. Neural networks are generally more sensitive to irrelevant features than Ridge but might handle redundancy well through weight sharing. However, their behavior is less interpretable.

**Q8. What if all 81 original features were informative (no noise features)?**  
We would expect performance to keep improving as we add more features — Test RMSE would continue declining from A to B to C and beyond. This is actually what happens from A (10) to C (81) — each genuine feature adds signal. The plateau/degradation we see at D, E, F is specifically because those added features are not informative.

**Q9. What if you added features that are correlated with SalePrice but also correlated with each other (real-world multicollinearity)?**  
Ridge handles multicollinearity well — L2 regularization distributes weights among correlated predictors. Random Forest also handles it reasonably, since the random feature subsetting at each split means no single feature dominates. However, with very high multicollinearity (r > 0.95), both models would show effects similar to our redundant features — adding more correlated features would provide diminishing returns.

**Q10. What if the Ames Housing dataset had 100,000 rows instead of 2,930?**  
With more data, all models would generalize better. The generalization gap for Random Forest would likely decrease substantially. Models could handle more features without overfitting. The effect of adding 10 irrelevant features to 81 would be even smaller proportionally, because the model would have enough data to more reliably identify which features are truly predictive.

**Q11. What if you measured performance using only R² instead of RMSE?**  
The directional conclusions would be the same — R² and RMSE are highly correlated (one increases when the other decreases). However, R² changes are in absolute units (0.000–0.029 in this investigation) while RMSE changes are in dollars. For practical interpretation, RMSE in dollars is more intuitive: "the model's average error is $26,711" is more meaningful than "R² = 0.911." RMSE also differentiates performance differences more clearly at the high-performance end.

**Q12. What if you had used median absolute error instead of RMSE as the primary metric?**  
Median absolute error is even more robust to outliers than MAE. For a dataset like Ames Housing with some very high-priced luxury homes, median absolute error might give a better picture of typical prediction accuracy. RMSE was chosen as primary because it is the standard in regression literature and because it penalizes large errors, which is important when mispredicting a $500,000 house by $100,000.

---

## Section I: Critical Thinking Questions

**Q1. Is it possible that your conclusions are wrong?**  
Yes, with some probability. The conclusions are based on a single 80/20 train/test split. Small effects like Ridge C→D (Δ = −$78) could be reversed with a different split. The large effects (A→B Δ = −$2,928 for Ridge) are much more likely to be genuine. To strengthen the conclusions, cross-validation would be needed to confirm that results are consistent across multiple splits.

**Q2. Why might your experiment understate the harm of irrelevant features?**  
Because our irrelevant features are pure random normal noise — the most obvious form of irrelevance. In real datasets, "irrelevant" features often have weak correlations with the target (e.g., r = 0.05) or correlations with other features. These weaker forms of irrelevance might actually cause more harm because models may assign small but non-zero weights to them, rather than the near-zero weights they assign to pure noise.

**Q3. What is the difference between a limitation and a flaw?**  
A limitation is a known constraint of the investigation that reduces generalizability but does not invalidate the conclusions. For example, using only Ames Housing data is a limitation — results may not apply to London housing. A flaw would be something that makes the conclusions invalid — like data leakage, which would mean all reported RMSE values are artificially optimistic. This investigation has limitations (single dataset, single split) but not fundamental flaws in methodology.

**Q4. Why is the small Ridge improvement with irrelevant features (C→D: −$78) particularly important to flag?**  
Because a naive reader might interpret it as "adding irrelevant features improved Ridge" and draw the wrong conclusion. Being transparent about this counter-intuitive result and explaining why it is likely a noise artifact (0.27% change on a single split) demonstrates critical thinking. It shows the team understands the difference between statistically significant effects and noise.

**Q5. Could a different preprocessing choice affect your conclusions?**  
Yes. For example, using mean imputation instead of median imputation might produce different results for features with many outliers (like `SalePrice`-adjacent numerical features). Using drop strategy for missing values instead of imputation would reduce the dataset size and potentially change the train/test composition. However, the directional conclusions (informative helps, irrelevant hurts RF, redundant provides little benefit) would likely remain stable across reasonable preprocessing choices.

**Q6. What would you change if you repeated this investigation?**  
(1) Add 5-fold cross-validation to all experiments to measure stability. (2) Test multiple random seeds for generating irrelevant and redundant features to confirm results don't depend on the specific seed. (3) Include gradient boosting as a third model. (4) Test features with weak informative value (r ≈ 0.1–0.3 with SalePrice) to study the grey zone between informative and irrelevant.

**Q7. Why is it important that you formulated hypotheses *before* seeing the results?**  
Pre-formulated hypotheses prevent confirmation bias — the tendency to interpret results as supporting whatever you already expected. If you write hypotheses after seeing results, you can always construct a hypothesis that matches the data, which is not genuine scientific testing. The team's hypotheses were formulated based on ML theory (what Ridge regularization does to noisy features, what RF feature selection does) before running the experiment.

**Q8. How would you explain this project to someone who doesn't know ML?**  
Imagine you're trying to predict a house's sale price. You could use 10 important facts about the house: overall quality, living area size, number of garage spaces, etc. Or you could use 81 facts. Or you could add 10 completely random numbers that have nothing to do with the house. Or add 5 near-copies of facts you already have. The question is: does adding more facts always help? Our answer: adding useful facts helps (going from 10 to 81 improved accuracy by ~$10,000). Adding random garbage doesn't help and can hurt some algorithms. Adding near-duplicate facts provides almost no benefit.

---

## Section J: Defense Questions

**Q1. Why didn't you use the Boston Housing dataset, which is more commonly known?**  
**Short viva answer:** The Ames Housing dataset is better suited to our investigation because it has 2,930 samples (vs Boston's 506), a much wider range of feature types (81 features including many categorical ones), and is considered more representative of real-world housing data. The larger size gives more stable comparisons across 12 model/experiment combinations.

**Detailed explanation:** The Boston Housing dataset has only 506 rows — too few for reliable comparison across 12 combinations. It also has only 13 features, leaving little room to study the effect of adding different types of features. Ames Housing's 82 columns (before feature engineering) provide a natural "all original features" baseline that can be expanded with irrelevant/redundant features.

**Key terms:** sample size, statistical stability, feature richness.

**Q2. How do you know your redundant features are actually redundant?**  
**Short viva answer:** We measured Pearson correlation between each redundant feature and its original. Correlations of 1.000, 0.990, 1.000, 1.000, and 0.998 confirm that the features are near-perfect copies. This provides objective evidence of redundancy, not just an assumption based on how they were constructed.

**Detailed explanation:** The construction method (original × multiplier + small noise) guarantees near-perfect correlation by design. The Pearson correlation measurement (Cell 10 in the notebook) verifies this empirically. A correlation of 1.000 means the two features move together perfectly — knowing one tells you exactly what the other is.

**Key terms:** Pearson correlation, linear dependence, multicollinearity.

**Q3. Your Ridge model showed a slight improvement with irrelevant features (−$78). Doesn't this contradict H2?**  
**Short viva answer:** No. An $78 improvement on a baseline of $29,161 is a 0.27% change — well within the noise range of a single train/test split. H2 states that irrelevant features will "not systematically improve performance." A 0.27% change in one direction on one random split is not systematic improvement. The correct interpretation is that Ridge was robust to irrelevant features (no meaningful change).

**Detailed explanation:** The sign of this tiny effect could easily be reversed with a different train/test split seed. If we ran 10 different splits, the Ridge C→D delta would likely sometimes be +$78 and sometimes −$78, averaging near zero. Statistical significance requires consistency across multiple evaluations, not a single split. The large RF effect (+$849) is much more likely to be genuine.

**Key terms:** noise artifact, single-split evaluation, statistical significance, null effect.

**Q4. What is the practical significance of your findings for a real ML practitioner?**  
**Short viva answer:** The investigation provides direct guidance for feature engineering: curate your features carefully rather than including everything available. Adding all features from a dataset is not always the best strategy. Informative features reliably help. Irrelevant features can hurt tree-based models. Redundant features waste computational resources for no gain.

**Detailed explanation:** In practice, datasets often have many candidate features of uncertain utility. The instinct to "include everything and let the model sort it out" is tested here. Our results show that for tree models like RF, including genuine noise features (D) degrades performance by ~3%. For large-scale applications with hundreds of uninformative features, the degradation could be much larger. Ridge's regularization provides a safety net, but the best approach is always to clean the feature set first.

**Key terms:** feature engineering, model robustness, regularization, computational efficiency.

**Q5. What is the significance of the Transformed_Features column (e.g., 302 for experiment C)?**  
**Short viva answer:** When categorical features are one-hot encoded, each unique value becomes a separate binary column. The original 81 features contain many categorical columns (Neighborhood, Sale Type, etc.), which expand to 302 features after encoding. This is the actual dimensionality that the Ridge model estimates coefficients for — 302 parameters from 2,344 training samples.

**Detailed explanation:** This matters because it demonstrates why regularization is important. With 302 features and 2,344 samples, the feature-to-sample ratio is about 1:7.8 — manageable but not trivially small. Without Ridge's regularization, estimating 302 coefficients from 2,344 samples with many correlated features would likely produce unstable, overfitted coefficients. The regularization (alpha=10) is chosen to address this.

**Key terms:** one-hot encoding, dimensionality, regularization, feature-to-sample ratio.

**Q6. Why does the generalization gap increase from B (3,154) to C (6,210) for Ridge, even though performance improved?**  
**Short viva answer:** Experiment C adds all 81 original features, many of which are weak predictors or categorical with many values. Ridge can fit the training data better with more features (reducing Train RMSE from $33,467 to $22,951), but the generalization to test data doesn't improve proportionally, widening the gap. The model finds more patterns in training data, some of which are training-specific.

**Detailed explanation:** Going from B (20 features) to C (81 features) improves Test RMSE from $36,621 to $29,161 — a genuine improvement. But Train RMSE drops from $33,467 to $22,951 — an even larger drop. The gap between train and test grows because the additional 61 features in C include many weak predictors and high-cardinality categoricals that allow the model to fit training data more specifically. This is a mild form of overfitting — the model extracts some real signal but also some noise from the extra features.

**Key terms:** overfitting, generalization, train/test gap, weak predictors.

**Q7. What is the academic contribution of your investigation?**  
**Short viva answer:** The investigation produces clean, reproducible experimental evidence that feature type matters more than feature count. It confirms established ML theory (redundancy → no benefit, noise → potential harm for RF, regularization → robustness for Ridge) using real data, controlled conditions, and transparent methodology. The experimental framework (6 configurations, 2 models, fixed split) is a template that could be extended to other datasets or model types.

**Detailed explanation:** While the findings are consistent with known ML theory, the value lies in the experimental confirmation with real data under controlled conditions. The investigation also reveals the model dependence of the effect (H4), which is a nuanced finding: the same additional features hurt RF but not Ridge. This model-dependence is an important practical consideration often glossed over in introductory ML curricula.

**Key terms:** controlled experiment, reproducibility, feature type, model dependence.

**Q8. If you had to summarize your finding in one sentence, what would it say?**  
"Adding more features improves prediction only when those features are genuinely informative — irrelevant features hurt tree-based models and provide no benefit to regularized linear models, while redundant features provide negligible additional value regardless of model type."

**Q9. What is data leakage and how did you prevent it?**  
**Short viva answer:** Data leakage occurs when information from the test set influences the training process, producing artificially optimistic performance estimates. We prevented it by placing all preprocessing steps (imputation, scaling, encoding) inside scikit-learn Pipeline objects. The pipeline fits transformers on training data only and applies the same transformations to test data without re-fitting.

**Detailed explanation:** A common mistake is to scale the entire dataset before splitting. If you compute the mean and std of `Gr Liv Area` from all 2,930 rows, you've used test set statistics in your scaling. When you then split and train on the scaled training set, the model has indirectly "seen" information about the test set distribution. This inflates test performance artificially. Pipeline ensures that the scaler is `fit()` only on `X_train` and `transform()` applied to both `X_train` and `X_test`.

**Key terms:** data leakage, pipeline, cross-contamination, train/test separation.

**Q10. How would you improve this investigation if you had more time?**  
**Short viva answer:** I would add 5-fold cross-validation to all experiments to confirm result stability across different splits. I would also test features with weak-but-non-zero correlations with SalePrice (the grey zone between informative and irrelevant) to understand the threshold at which features become worth including. A third improvement would be to test feature importance rankings from RF to see which of the 81 original features the model finds most valuable.

**Detailed explanation:** These extensions would address the main limitation of the current investigation (single train/test split) and add richer insight into the informative-irrelevant spectrum. Testing multiple seeds for irrelevant/redundant feature generation would also confirm that the specific random realization doesn't affect the conclusions.

**Key terms:** cross-validation, stability, feature importance, experimental extension.

---

*End of Viva Preparation Guide*
