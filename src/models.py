"""
models.py
=========
MEMBER 3 — Model Behaviour
Branch: feature/member3-models

WHY Ridge(alpha=10)?
  - Ridge adds an L2 penalty (alpha * sum(w_i^2)) to the least-squares loss.
  - With 300+ one-hot encoded features from 2,344 training samples, un-regularized
    regression would produce unstable, over-fitted coefficients.
  - alpha=10 is a moderate value: shrinks noisy/irrelevant feature coefficients
    toward zero without over-penalising genuinely useful features.
  - Requires StandardScaler because L2 penalises all coefficients equally —
    features on different scales would be penalised unfairly without scaling.

WHY RandomForestRegressor(n_estimators=300, max_features='sqrt')?
  - 300 trees provide a stable, well-averaged ensemble; more trees reduce variance.
  - max_features='sqrt' is the standard setting: at each split, only sqrt(n_features)
    randomly-selected features are considered. This is the key mechanism for H4:
    when irrelevant features are present, they occupy slots in the random subset,
    occasionally becoming the chosen split — degrading tree quality.
  - Scale-invariant (tree thresholds do not depend on magnitude), so no scaling needed.
  - n_jobs=-1 uses all available CPU cores.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42


def make_models() -> dict:
    """
    Return fresh unfitted instances of Ridge and Random Forest.

    Returns
    -------
    dict
        {"Ridge": Ridge(alpha=10), "Random Forest": RandomForestRegressor(...)}
    """
    return {
        "Ridge": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_features="sqrt",   # explicitly set — matches explanation in notebook
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate_on_split(pipeline: Pipeline, X_train_exp, X_test_exp,
                      y_train, y_test) -> dict:
    """
    Fit pipeline on training data, then evaluate on both train and test sets.

    Parameters
    ----------
    pipeline    : fitted or unfitted sklearn Pipeline (preprocessor + model)
    X_train_exp : training feature DataFrame for this experiment
    X_test_exp  : test feature DataFrame for this experiment
    y_train     : training target Series
    y_test      : test target Series

    Returns
    -------
    dict with keys:
        Train_RMSE         — lower = better; measures training accuracy
        Test_RMSE          — lower = better; PRIMARY metric (generalisation)
        Test_MAE           — lower = better; less sensitive to outliers than RMSE
        Test_R2            — higher = better; proportion of variance explained (max 1.0)
        Generalization_Gap — Test_RMSE - Train_RMSE; diagnostic for overfitting
    """
    # Fit the pipeline (preprocessor + model) on training data only.
    # This prevents data leakage: the scaler/imputer never sees test rows during fit.
    pipeline.fit(X_train_exp, y_train)

    train_pred = pipeline.predict(X_train_exp)
    test_pred  = pipeline.predict(X_test_exp)

    train_rmse = float(np.sqrt(mean_squared_error(y_train, train_pred)))
    test_rmse  = float(np.sqrt(mean_squared_error(y_test,  test_pred)))
    test_mae   = float(mean_absolute_error(y_test, test_pred))
    test_r2    = float(r2_score(y_test, test_pred))

    return {
        "Train_RMSE":         train_rmse,
        "Test_RMSE":          test_rmse,
        "Test_MAE":           test_mae,
        "Test_R2":            test_r2,
        "Generalization_Gap": test_rmse - train_rmse,
    }
