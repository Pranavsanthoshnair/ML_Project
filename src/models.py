"""
models.py
=========
MEMBER 3 — Model Behaviour
Branch: feature/member3-models

TODO for Member 3:
  - Implement make_models() — return Ridge(alpha=10) and RF(n=300, max_features='sqrt')
  - Implement evaluate_on_split() — fit pipeline, evaluate on train + test
  - Document WHY each model was chosen and WHY each hyperparameter was set
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42


def make_models() -> dict:
    """
    Return fresh unfitted instances of Ridge and Random Forest.

    Requirements (from handoff spec §17, §18):
    - Ridge: alpha=10
    - RandomForest: n_estimators=300, max_features='sqrt', random_state=42, n_jobs=-1
    - max_features MUST be explicitly set so code matches the explanation
    """
    raise NotImplementedError("Member 3: implement this function")


def evaluate_on_split(pipeline, X_train_exp, X_test_exp, y_train, y_test) -> dict:
    """
    Fit pipeline on train data. Evaluate on both train and test.

    Returns dict with keys:
    Train_RMSE, Test_RMSE, Test_MAE, Test_R2, Generalization_Gap
    (Gap = Test_RMSE - Train_RMSE — diagnostic only, not primary metric)
    """
    raise NotImplementedError("Member 3: implement this function")
