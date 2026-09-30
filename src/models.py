"""
models.py — Pranav S Nair
Branch: feature/pranav-models
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42


def make_models() -> dict:
    return {
        "Ridge": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate_on_split(pipeline: Pipeline, X_train_exp, X_test_exp,
                      y_train, y_test) -> dict:
    pipeline.fit(X_train_exp, y_train)

    train_pred = pipeline.predict(X_train_exp)
    test_pred  = pipeline.predict(X_test_exp)

    train_rmse = float(np.sqrt(mean_squared_error(y_train, train_pred)))
    test_rmse  = float(np.sqrt(mean_squared_error(y_test,  test_pred)))

    return {
        "Train_RMSE":         train_rmse,
        "Test_RMSE":          test_rmse,
        "Test_MAE":           float(mean_absolute_error(y_test, test_pred)),
        "Test_R2":            float(r2_score(y_test, test_pred)),
        "Generalization_Gap": test_rmse - train_rmse,
    }
