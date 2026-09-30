"""
validation.py  — COMPLETED IMPLEMENTATION
==========================================
MEMBER 4 — Validation, Metrics & Statistical Analysis
Branch: feature/Ananthan-validation

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocessing import create_preprocessor
from src.models import make_models
from src.feature_sets import build_feature_configs

RANDOM_STATE = 42
N_FOLDS      = 5

CV_COMPARISON_PAIRS = [
    ("M10",   "M20",     "M10 → M20 (H1: meaningful features)"),
    ("M20",   "M20_I",   "M20 → M20+I (H2: irrelevant)"),
    ("M20",   "M20_R",   "M20 → M20+R (H3: redundant)"),
    ("M20",   "M20_I_R", "M20 → M20+I+R (H4: both)"),
]


def make_kfold() -> KFold:
    """Shared KFold — same splits for all configs enabling paired comparison."""
    return KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def _neg_rmse(estimator, X, y):
    return -float(np.sqrt(mean_squared_error(y, estimator.predict(X))))

def _neg_mae(estimator, X, y):
    return -float(mean_absolute_error(y, estimator.predict(X)))

SCORERS = {"neg_rmse": _neg_rmse, "neg_mae": _neg_mae, "r2": "r2"}


def run_cross_validation(X_experiment, X_train, y_train, all_original_cols) -> pd.DataFrame:
    """5-fold CV for all configs x models. Same folds for all (paired comparison)."""
    feature_configs = build_feature_configs(all_original_cols)
    kf = make_kfold()
    records = []

    for config_name, features in feature_configs.items():
        X_tr = X_experiment.loc[X_train.index, features].copy()

        for model_name, estimator in make_models().items():
            scale = (model_name == "Ridge")
            preprocessor = create_preprocessor(X_tr, scale_numeric=scale)
            pipeline = Pipeline([("preprocessing", preprocessor), ("model", estimator)])

            cv_out = cross_validate(pipeline, X_tr, y_train, cv=kf,
                                    scoring=SCORERS, n_jobs=-1)

            fold_rmse = -cv_out["test_neg_rmse"]
            fold_mae  = -cv_out["test_neg_mae"]
            fold_r2   =  cv_out["test_r2"]

            records.append({
                "Config":       config_name,
                "Model":        model_name,
                "Raw_Features": len(features),
                "CV_RMSE_Mean": float(np.mean(fold_rmse)),
                "CV_RMSE_SD":   float(np.std(fold_rmse, ddof=1)),
                "CV_MAE_Mean":  float(np.mean(fold_mae)),
                "CV_MAE_SD":    float(np.std(fold_mae,  ddof=1)),
                "CV_R2_Mean":   float(np.mean(fold_r2)),
                "CV_R2_SD":     float(np.std(fold_r2,   ddof=1)),
                "_fold_rmse":   fold_rmse.tolist(),
                "_fold_mae":    fold_mae.tolist(),
                "_fold_r2":     fold_r2.tolist(),
            })
        print(f"  CV done: {config_name}")

    return pd.DataFrame(records)


def compute_cv_deltas(cv_df: pd.DataFrame) -> pd.DataFrame:
    """Per-fold ΔRMSE with mean +/- SD and 95% CI (t-distribution)."""
    from scipy import stats
    rows = []

    def get_folds(cfg, mdl):
        hit = cv_df[(cv_df["Config"] == cfg) & (cv_df["Model"] == mdl)]
        if hit.empty:
            raise KeyError(f"Config '{cfg}' / Model '{mdl}' not found in CV results")
        return np.array(hit.iloc[0]["_fold_rmse"])

    for base, modified, description in CV_COMPARISON_PAIRS:
        for model_name in cv_df["Model"].unique():
            deltas = get_folds(modified, model_name) - get_folds(base, model_name)
            k      = len(deltas)
            mean_d = float(np.mean(deltas))
            sd_d   = float(np.std(deltas, ddof=1))
            margin = float(stats.t.ppf(0.975, df=k-1)) * sd_d / np.sqrt(k)
            rows.append({
                "Model":       model_name,
                "Comparison":  f"{base} → {modified}",
                "Description": description,
                "Mean d RMSE": round(mean_d, 2),
                "SD d RMSE":   round(sd_d, 2),
                "CI Lower":    round(mean_d - margin, 2),
                "CI Upper":    round(mean_d + margin, 2),
            })

    return pd.DataFrame(rows)


def cv_summary_table(cv_df: pd.DataFrame) -> pd.DataFrame:
    """Clean display table without raw fold lists."""
    cols = ["Config","Model","Raw_Features",
            "CV_RMSE_Mean","CV_RMSE_SD","CV_MAE_Mean","CV_MAE_SD","CV_R2_Mean","CV_R2_SD"]
    return cv_df[cols].copy()
