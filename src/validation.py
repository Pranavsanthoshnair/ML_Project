"""
validation.py
=============
MEMBER 4 — Validation, Metrics & Statistical Analysis
Branch: feature/member4-validation

TODO for Member 4:
  - Implement make_kfold() — shared KFold(n_splits=5, shuffle=True, random_state=42)
  - Implement run_cross_validation() — 5-fold CV, same folds for ALL configs
  - Implement compute_cv_deltas() — per-fold ΔRMSE + mean ± SD + 95% CI
  - Implement cv_summary_table() — clean display table
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
    """
    Return the SHARED KFold splitter used for all configurations.
    SAME object → same fold splits for every config → paired comparison.
    """
    raise NotImplementedError("Member 4: implement this function")


def run_cross_validation(X_experiment, X_train, y_train, all_original_cols) -> pd.DataFrame:
    """
    5-fold CV for every config × model. Only uses training data (test untouched).

    Returns DataFrame with per-config × per-model CV summary:
    Config, Model, Raw_Features,
    CV_RMSE_Mean, CV_RMSE_SD, CV_MAE_Mean, CV_MAE_SD, CV_R2_Mean, CV_R2_SD,
    _fold_rmse, _fold_mae, _fold_r2  (lists, used for paired CI)
    """
    raise NotImplementedError("Member 4: implement this function")


def compute_cv_deltas(cv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-fold ΔRMSE for each comparison pair. Report mean ± SD and 95% CI.
    Use scipy.stats.t for CI (t-distribution, df=K-1).
    """
    raise NotImplementedError("Member 4: implement this function")


def cv_summary_table(cv_df: pd.DataFrame) -> pd.DataFrame:
    """Return clean display table without raw fold lists."""
    raise NotImplementedError("Member 4: implement this function")
