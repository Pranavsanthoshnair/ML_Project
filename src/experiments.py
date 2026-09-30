"""
experiments.py
==============
MEMBER 3 — Model Behaviour (experiment runner)
Branch: feature/member3-models

TODO for Member 3:
  - Implement run_all_experiments() — loop all configs × models on fixed split
  - Implement compute_pairwise_deltas() — ΔRMSE with M20 as baseline (NOT all-original)
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.preprocessing import create_preprocessor
from src.models import make_models, evaluate_on_split
from src.feature_sets import build_feature_configs

# Comparison pairs — M20 is the baseline (handoff spec §4 methodological correction)
COMPARISON_PAIRS = [
    ("M10",   "M20",     "M10 → M20: adding meaningful features (H1)"),
    ("M20",   "M20_I",   "M20 → M20+I: adding irrelevant features (H2)"),
    ("M20",   "M20_R",   "M20 → M20+R: adding redundant features (H3)"),
    ("M20",   "M20_I_R", "M20 → M20+I+R: adding both (H4)"),
]


def run_all_experiments(X_experiment, X_train, X_test, y_train, y_test, all_original_cols) -> pd.DataFrame:
    """
    Run every feature configuration through Ridge and RF on fixed 80/20 split.

    Returns DataFrame with columns:
    Config, Model, Raw_Features, Transformed_Features,
    Train_RMSE, Test_RMSE, Test_MAE, Test_R2, Generalization_Gap
    """
    raise NotImplementedError("Member 3: implement this function")


def compute_pairwise_deltas(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ΔRMSE/ΔMAE/ΔR² for each comparison pair, per model.
    IMPORTANT: M20 must be the baseline — not C_All_Original.
    """
    raise NotImplementedError("Member 3: implement this function")
