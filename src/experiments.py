"""
experiments.py
==============
MEMBER 3 — Model Behaviour (experiment runner)
Branch: feature/member3-models

Design notes
------------
- run_all_experiments() iterates over every feature configuration × every model.
- The SAME X_train / X_test indices are used for every combination so comparisons
  are controlled: only the feature set changes, not the train/test sample.
- compute_pairwise_deltas() uses M20 as the baseline (NOT M10 or M_All_Original).
  M20 is the full 20-meaningful-feature set — it is the "clean" reference point
  against which we measure the effect of adding irrelevant (H2) or redundant (H3)
  features.  See COMPARISON_PAIRS below.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.preprocessing import create_preprocessor, build_pipeline
from src.models import make_models, evaluate_on_split
from src.feature_sets import build_feature_configs

# ---------------------------------------------------------------------------
# Comparison pairs
# M20 is the baseline for H2, H3, H4 (not M_All_Original).
# M10 → M20 tests H1: does adding meaningful features help?
# ---------------------------------------------------------------------------
COMPARISON_PAIRS = [
    ("M10",   "M20",     "M10 → M20: adding meaningful features (H1)"),
    ("M20",   "M20_I",   "M20 → M20+I: adding irrelevant features (H2)"),
    ("M20",   "M20_R",   "M20 → M20+R: adding redundant features (H3)"),
    ("M20",   "M20_I_R", "M20 → M20+I+R: adding both (H4)"),
]


def run_all_experiments(
    X_experiment: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    all_original_cols: list,
) -> pd.DataFrame:
    """
    Run every feature configuration through Ridge and RF on the fixed 80/20 split.

    Parameters
    ----------
    X_experiment     : full DataFrame that includes all synthetic columns
    X_train          : training rows of X (used for .index only, NOT features)
    X_test           : test rows of X (used for .index only)
    y_train          : training target
    y_test           : test target
    all_original_cols: list of original column names (no synthetic ones)

    Returns
    -------
    pd.DataFrame with columns:
        Config, Model, Raw_Features, Transformed_Features,
        Train_RMSE, Test_RMSE, Test_MAE, Test_R2, Generalization_Gap
    """
    # Build the 6 feature configuration lists
    feature_configs = build_feature_configs(all_original_cols)

    results = []

    for config_name, feature_list in feature_configs.items():

        # Slice the correct columns from X_experiment,
        # using the SAME train/test row indices for every config.
        X_train_exp = X_experiment.loc[X_train.index, feature_list].copy()
        X_test_exp  = X_experiment.loc[X_test.index,  feature_list].copy()

        for model_name, estimator in make_models().items():

            # Ridge needs StandardScaler; Random Forest does not (scale-invariant).
            scale = (model_name == "Ridge")

            # Build a fresh pipeline: preprocessor + this model.
            pipeline = build_pipeline(estimator, X_train_exp, scale_numeric=scale)

            # Fit + evaluate on the fixed split.
            metrics = evaluate_on_split(
                pipeline, X_train_exp, X_test_exp, y_train, y_test
            )

            # Count transformed features (after one-hot encoding).
            fitted_prep = pipeline.named_steps["preprocessor"]
            try:
                transformed_count = len(fitted_prep.get_feature_names_out())
            except AttributeError:
                transformed_count = -1   # older sklearn versions

            results.append({
                "Config":               config_name,
                "Model":                model_name,
                "Raw_Features":         len(feature_list),
                "Transformed_Features": transformed_count,
                **metrics,
            })

    results_df = pd.DataFrame(results)

    # Sort for readability: Ridge first then RF, in config order.
    config_order = list(feature_configs.keys())
    results_df["Config"] = pd.Categorical(
        results_df["Config"], categories=config_order, ordered=True
    )
    results_df = results_df.sort_values(["Model", "Config"]).reset_index(drop=True)

    return results_df


def compute_pairwise_deltas(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute delta metrics for each comparison pair, per model.

    Delta = modified_value - baseline_value
    Negative delta for RMSE/MAE = improvement.
    Positive delta for R2 = improvement.

    Parameters
    ----------
    results_df : output from run_all_experiments()

    Returns
    -------
    pd.DataFrame with columns:
        Model, Comparison, Description,
        Delta_Test_RMSE, Delta_Test_MAE, Delta_Test_R2, Delta_Gen_Gap
    """
    rows = []

    for base_config, modified_config, description in COMPARISON_PAIRS:
        for model_name in results_df["Model"].unique():

            mask_base     = (results_df["Config"] == base_config)     & (results_df["Model"] == model_name)
            mask_modified = (results_df["Config"] == modified_config) & (results_df["Model"] == model_name)

            if not mask_base.any() or not mask_modified.any():
                # Skip if one of the configs is missing (e.g. feature_sets not yet implemented)
                continue

            base_row     = results_df[mask_base].iloc[0]
            modified_row = results_df[mask_modified].iloc[0]

            rows.append({
                "Model":           model_name,
                "Comparison":      f"{base_config} → {modified_config}",
                "Description":     description,
                "Delta_Test_RMSE": round(modified_row["Test_RMSE"] - base_row["Test_RMSE"], 3),
                "Delta_Test_MAE":  round(modified_row["Test_MAE"]  - base_row["Test_MAE"],  3),
                "Delta_Test_R2":   round(modified_row["Test_R2"]   - base_row["Test_R2"],   4),
                "Delta_Gen_Gap":   round(
                    modified_row["Generalization_Gap"] - base_row["Generalization_Gap"], 3
                ),
            })

    delta_df = pd.DataFrame(rows)
    return delta_df
