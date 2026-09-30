"""
experiments.py — Pranav S Nair
Branch: feature/pranav-models
"""

import pandas as pd
from sklearn.pipeline import Pipeline

from src.preprocessing import create_preprocessor, build_pipeline
from src.models import make_models, evaluate_on_split
from src.feature_sets import build_feature_configs

COMPARISON_PAIRS = [
    ("M10",   "M20",     "M10 → M20: adding meaningful features (H1)"),
    ("M20",   "M20_I",   "M20 → M20+I: adding irrelevant features (H2)"),
    ("M20",   "M20_R",   "M20 → M20+R: adding redundant features (H3)"),
    ("M20",   "M20_I_R", "M20 → M20+I+R: adding both (H4)"),
]


def run_all_experiments(X_experiment, X_train, X_test,
                        y_train, y_test, all_original_cols) -> pd.DataFrame:
    feature_configs = build_feature_configs(all_original_cols)
    results = []

    for config_name, feature_list in feature_configs.items():
        X_train_exp = X_experiment.loc[X_train.index, feature_list].copy()
        X_test_exp  = X_experiment.loc[X_test.index,  feature_list].copy()

        for model_name, estimator in make_models().items():
            scale    = (model_name == "Ridge")
            pipeline = build_pipeline(estimator, X_train_exp, scale_numeric=scale)
            metrics  = evaluate_on_split(pipeline, X_train_exp, X_test_exp,
                                         y_train, y_test)

            fitted_prep = pipeline.named_steps["preprocessing"]
            try:
                transformed_count = len(fitted_prep.get_feature_names_out())
            except AttributeError:
                transformed_count = -1

            results.append({
                "Config":               config_name,
                "Model":                model_name,
                "Raw_Features":         len(feature_list),
                "Transformed_Features": transformed_count,
                **metrics,
            })

    results_df = pd.DataFrame(results)
    config_order = list(feature_configs.keys())
    results_df["Config"] = pd.Categorical(
        results_df["Config"], categories=config_order, ordered=True
    )
    return results_df.sort_values(["Model", "Config"]).reset_index(drop=True)


def compute_pairwise_deltas(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for base_config, modified_config, description in COMPARISON_PAIRS:
        for model_name in results_df["Model"].unique():
            mask_base     = (results_df["Config"] == base_config)     & (results_df["Model"] == model_name)
            mask_modified = (results_df["Config"] == modified_config) & (results_df["Model"] == model_name)

            if not mask_base.any() or not mask_modified.any():
                continue

            b = results_df[mask_base].iloc[0]
            m = results_df[mask_modified].iloc[0]

            rows.append({
                "Model":           model_name,
                "Comparison":      f"{base_config} → {modified_config}",
                "Description":     description,
                "Delta_Test_RMSE": round(m["Test_RMSE"] - b["Test_RMSE"], 3),
                "Delta_Test_MAE":  round(m["Test_MAE"]  - b["Test_MAE"],  3),
                "Delta_Test_R2":   round(m["Test_R2"]   - b["Test_R2"],   4),
                "Delta_Gen_Gap":   round(m["Generalization_Gap"] - b["Generalization_Gap"], 3),
            })

    return pd.DataFrame(rows)
