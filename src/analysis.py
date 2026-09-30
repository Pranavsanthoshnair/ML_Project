"""
analysis.py
===========
MEMBER 4 — Validation, Metrics & Statistical Analysis (visualizations)
Branch: feature/member4-validation

TODO for Member 4:
  - Implement all 5 required plots (handoff spec §23)
  - Implement plot_redundancy_heatmap()
  - Implement build_evidence_table() — save CSVs to results/tables/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

CONFIG_ORDER = ["M10", "M20", "M20_I", "M20_R", "M20_I_R", "ALL"]
PALETTE      = {"Ridge": "#4C72B0", "Random Forest": "#DD8452"}
FIGURES_DIR  = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_rmse_by_config(results_df: pd.DataFrame, save: bool = True):
    """Plot 1: Bar chart — Test RMSE per config × model. M20 is main baseline."""
    raise NotImplementedError("Member 4: implement this function")


def plot_generalization(results_df: pd.DataFrame, cv_df: pd.DataFrame, save: bool = True):
    """Plot 2: Train vs CV vs Test RMSE — shows generalization behaviour."""
    raise NotImplementedError("Member 4: implement this function")


def plot_cv_distribution(cv_df: pd.DataFrame, save: bool = True):
    """Plot 3: Boxplot of fold-level CV RMSE — shows stability across folds."""
    raise NotImplementedError("Member 4: implement this function")


def plot_delta_rmse(delta_df: pd.DataFrame, save: bool = True):
    """Plot 4: Horizontal bar — pairwise ΔRMSE. Direct answer to research question."""
    raise NotImplementedError("Member 4: implement this function")


def plot_feature_count_vs_rmse(results_df: pd.DataFrame, save: bool = True):
    """Plot 5: Feature count vs Test RMSE — shows quantity does not equal quality."""
    raise NotImplementedError("Member 4: implement this function")


def plot_redundancy_heatmap(X_experiment: pd.DataFrame, redundant_specs: dict, save: bool = True):
    """Heatmap: correlation between originals and redundant features (evidence)."""
    raise NotImplementedError("Member 4: implement this function")


def build_evidence_table(results_df, cv_df, delta_df, cv_delta_df) -> pd.DataFrame:
    """Merge all results into one table. Save CSVs to results/tables/."""
    raise NotImplementedError("Member 4: implement this function")
