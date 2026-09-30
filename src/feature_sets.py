"""
feature_sets.py
===============
MEMBER 1 — Experimental Design & Feature Construction
Branch: feature/member1-feature-design

TODO for Member 1:
  - Define M10_FEATURES (10 meaningful features)
  - Define M20_FEATURES (20 meaningful features)
  - Implement add_irrelevant_features()
  - Implement add_redundant_features()
  - Implement build_feature_configs()
  - Implement redundancy_correlation_table()
  - Document hypotheses H1-H4 in this file
"""

import numpy as np
import pandas as pd

RANDOM_STATE = 42

# ── Member 1: Define the 10 core meaningful features ─────────────────────────
M10_FEATURES = []          # TODO: fill in 10 feature names

# ── Member 1: Define the 20 meaningful features (M10 + 10 more) ─────────────
M20_FEATURES = []          # TODO: fill in 20 feature names

# ── Member 1: Define irrelevant feature names ─────────────────────────────────
IRRELEVANT_FEATURE_NAMES = []   # TODO: define 10 names like "RandomFeature_1" etc.

# ── Member 1: Define redundant feature specs ──────────────────────────────────
REDUNDANT_SPECS = {}            # TODO: {new_name: (original_col, multiplier, noise_sd)}
REDUNDANT_FEATURE_NAMES = []    # TODO: list(REDUNDANT_SPECS.keys())


def add_irrelevant_features(X: pd.DataFrame) -> pd.DataFrame:
    """Append 10 irrelevant (random noise) columns to X. Fixed seed."""
    raise NotImplementedError("Member 1: implement this function")


def add_redundant_features(X: pd.DataFrame) -> pd.DataFrame:
    """Append 5 redundant (noisy linear transform) columns to X. Fixed seed."""
    raise NotImplementedError("Member 1: implement this function")


def add_all_synthetic_features(X: pd.DataFrame) -> pd.DataFrame:
    """Convenience: add both irrelevant and redundant features."""
    raise NotImplementedError("Member 1: implement this function")


def build_feature_configs(all_original_cols: list) -> dict:
    """Return dict: {config_name: [feature_names]} for all 6 experiments."""
    raise NotImplementedError("Member 1: implement this function")


def redundancy_correlation_table(X_experiment: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame showing Pearson r for each redundant pair."""
    raise NotImplementedError("Member 1: implement this function")
