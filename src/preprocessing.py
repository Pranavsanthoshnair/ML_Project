"""
preprocessing.py
================
MEMBER 2 — Data Processing & Preprocessing (Part B)
Branch: feature/member2-preprocessing

TODO for Member 2:
  - Implement create_preprocessor() — ColumnTransformer for num + cat columns
  - Implement build_pipeline() — wrap preprocessor + model in sklearn Pipeline
  - Ensure all fitting happens inside Pipeline (no leakage)
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def create_preprocessor(X_data: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    """
    Build ColumnTransformer: median imputation (+optional scaling) for numerics,
    most_frequent imputation + OneHotEncoder for categoricals.
    scale_numeric=True for Ridge (scale-sensitive), False for Random Forest.
    """
    raise NotImplementedError("Member 2: implement this function")


def build_pipeline(estimator, X_data: pd.DataFrame, scale_numeric: bool = True) -> Pipeline:
    """Wrap preprocessor + estimator into a leakage-safe sklearn Pipeline."""
    raise NotImplementedError("Member 2: implement this function")
