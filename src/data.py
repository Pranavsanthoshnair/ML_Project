"""
data.py
=======
MEMBER 2 — Data Processing & Preprocessing (Part A)
Branch: feature/member2-preprocessing

TODO for Member 2:
  - Implement load_data()
  - Implement inspect_data()
  - Implement recode_absence_nans() — handle Ames NaN semantics
  - Implement prepare_XY() — drop SalePrice, Order, PID, Id
  - Implement make_train_test_split() — fixed 80/20, random_state=42
  - Implement full_pipeline() convenience function
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

TARGET       = "SalePrice"
RANDOM_STATE = 42
DROP_COLS    = ["SalePrice", "Order", "PID", "Id"]

# TODO Member 2: List columns where NaN means absence (no garage, no basement etc.)
CATEGORICAL_ABSENCE_COLS = []   # fill in column names → fillna("None")
NUMERICAL_ABSENCE_COLS   = []   # fill in column names → fillna(0)


def recode_absence_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Replace absence NaNs with 'None' (categorical) or 0 (numerical)."""
    raise NotImplementedError("Member 2: implement this function")


def load_data(data_path: str = "data/raw/train.csv") -> pd.DataFrame:
    """Load CSV and return raw DataFrame."""
    raise NotImplementedError("Member 2: implement this function")


def inspect_data(df: pd.DataFrame) -> None:
    """Print shape, target presence, duplicates, top missing columns."""
    raise NotImplementedError("Member 2: implement this function")


def prepare_XY(df: pd.DataFrame):
    """Separate X (predictors) and y (SalePrice). Drop identifier columns."""
    raise NotImplementedError("Member 2: implement this function")


def make_train_test_split(X: pd.DataFrame, y: pd.Series):
    """Fixed 80/20 split with random_state=42. Return X_train, X_test, y_train, y_test."""
    raise NotImplementedError("Member 2: implement this function")


def full_pipeline(data_path: str = "data/raw/train.csv"):
    """End-to-end: load → recode → prepare_XY → split. Return X, y, X_train, X_test, y_train, y_test."""
    raise NotImplementedError("Member 2: implement this function")
