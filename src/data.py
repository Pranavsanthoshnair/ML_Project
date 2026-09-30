"""
data.py  — COMPLETED IMPLEMENTATION
=====================================
Aishwarya— Data Processing & Preprocessing (Part A)
Branch: feature/member2-preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

TARGET       = "SalePrice"
RANDOM_STATE = 42
DROP_COLS    = ["SalePrice", "Order", "PID", "Id"]

# Ames data dictionary: NaN = absence, not unknown
CATEGORICAL_ABSENCE_COLS = [
    "Alley", "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure",
    "BsmtFin Type 1", "BsmtFin Type 2", "Fireplace Qu",
    "Garage Type", "Garage Finish", "Garage Qual", "Garage Cond",
    "Pool QC", "Fence", "Misc Feature", "Mas Vnr Type",
]
NUMERICAL_ABSENCE_COLS = [
    "Mas Vnr Area", "BsmtFin SF 1", "BsmtFin SF 2",
    "Bsmt Unf SF", "Total Bsmt SF", "Bsmt Full Bath",
    "Bsmt Half Bath", "Garage Yr Blt", "Garage Cars", "Garage Area",
]


def recode_absence_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Replace absence NaNs: categorical → 'None', numerical → 0."""
    df = df.copy()
    for col in CATEGORICAL_ABSENCE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("None")
    for col in NUMERICAL_ABSENCE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    return df


def load_data(data_path: str = "data/raw/train.csv") -> pd.DataFrame:
    """Load CSV and return raw DataFrame."""
    df = pd.read_csv(data_path)
    print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    assert TARGET in df.columns, f"Target column '{TARGET}' not found"
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Print shape, target presence, duplicates, top missing columns."""
    print("=" * 60)
    print(f"Rows     : {df.shape[0]}")
    print(f"Columns  : {df.shape[1]}")
    print(f"SalePrice present: {'SalePrice' in df.columns}")
    print(f"Duplicates: {df.duplicated().sum()}")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing):
        print("Top missing columns:")
        print(missing.head(15).to_string())
    print("=" * 60)


def prepare_XY(df: pd.DataFrame):
    """Separate X and y. Drop SalePrice, Order, PID, Id."""
    y = df[TARGET].copy()
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=cols_to_drop).copy()
    print(f"Predictor shape: {X.shape}  |  Dropped: {cols_to_drop}")
    return X, y


def make_train_test_split(X: pd.DataFrame, y: pd.Series):
    """Fixed 80/20 split, random_state=42. Same split for all experiments."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def full_pipeline(data_path: str = "data/raw/train.csv"):
    """End-to-end: load → inspect → recode absence NaNs → prepare → split."""
    df = load_data(data_path)
    inspect_data(df)
    df = recode_absence_nans(df)
    X, y = prepare_XY(df)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)
    return X, y, X_train, X_test, y_train, y_test
