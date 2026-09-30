"""
preprocessing.py  — COMPLETED IMPLEMENTATION
=============================================
aishwarya— Data Processing & Preprocessing (Part B)
Branch: feature/member2-preprocessing


"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def create_preprocessor(X_data: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    """
    Build ColumnTransformer for numerical and categorical columns.

    NOT fitted here — fitted inside Pipeline during cross_validate() or fit(),
    ensuring no test-data leakage.

    scale_numeric=True  → StandardScaler applied (for Ridge, scale-sensitive)
    scale_numeric=False → no scaling (for Random Forest, scale-invariant)
    """
    numeric_features = X_data.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = X_data.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_pipeline(estimator, X_data: pd.DataFrame, scale_numeric: bool = True) -> Pipeline:
    """Wrap preprocessor + estimator into a leakage-safe sklearn Pipeline."""
    preprocessor = create_preprocessor(X_data, scale_numeric=scale_numeric)
    return Pipeline([
        ("preprocessing", preprocessor),
        ("model",         estimator),
    ])
