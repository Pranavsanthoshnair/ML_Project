import numpy as np
import pandas as pd

RANDOM_STATE = 42

# ── M10: 10 core meaningful features ─────────────────────────────────────────
M10_FEATURES = [
    "Overall Qual",    # Overall material and finish quality (1–10)
    "Gr Liv Area",     # Above-grade living area in square feet
    "Garage Cars",     # Garage capacity in car units
    "Total Bsmt SF",   # Total basement area in square feet
    "1st Flr SF",      # First floor area in square feet
    "Year Built",      # Original construction year
    "Year Remod/Add",  # Remodel/addition year
    "Full Bath",       # Full bathrooms above grade
    "TotRms AbvGrd",   # Total rooms above grade
    "Garage Area",     # Garage size in square feet
]

# ── M20: 20 meaningful features (M10 + 10 more) ──────────────────────────────
M20_FEATURES = M10_FEATURES + [
    "Overall Cond",    # Overall condition rating (1–10)
    "Bedroom AbvGr",   # Bedrooms above grade
    "Fireplaces",      # Number of fireplaces
    "Garage Yr Blt",   # Year garage was built
    "Mas Vnr Area",    # Masonry veneer area in square feet
    "BsmtFin SF 1",    # Type 1 finished basement area
    "Lot Area",        # Total lot size in square feet
    "Neighborhood",    # Physical location within Ames (categorical)
    "Kitchen Qual",    # Kitchen quality (categorical)
    "Central Air",     # Central air conditioning Y/N (categorical)
]

assert len(M10_FEATURES) == 10, "M10 must have exactly 10 features"
assert len(M20_FEATURES) == 20, "M20 must have exactly 20 features"

# ── Irrelevant features: 10 columns of pure random noise ─────────────────────
IRRELEVANT_FEATURE_NAMES = [f"RandomFeature_{i+1}" for i in range(10)]

# ── Redundant feature specs: {new_name: (original_col, multiplier, noise_sd)} ─
REDUNDANT_SPECS = {
    "Redundant_GrLivArea":    ("Gr Liv Area",   1.02, 10.0),
    "Redundant_OverallQual":  ("Overall Qual",  1.00,  0.20),
    "Redundant_GarageArea":   ("Garage Area",   1.01,  5.0),
    "Redundant_TotalBsmtSF":  ("Total Bsmt SF", 0.99, 10.0),
    "Redundant_YearBuilt":    ("Year Built",    1.00,  2.0),
}
REDUNDANT_FEATURE_NAMES = list(REDUNDANT_SPECS.keys())


def add_irrelevant_features(X: pd.DataFrame) -> pd.DataFrame:
    """Append 10 irrelevant (random noise) columns. Fixed seed — same values every run."""
    rng = np.random.RandomState(RANDOM_STATE)
    X_out = X.copy()
    for name in IRRELEVANT_FEATURE_NAMES:
        X_out[name] = rng.normal(0, 1, len(X_out))
    return X_out


def add_redundant_features(X: pd.DataFrame) -> pd.DataFrame:
    """Append 5 redundant (noisy linear transform) columns. Fixed seed."""
    rng = np.random.RandomState(RANDOM_STATE)
    X_out = X.copy()
    for new_name, (orig, mult, noise) in REDUNDANT_SPECS.items():
        X_out[new_name] = X_out[orig] * mult + rng.normal(0, noise, len(X_out))
    return X_out


def add_all_synthetic_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add both irrelevant and redundant features in one call."""
    return add_redundant_features(add_irrelevant_features(X))


def build_feature_configs(all_original_cols: list) -> dict:
    """
    Return experiment configuration dictionary.
    M20 is the PRIMARY BASELINE for H2/H3/H4 comparisons.
    ALL is contextual only — NOT the baseline for irrelevant/redundant tests.
    """
    return {
        "M10":     M10_FEATURES,
        "M20":     M20_FEATURES,
        "M20_I":   M20_FEATURES + IRRELEVANT_FEATURE_NAMES,
        "M20_R":   M20_FEATURES + REDUNDANT_FEATURE_NAMES,
        "M20_I_R": M20_FEATURES + IRRELEVANT_FEATURE_NAMES + REDUNDANT_FEATURE_NAMES,
        "ALL":     all_original_cols,
    }


def redundancy_correlation_table(X_experiment: pd.DataFrame) -> pd.DataFrame:
    """Return Pearson correlation for each redundant feature vs its original."""
    rows = []
    for new_name, (orig, mult, noise) in REDUNDANT_SPECS.items():
        corr = X_experiment[orig].corr(X_experiment[new_name])
        rows.append({
            "Original Feature":  orig,
            "Redundant Feature": new_name,
            "Multiplier":        mult,
            "Noise SD":          noise,
            "Pearson r":         round(corr, 4),
        })
    return pd.DataFrame(rows)