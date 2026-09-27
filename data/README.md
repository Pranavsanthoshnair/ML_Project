# Data

## Folder Structure

```
data/
└── raw/
    └── train.csv   ← Place the dataset here
```

## Dataset

**Name:** Ames Housing Dataset  
**Source:** [Dean De Cock (2011)](http://jse.amstat.org/v19n3/decock.pdf) — commonly available via Kaggle  
**File:** `train.csv`  
**Shape:** ~2930 rows × 82 columns (including `SalePrice` target)

## How to Obtain

1. Download the Ames Housing dataset (`train.csv`) from Kaggle or another trusted source.
2. Place the file at `data/raw/train.csv`.

> **Note:** The dataset file is excluded from version control (see `.gitignore`).
> You must place `train.csv` manually before running the notebook.

## Key Columns

| Column | Type | Description |
|---|---|---|
| `SalePrice` | Numeric | Target variable — house sale price in USD |
| `Overall Qual` | Numeric (1–10) | Overall material and finish quality |
| `Gr Liv Area` | Numeric | Above-grade living area in sq. ft. |
| `Garage Cars` | Numeric | Garage capacity in car units |
| `Total Bsmt SF` | Numeric | Total basement area in sq. ft. |
| `1st Flr SF` | Numeric | First floor area in sq. ft. |
| `Year Built` | Numeric | Original construction year |
| `Year Remod/Add` | Numeric | Remodel year (same as build year if no remodel) |
| `Full Bath` | Numeric | Full bathrooms above grade |
| `TotRms AbvGrd` | Numeric | Total rooms above grade (excl. bathrooms) |
| `Garage Area` | Numeric | Garage area in sq. ft. |

> The dataset contains 82 total columns, including both numerical and categorical features,
> and various levels of missing values (handled inside the preprocessing pipeline).
