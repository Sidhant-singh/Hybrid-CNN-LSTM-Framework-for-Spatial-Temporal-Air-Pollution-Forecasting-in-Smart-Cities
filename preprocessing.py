"""
preprocessing.py
Full preprocessing pipeline as described in Section 4.2 of the paper:
  1. Temporal sorting
  2. Missing value imputation (short gaps → linear interp;
                               long gaps  → KNN imputation)
  3. Outlier detection & treatment (IQR-based)
  4. Min-Max normalisation  (fit on train only — done inside window_builder)
     Here we return the fitted scaler so it can be used for inverse-transform.
  5. Feature / column alignment
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer


# Columns expected in the raw DataFrame
FEATURE_COLS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3",
                "Temperature", "Humidity", "Wind_Speed"]


def preprocess(df: pd.DataFrame,
               target_col: str = "PM2.5",
               short_gap_threshold: int = 3,
               knn_neighbours: int = 5
               ) -> tuple[pd.DataFrame, MinMaxScaler, list]:
    """
    Parameters
    ----------
    df                    : raw DataFrame (must contain FEATURE_COLS + 'Date')
    target_col            : column to forecast
    short_gap_threshold   : consecutive NaN runs ≤ this are linearly interpolated
    knn_neighbours        : k for KNN imputation of longer gaps

    Returns
    -------
    df_clean  : normalised DataFrame with FEATURE_COLS
    scaler    : fitted MinMaxScaler (for later inverse-transform)
    feat_cols : ordered list of feature column names
    """

    df = df.copy()

    # ── 1. Ensure Date index and temporal ordering ────────────────────────
    if "Date" in df.columns:
        df = df.set_index("Date")
    df = df.sort_index()

    # Keep only relevant columns
    available = [c for c in FEATURE_COLS if c in df.columns]
    df = df[available]

    # ── 2. Missing value imputation ───────────────────────────────────────
    print("      Step 1 – Imputing missing values …")
    missing_before = df.isna().sum().sum()

    # 2a. Short gaps: linear interpolation (limit = short_gap_threshold)
    df = df.interpolate(method="linear", limit=short_gap_threshold, axis=0)

    # 2b. Remaining longer gaps: KNN imputation
    remaining = df.isna().sum().sum()
    if remaining > 0:
        imputer = KNNImputer(n_neighbors=knn_neighbours)
        df[available] = imputer.fit_transform(df[available])

    missing_after = df.isna().sum().sum()
    print(f"         Missing before: {missing_before}  →  after: {missing_after}")

    # ── 3. Outlier treatment (IQR × 3 rule) ──────────────────────────────
    print("      Step 2 – Outlier treatment (IQR × 3) …")
    outliers_replaced = 0
    for col in available:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        upper = Q3 + 3 * IQR
        lower = Q1 - 3 * IQR
        mask = (df[col] > upper) | (df[col] < lower)
        if mask.sum() > 0:
            # Replace with rolling median (window=5)
            rolling_med = df[col].rolling(5, center=True, min_periods=1).median()
            df.loc[mask, col] = rolling_med[mask]
            outliers_replaced += mask.sum()
    print(f"         Outlier cells replaced: {outliers_replaced}")

    # ── 4. Min-Max normalisation ──────────────────────────────────────────
    print("      Step 3 – Min-Max normalisation …")
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Fit on the first 70% (train portion) to prevent leakage
    n_train = int(len(df) * 0.70)
    scaler.fit(df.iloc[:n_train][available])
    df_scaled = pd.DataFrame(
        scaler.transform(df[available]),
        index=df.index,
        columns=available
    )

    # Reorder so target is last (convenient for window building)
    feat_cols = [c for c in available if c != target_col] + [target_col]
    df_scaled = df_scaled[feat_cols]

    return df_scaled, scaler, feat_cols
