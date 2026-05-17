"""
window_builder.py
Converts the normalised time-series DataFrame into supervised learning
tensors using a sliding window approach (Section 4.2.4 of the paper).

Output shapes
─────────────
X          : (N, lookback, n_features)   — 3-D input for CNN / LSTM
y_h        : (N,)                        — 1-D targets per horizon H

Train / Val / Test split is 70 / 15 / 15 % using strict temporal ordering
to prevent data leakage.
"""

import numpy as np
import pandas as pd
from typing import Dict, List


def build_windows(df: pd.DataFrame,
                  feature_cols: List[str],
                  target_col:   str,
                  lookback:     int = 24,
                  horizons:     List[int] = [1, 6, 24]
                  ) -> Dict:
    """
    Parameters
    ----------
    df           : normalised DataFrame (output of preprocess())
    feature_cols : list of ALL feature column names (target included)
    target_col   : name of the target column
    lookback     : number of past time-steps as input (T)
    horizons     : list of forecast horizons H (in time steps)

    Returns
    -------
    dict with keys:
      X_train, X_val, X_test               (3-D float32 arrays)
      y_train_h, y_val_h, y_test_h         (dict horizon → 1-D array)
    """

    values      = df[feature_cols].values.astype(np.float32)
    target_idx  = feature_cols.index(target_col)
    max_horizon = max(horizons)
    n           = len(values)

    X_list = []
    y_dict = {H: [] for H in horizons}

    for i in range(lookback, n - max_horizon + 1):
        X_list.append(values[i - lookback : i, :])        # (lookback, F)
        for H in horizons:
            y_dict[H].append(values[i + H - 1, target_idx])  # scalar

    X = np.array(X_list,  dtype=np.float32)   # (N, lookback, F)
    y = {H: np.array(y_dict[H], dtype=np.float32) for H in horizons}

    N = len(X)
    n_train = int(N * 0.70)
    n_val   = int(N * 0.15)
    # test = remaining

    splits = {
        "X_train": X[:n_train],
        "X_val"  : X[n_train : n_train + n_val],
        "X_test" : X[n_train + n_val :],
        "y_train_h": {H: y[H][:n_train]               for H in horizons},
        "y_val_h"  : {H: y[H][n_train:n_train + n_val] for H in horizons},
        "y_test_h" : {H: y[H][n_train + n_val:]        for H in horizons},
    }

    return splits
