"""
train_evaluate.py
Training and evaluation routines for all five models:
  - Deep models (CNN, LSTM, CNN-LSTM) → train_deep_model / evaluate_deep
  - ARIMA                             → train_arima    / evaluate_arima
  - SVR                               → train_svr      / evaluate_svr

Metrics returned: RMSE, MAE, R² (on the inverse-transformed scale).
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import warnings

import tensorflow as tf
from tensorflow.keras.callbacks import (EarlyStopping,
                                         ReduceLROnPlateau,
                                         ModelCheckpoint)


# ════════════════════════════════════════════════════════════════════════════
# 1. Deep Learning (CNN / LSTM / CNN-LSTM)
# ════════════════════════════════════════════════════════════════════════════
def train_deep_model(model,
                     X_train, y_train,
                     X_val,   y_val,
                     X_test,
                     batch_size: int = 64,
                     epochs:     int = 150,
                     patience:   int = 20):
    """
    Train a compiled Keras model with early stopping and LR reduction.

    Returns
    -------
    history   : Keras History object  (loss / val_loss curves)
    y_pred    : 1-D numpy array of test-set predictions (normalised scale)
    """
    callbacks = [
        EarlyStopping(monitor="val_loss",
                      patience=patience,
                      restore_best_weights=True,
                      verbose=0),
        ReduceLROnPlateau(monitor="val_loss",
                          factor=0.5,
                          patience=patience // 2,
                          min_lr=1e-6,
                          verbose=0),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    y_pred = model.predict(X_test, verbose=0).ravel()
    return history, y_pred


def evaluate_deep(model, X_test):
    """Return raw normalised predictions from a trained Keras model."""
    return model.predict(X_test, verbose=0).ravel()


# ════════════════════════════════════════════════════════════════════════════
# 2. ARIMA Baseline
# ════════════════════════════════════════════════════════════════════════════
def train_arima(series: np.ndarray,
                n_test: int,
                horizon: int = 1,
                order:   tuple = (2, 1, 2)):
    """
    Fit ARIMA(p,d,q) on the training portion of `series`.

    Parameters
    ----------
    series  : full (normalised) target series
    n_test  : number of test samples (to determine train/test split)
    horizon : forecast horizon (we use a re-fit rolling forecast)
    order   : ARIMA (p, d, q)

    Returns
    -------
    Fitted ARIMA model result object (for evaluate_arima)
    """
    train_series = series[:-(n_test + horizon - 1)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_fit = ARIMA(train_series, order=order).fit()
    return model_fit


def evaluate_arima(model_fit,
                   n_test: int,
                   horizon: int = 1) -> np.ndarray:
    """
    Rolling one-step (or H-step) ARIMA forecast for n_test steps.
    Returns normalised predictions array of length n_test.
    """
    # Use the in-sample model for forecasting; simple rolling predict
    forecasts = model_fit.forecast(steps=n_test + horizon - 1)
    # For horizon H, the prediction for step i is forecasts[i + H - 1]
    preds = np.array([forecasts[i + horizon - 1] for i in range(n_test)])
    return preds.ravel()


# ════════════════════════════════════════════════════════════════════════════
# 3. SVR Baseline
# ════════════════════════════════════════════════════════════════════════════
def train_svr(X_train: np.ndarray,
              y_train: np.ndarray,
              X_test:  np.ndarray,
              C:       float = 10.0,
              epsilon: float = 0.05,
              gamma:   str   = "scale"):
    """
    Train SVR with RBF kernel.
    X_train shape: (N, lookback, n_features) — flattened to 2-D internally.

    Returns
    -------
    svr_model  : fitted SVR
    y_pred     : normalised predictions on X_test
    """
    # Flatten temporal dimension: (N, lookback*n_features)
    Xtr = X_train.reshape(X_train.shape[0], -1)
    Xte = X_test.reshape(X_test.shape[0], -1)

    svr = SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma)
    svr.fit(Xtr, y_train.ravel())
    y_pred = svr.predict(Xte)
    return svr, y_pred.ravel()


# ════════════════════════════════════════════════════════════════════════════
# 4. Shared metric helper
# ════════════════════════════════════════════════════════════════════════════
def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    scaler,
                    target_idx: int,
                    n_features:  int) -> dict:
    """
    Inverse-transform both arrays then compute RMSE / MAE / R².

    Parameters
    ----------
    y_true / y_pred  : 1-D arrays in normalised scale
    scaler           : fitted MinMaxScaler
    target_idx       : column index of the target in scaler
    n_features       : total number of features
    """
    def inv(arr):
        dummy = np.zeros((len(arr), n_features))
        dummy[:, target_idx] = arr.ravel()
        return scaler.inverse_transform(dummy)[:, target_idx]

    yt = inv(y_true)
    yp = inv(y_pred)

    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae  = float(mean_absolute_error(yt, yp))
    r2   = float(r2_score(yt, yp))

    return {"rmse": rmse, "mae": mae, "r2": r2,
            "y_true_inv": yt, "y_pred_inv": yp}
