"""
=============================================================================
Hybrid CNN-LSTM Framework for Spatial-Temporal Air Pollution Forecasting
=============================================================================
Full pipeline  (run: python main.py)
  1. Data loading & synthetic generation
  2. Preprocessing  (imputation -> outlier removal -> normalisation)
  3. Sliding-window sequence construction
  4. Train & evaluate: ARIMA . SVR . CNN . LSTM . CNN-LSTM
  5. Visualisations: loss curves . bar charts . predictions . feature importance
  6. Summary table printed to console
=============================================================================
"""

import os, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from data_generator  import generate_synthetic_dataset
from preprocessing   import preprocess
from window_builder  import build_windows
from models          import build_cnn, build_lstm, build_cnn_lstm
from train_evaluate  import (train_deep_model,
                              train_arima, evaluate_arima,
                              train_svr,
                              compute_metrics)
from visualise       import (plot_loss_curves, plot_comparison_bar,
                              plot_predictions, plot_feature_importance,
                              plot_horizon_comparison, print_summary_table)

# ---- CONFIG ----------------------------------------------------------------
DATA_PATH  = None          # set to CSV path to use real Kaggle data
TARGET_COL = "PM2.5"
HORIZONS   = [1, 6, 24]
LOOKBACK   = 24
BATCH_SIZE = 64
MAX_EPOCHS = 150
PATIENCE   = 20
SEED       = 42
OUT_DIR    = "outputs"

np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# ---- 1. DATA ---------------------------------------------------------------
print("\n" + "="*70)
print("  AIR POLLUTION FORECASTING -- CNN-LSTM HYBRID FRAMEWORK")
print("="*70)
print("\n[1/6] Loading data ...")

if DATA_PATH and os.path.exists(DATA_PATH):
    import pandas as pd
    raw_df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    print(f"      Real dataset loaded  ->  {raw_df.shape}")
else:
    raw_df = generate_synthetic_dataset(n_hours=8760 * 2)
    print(f"      Synthetic dataset created  ->  {raw_df.shape}")

# ---- 2. PREPROCESSING ------------------------------------------------------
print("\n[2/6] Preprocessing ...")
df_clean, scaler, feature_cols = preprocess(raw_df, target_col=TARGET_COL)
target_idx = feature_cols.index(TARGET_COL)
n_features = len(feature_cols)
print(f"      Features ({n_features}): {feature_cols}")
print(f"      Cleaned shape: {df_clean.shape}")

# ---- 3. SLIDING WINDOWS ----------------------------------------------------
print("\n[3/6] Building sliding windows ...")
data = build_windows(df_clean, feature_cols, TARGET_COL,
                     lookback=LOOKBACK, horizons=HORIZONS)
print(f"      X_train {data['X_train'].shape}  "
      f"X_val {data['X_val'].shape}  "
      f"X_test {data['X_test'].shape}")

# ---- 4. TRAIN & EVALUATE ---------------------------------------------------
print("\n[4/6] Training models ...\n")

results   = {}
histories = {}
preds_h1  = {}
y_true_h1 = None
hybrid_model_h1 = None

for H in HORIZONS:
    print(f"  -- Horizon H = {H:2d} h " + "-"*40)

    X_tr = data["X_train"];  y_tr = data["y_train_h"][H]
    X_va = data["X_val"];    y_va = data["y_val_h"][H]
    X_te = data["X_test"];   y_te = data["y_test_h"][H]

    # ARIMA
    name = "ARIMA"
    print(f"    [{name:<8}] ...", end=" ", flush=True)
    arima_fit   = train_arima(df_clean[TARGET_COL].values, len(y_te), H)
    arima_preds = evaluate_arima(arima_fit, len(y_te), H)
    m = compute_metrics(y_te, arima_preds, scaler, target_idx, n_features)
    results.setdefault(name, {})[H] = m
    if H == 1: preds_h1[name] = arima_preds
    print(f"RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  R2={m['r2']:.3f}")

    # SVR
    name = "SVR"
    print(f"    [{name:<8}] ...", end=" ", flush=True)
    _, svr_preds = train_svr(X_tr, y_tr, X_te)
    m = compute_metrics(y_te, svr_preds, scaler, target_idx, n_features)
    results.setdefault(name, {})[H] = m
    if H == 1: preds_h1[name] = svr_preds
    print(f"RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  R2={m['r2']:.3f}")

    # Standalone CNN
    name = "CNN"
    print(f"    [{name:<8}] ...", end=" ", flush=True)
    cnn_model = build_cnn(LOOKBACK, n_features, H)
    hist, cnn_preds = train_deep_model(cnn_model, X_tr, y_tr, X_va, y_va, X_te,
                                       batch_size=BATCH_SIZE, epochs=MAX_EPOCHS,
                                       patience=PATIENCE)
    if H == 1: histories[name] = hist
    m = compute_metrics(y_te, cnn_preds, scaler, target_idx, n_features)
    results.setdefault(name, {})[H] = m
    if H == 1: preds_h1[name] = cnn_preds
    print(f"RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  R2={m['r2']:.3f}")

    # Standalone LSTM
    name = "LSTM"
    print(f"    [{name:<8}] ...", end=" ", flush=True)
    lstm_model = build_lstm(LOOKBACK, n_features, H)
    hist, lstm_preds = train_deep_model(lstm_model, X_tr, y_tr, X_va, y_va, X_te,
                                        batch_size=BATCH_SIZE, epochs=MAX_EPOCHS,
                                        patience=PATIENCE)
    if H == 1: histories[name] = hist
    m = compute_metrics(y_te, lstm_preds, scaler, target_idx, n_features)
    results.setdefault(name, {})[H] = m
    if H == 1: preds_h1[name] = lstm_preds
    print(f"RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  R2={m['r2']:.3f}")

    # Hybrid CNN-LSTM
    name = "CNN-LSTM"
    print(f"    [{name:<8}] ...", end=" ", flush=True)
    hybrid_model = build_cnn_lstm(LOOKBACK, n_features, H)
    hist, hybrid_preds = train_deep_model(hybrid_model, X_tr, y_tr, X_va, y_va,
                                          X_te, batch_size=BATCH_SIZE,
                                          epochs=MAX_EPOCHS, patience=PATIENCE)
    if H == 1:
        histories[name]  = hist
        y_true_h1        = y_te
        hybrid_model_h1  = hybrid_model
    m = compute_metrics(y_te, hybrid_preds, scaler, target_idx, n_features)
    results.setdefault(name, {})[H] = m
    if H == 1: preds_h1[name] = hybrid_preds
    print(f"RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  R2={m['r2']:.3f}")

# ---- 5. VISUALISATIONS -----------------------------------------------------
print("\n[5/6] Generating visualisations ...")

plot_loss_curves(histories,
                 save_path=f"{OUT_DIR}/loss_curves.png")
plot_comparison_bar(results,
                    save_path=f"{OUT_DIR}/comparison_bar.png")
plot_predictions(y_true_h1, preds_h1, scaler, target_idx, n_features,
                 save_path=f"{OUT_DIR}/predictions_h1.png")
plot_horizon_comparison(results,
                        save_path=f"{OUT_DIR}/horizon_comparison.png")
plot_feature_importance(hybrid_model_h1, data["X_test"],
                        data["y_test_h"][1], feature_cols,
                        save_path=f"{OUT_DIR}/feature_importance.png")

# ---- 6. SUMMARY ------------------------------------------------------------
print("\n[6/6] Results Summary")
print_summary_table(results)
print(f"  All outputs saved to -> ./{OUT_DIR}/")
print("="*70 + "\n")
