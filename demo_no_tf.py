"""
demo_no_tf.py  --  validates the full pipeline with sklearn-only models
(SVR + Ridge regression as CNN/LSTM proxies) so the code runs in any
environment. On your machine with TF/statsmodels installed, run main.py.
"""
import os, warnings, sys
warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── 1. synthetic data ────────────────────────────────────────────────────────
sys.path.insert(0, ".")
from data_generator import generate_synthetic_dataset

print("Generating synthetic dataset ...")
raw_df = generate_synthetic_dataset(n_hours=8760)
print(f"  shape: {raw_df.shape}")

# ── 2. preprocessing ─────────────────────────────────────────────────────────
print("Preprocessing ...")
from preprocessing import preprocess
df_clean, scaler, feature_cols = preprocess(raw_df, target_col="PM2.5")
target_idx = feature_cols.index("PM2.5")
n_features = len(feature_cols)

# ── 3. window builder ────────────────────────────────────────────────────────
print("Building windows ...")
from window_builder import build_windows
data = build_windows(df_clean, feature_cols, "PM2.5",
                     lookback=24, horizons=[1, 6, 24])
print(f"  X_train {data['X_train'].shape} | X_test {data['X_test'].shape}")

# ── 4. helper: inverse-transform + metrics ───────────────────────────────────
def inv(arr):
    dummy = np.zeros((len(arr), n_features))
    dummy[:, target_idx] = arr.ravel()
    return scaler.inverse_transform(dummy)[:, target_idx]

def metrics(y_true, y_pred):
    yt, yp = inv(y_true), inv(y_pred)
    return {"rmse": float(np.sqrt(mean_squared_error(yt,yp))),
            "mae" : float(mean_absolute_error(yt,yp)),
            "r2"  : float(r2_score(yt,yp)),
            "yt"  : yt, "yp": yp}

# ── 5. train surrogate models for all horizons ───────────────────────────────
HORIZONS = [1, 6, 24]
results  = {}
preds_h1 = {}
MODEL_ORDER = ["SVR", "Ridge-CNN-proxy", "Ridge-LSTM-proxy",
               "Ridge-Hybrid-proxy"]

print("\nTraining models ...")
for H in HORIZONS:
    X_tr = data["X_train"].reshape(data["X_train"].shape[0], -1)
    X_te = data["X_test"].reshape(data["X_test"].shape[0], -1)
    y_tr = data["y_train_h"][H]
    y_te = data["y_test_h"][H]

    for mname, mdl in [
        ("SVR",               SVR(C=10, epsilon=0.05)),
        ("Ridge-CNN-proxy",   Ridge(alpha=0.1)),
        ("Ridge-LSTM-proxy",  Ridge(alpha=1.0)),
        ("Ridge-Hybrid-proxy",Ridge(alpha=0.01)),
    ]:
        mdl.fit(X_tr, y_tr)
        yp = mdl.predict(X_te)
        m  = metrics(y_te, yp)
        results.setdefault(mname, {})[H] = m
        if H == 1:
            preds_h1[mname] = yp
        print(f"  H={H:2d}h | {mname:<22} "
              f"RMSE={m['rmse']:6.2f}  MAE={m['mae']:6.2f}  R2={m['r2']:.3f}")

y_true_h1 = data["y_test_h"][1]

# ── 6. visualisations ────────────────────────────────────────────────────────
PALETTE = {"SVR"              : "#E07B54",
           "Ridge-CNN-proxy"  : "#5BA4CF",
           "Ridge-LSTM-proxy" : "#6ABF69",
           "Ridge-Hybrid-proxy":"#9B59B6"}

# Bar chart
horizons = HORIZONS
fig, axes = plt.subplots(len(horizons), 3, figsize=(16, 4.5*len(horizons)))
metrics_list = ["rmse","mae","r2"]
ylabels = ["RMSE (ug/m3)","MAE (ug/m3)","R2 Score"]
for row, H in enumerate(horizons):
    for col, (met, yl) in enumerate(zip(metrics_list, ylabels)):
        ax   = axes[row][col]
        vals = [results[m][H][met] for m in MODEL_ORDER]
        clrs = [PALETTE[m] for m in MODEL_ORDER]
        bars = ax.bar(MODEL_ORDER, vals, color=clrs, width=0.55,
                      edgecolor="white", linewidth=1.2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+max(vals)*0.02,
                    f"{v:.3f}" if met=="r2" else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_title(f"H={H}h -- {yl}", fontsize=10, fontweight="bold")
        ax.set_ylim(0, max(vals)*1.20)
        ax.tick_params(axis="x", labelsize=7, rotation=15)
        ax.grid(axis="y", alpha=0.3)
        best = int(np.argmin(vals)) if met!="r2" else int(np.argmax(vals))
        bars[best].set_edgecolor("black"); bars[best].set_linewidth(2.5)
fig.suptitle("Model Comparison -- PM2.5 Forecasting (Demo)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig("outputs/comparison_bar.png", dpi=150, bbox_inches="tight")
plt.close(); print("\nSaved outputs/comparison_bar.png")

# Prediction plot H=1
n_plot = 200
yt = inv(y_true_h1[:n_plot])
fig, axes = plt.subplots(len(MODEL_ORDER), 1,
                         figsize=(14, 3.5*len(MODEL_ORDER)), sharex=True)
for ax, name in zip(axes, MODEL_ORDER):
    yp = inv(preds_h1[name][:n_plot])
    ax.plot(yt, color="black", lw=1.2, label="Actual PM2.5", alpha=0.85)
    ax.plot(yp, color=PALETTE[name], lw=1.2, linestyle="--",
            label=f"{name} Pred", alpha=0.80)
    ax.set_ylabel("PM2.5 (ug/m3)", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(name, fontsize=11, fontweight="bold", color=PALETTE[name])
    ax.grid(alpha=0.25)
axes[-1].set_xlabel("Test Sample Index", fontsize=10)
fig.suptitle("Actual vs Predicted PM2.5 -- H=1h (first 200 samples)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig("outputs/predictions_h1.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved outputs/predictions_h1.png")

# Horizon comparison line chart
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, met, yl in zip(axes, metrics_list, ylabels):
    for name in MODEL_ORDER:
        vals = [results[name][H][met] for H in horizons]
        ax.plot(horizons, vals, marker="o", lw=2.2,
                label=name, color=PALETTE[name])
    ax.set_xlabel("Forecast Horizon (h)", fontsize=11)
    ax.set_ylabel(yl, fontsize=11)
    ax.set_title(yl, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8); ax.set_xticks(horizons); ax.grid(alpha=0.3)
fig.suptitle("Performance vs Forecast Horizon", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig("outputs/horizon_comparison.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved outputs/horizon_comparison.png")

# ── 7. summary table ─────────────────────────────────────────────────────────
print("\n" + "="*80)
print(f"  {'Model':<24}" +
      "".join([f"  H={H:2d}h RMSE   MAE    R2  " for H in HORIZONS]))
print("="*80)
for name in MODEL_ORDER:
    row = f"  {name:<24}"
    for H in HORIZONS:
        m = results[name][H]
        row += f"  {m['rmse']:6.2f}  {m['mae']:5.2f}  {m['r2']:.3f} "
    print(row)
print("="*80)
print("\nAll outputs saved to ./outputs/\nPIPELINE COMPLETE.\n")
