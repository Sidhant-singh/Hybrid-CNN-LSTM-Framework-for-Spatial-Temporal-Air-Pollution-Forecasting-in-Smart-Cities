"""
visualise.py
All visualisation and reporting functions used in the paper:
  1. plot_loss_curves()         – training vs validation loss per deep model
  2. plot_comparison_bar()      – RMSE / MAE / R² grouped bar charts
  3. plot_predictions()         – actual vs predicted for H=1
  4. plot_horizon_comparison()  – RMSE across all models × all horizons
  5. plot_feature_importance()  – permutation-based feature importance
  6. print_summary_table()      – LaTeX-ready ASCII table to console
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# Palette consistent across all figures
PALETTE = {
    "ARIMA"   : "#E07B54",
    "SVR"     : "#F5C842",
    "CNN"     : "#5BA4CF",
    "LSTM"    : "#6ABF69",
    "CNN-LSTM": "#9B59B6",
}
MODEL_ORDER = ["ARIMA", "SVR", "CNN", "LSTM", "CNN-LSTM"]


# ════════════════════════════════════════════════════════════════════════════
# 1. Loss Curves
# ════════════════════════════════════════════════════════════════════════════
def plot_loss_curves(histories: dict, save_path: str = "loss_curves.png"):
    """Plot training & validation loss curves for CNN, LSTM, CNN-LSTM."""
    deep_models = [m for m in MODEL_ORDER if m in histories]
    n = len(deep_models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, deep_models):
        hist = histories[name]
        ep   = range(1, len(hist.history["loss"]) + 1)
        ax.plot(ep, hist.history["loss"],    label="Train Loss",
                color=PALETTE[name], linewidth=2)
        ax.plot(ep, hist.history["val_loss"], label="Val Loss",
                color=PALETTE[name], linewidth=2, linestyle="--", alpha=0.7)
        ax.set_title(f"{name} – Loss Curves", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("MSE Loss", fontsize=11)
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)

    fig.suptitle("Training vs Validation Loss (H = 1 h)", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved → {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# 2. Comparison Bar Chart (RMSE, MAE, R²) for each horizon
# ════════════════════════════════════════════════════════════════════════════
def plot_comparison_bar(results: dict, save_path: str = "comparison_bar.png"):
    """
    Grouped bar charts: one subplot per horizon, 3 metrics per model.
    results[model_name][horizon] = {rmse, mae, r2}
    """
    horizons = sorted(list(next(iter(results.values())).keys()))
    metrics  = ["rmse", "mae", "r2"]
    labels   = ["RMSE (μg/m³)", "MAE (μg/m³)", "R² Score"]

    n_h = len(horizons)
    fig, axes = plt.subplots(n_h, 3, figsize=(16, 4.5 * n_h))
    if n_h == 1:
        axes = [axes]

    x    = np.arange(len(MODEL_ORDER))
    width = 0.65 / len(MODEL_ORDER)          # not needed — one bar per model

    for row, H in enumerate(horizons):
        for col, (metric, label) in enumerate(zip(metrics, labels)):
            ax = axes[row][col]
            vals   = [results[m][H][metric] for m in MODEL_ORDER]
            colors = [PALETTE[m]            for m in MODEL_ORDER]
            bars   = ax.bar(MODEL_ORDER, vals, color=colors,
                            edgecolor="white", linewidth=1.2, width=0.55)

            # Annotate bar tops
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(vals) * 0.02,
                        f"{v:.3f}" if metric == "r2" else f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8.5)

            ax.set_title(f"H = {H} h  –  {label}", fontsize=11,
                         fontweight="bold")
            ax.set_ylabel(label, fontsize=10)
            ax.set_ylim(0, max(vals) * 1.20)
            ax.tick_params(axis="x", labelsize=9)
            ax.grid(axis="y", alpha=0.3)

            # Highlight best model
            best_idx = np.argmin(vals) if metric != "r2" else np.argmax(vals)
            bars[best_idx].set_edgecolor("black")
            bars[best_idx].set_linewidth(2.5)

    fig.suptitle("Model Performance Comparison — PM2.5 Forecasting",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved → {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# 3. Actual vs Predicted  (H = 1)
# ════════════════════════════════════════════════════════════════════════════
def plot_predictions(y_true, all_preds, scaler, target_idx, n_features,
                     n_plot: int = 200, save_path: str = "predictions_h1.png"):
    """
    Overlay actual and predicted PM2.5 for the first n_plot test samples.
    Inverse-transforms everything to μg/m³.
    """
    def inv(arr):
        dummy = np.zeros((len(arr), n_features))
        dummy[:, target_idx] = arr.ravel()
        return scaler.inverse_transform(dummy)[:, target_idx]

    n = min(n_plot, len(y_true))
    yt = inv(y_true[:n])

    fig, axes = plt.subplots(len(MODEL_ORDER), 1,
                             figsize=(14, 3.5 * len(MODEL_ORDER)),
                             sharex=True)
    x = np.arange(n)

    for ax, name in zip(axes, MODEL_ORDER):
        yp = inv(all_preds[name][:n])
        ax.plot(x, yt, color="black",       linewidth=1.2, label="Actual PM2.5",
                alpha=0.85)
        ax.plot(x, yp, color=PALETTE[name], linewidth=1.2, label=f"{name} Pred",
                alpha=0.80, linestyle="--")
        ax.set_ylabel("PM2.5 (μg/m³)", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(name, fontsize=11, fontweight="bold", color=PALETTE[name])
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Test Sample Index", fontsize=10)
    fig.suptitle("Actual vs Predicted PM2.5 — H = 1 h (first 200 test samples)",
                 fontsize=13, fontweight="bold", y=1.005)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved → {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# 4. RMSE across horizons  (line plot)
# ════════════════════════════════════════════════════════════════════════════
def plot_horizon_comparison(results: dict,
                            save_path: str = "horizon_comparison.png"):
    """Line chart: RMSE vs forecasting horizon for each model."""
    horizons = sorted(list(next(iter(results.values())).keys()))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ["rmse", "mae", "r2"]
    ylabels = ["RMSE (μg/m³)", "MAE (μg/m³)", "R² Score"]

    for ax, metric, ylabel in zip(axes, metrics, ylabels):
        for name in MODEL_ORDER:
            vals = [results[name][H][metric] for H in horizons]
            ax.plot(horizons, vals, marker="o", linewidth=2.2,
                    label=name, color=PALETTE[name])
        ax.set_xlabel("Forecast Horizon (hours)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xticks(horizons)
        ax.grid(alpha=0.3)

    fig.suptitle("Performance vs Forecast Horizon — All Models",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved → {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# 5. Feature Importance (Permutation-Based)
# ════════════════════════════════════════════════════════════════════════════
def plot_feature_importance(model, X_test, y_test, feature_cols,
                             save_path: str = "feature_importance.png"):
    """
    Permutation feature importance: shuffle each feature column across the
    time axis, measure R² drop, rank features by importance.
    """
    from sklearn.metrics import r2_score

    baseline_pred = model.predict(X_test, verbose=0).ravel()
    baseline_r2   = r2_score(y_test, baseline_pred)

    importances = {}
    for i, feat in enumerate(feature_cols):
        X_perm = X_test.copy()
        # Shuffle this feature across all samples and time steps
        perm_idx = np.random.permutation(X_perm.shape[0])
        X_perm[:, :, i] = X_perm[perm_idx, :, i]
        perm_pred = model.predict(X_perm, verbose=0).ravel()
        perm_r2   = r2_score(y_test, perm_pred)
        importances[feat] = max(baseline_r2 - perm_r2, 0)

    # Sort descending
    sorted_feats = sorted(importances, key=importances.get, reverse=True)
    sorted_vals  = [importances[f] for f in sorted_feats]
    colors       = ["#9B59B6" if v == max(sorted_vals) else "#BDC3C7"
                    for v in sorted_vals]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(sorted_feats[::-1], sorted_vals[::-1],
                   color=colors[::-1], edgecolor="white", linewidth=1)
    ax.set_xlabel("R² Score Decrease (Permutation Importance)", fontsize=11)
    ax.set_title("Permutation Feature Importance — Hybrid CNN-LSTM",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for bar, v in zip(bars, sorted_vals[::-1]):
        ax.text(v + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved → {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# 6. Console Summary Table
# ════════════════════════════════════════════════════════════════════════════
def print_summary_table(results: dict):
    """Print a formatted ASCII comparison table to the console."""
    horizons = sorted(list(next(iter(results.values())).keys()))
    sep = "─" * 90

    print("\n" + sep)
    print(f"  {'Model':<12}" +
          "".join([f"  H={H:2d}h RMSE   MAE    R²  " for H in horizons]))
    print(sep)

    for name in MODEL_ORDER:
        row = f"  {name:<12}"
        for H in horizons:
            m = results[name][H]
            row += f"  {m['rmse']:7.2f}  {m['mae']:6.2f}  {m['r2']:.3f} "
        print(row)

    print(sep)
    print("  * Best values per metric/horizon marked by lowest RMSE/MAE "
          "or highest R².")
    print(sep + "\n")
