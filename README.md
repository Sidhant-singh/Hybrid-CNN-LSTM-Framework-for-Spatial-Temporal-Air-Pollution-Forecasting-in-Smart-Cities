# 🌫️ Hybrid CNN-LSTM Framework for Spatial-Temporal Air Pollution Forecasting in Smart Cities

<div align="center">

![Status](https://img.shields.io/badge/Status-Accepted%20at%20Conference-brightgreen?style=for-the-badge)
![Type](https://img.shields.io/badge/Type-Research%20Paper-blue?style=for-the-badge)
![Domain](https://img.shields.io/badge/Domain-AI%20%7C%20Environmental%20Science%20%7C%20Healthcare-purple?style=for-the-badge)

</div>

---

## 📄 Project Overview

| Field | Details |
|---|---|
| **Project Title** | Hybrid CNN-LSTM Framework for Spatial-Temporal Air Pollution Forecasting in Smart Cities |
| **Type** | Research Paper |
| **Domain** | Artificial Intelligence · Environmental Science · Healthcare |
| **Dataset** | Air Pollution in India (Kaggle) |
| **Current Status** | ✅ **Accepted by Conference** |

---

## 👥 Team Details

| # | Name | Roll Number | Role |
|---|---|---|---|
| 1 | **Karan Arora** | 2210991749 | Team Member |
| 2 | **Sidhant Singh Bhadauriya** | 2210992377 | Team Member |
| 3 | **Harsh Kumar Sahu** | 2210991620 | Team Member |
| 4 | **Bhargav Singla** | 2210990214 | Team Member |

---

## 🏆 Current Status

> **✅ Accepted by Conference**

The research paper has been **officially accepted** for presentation and publication at a peer-reviewed conference. This represents the successful culmination of the team's research work in applying hybrid deep learning techniques to the problem of urban air pollution forecasting.

---

## 🔬 Abstract

Air pollution forecasting in smart cities demands models capable of handling both spatial cross-variable interactions and long-range temporal dynamics. This research proposes a **Hybrid CNN-LSTM** architecture that:

- Uses **Convolutional Neural Networks (CNN)** to extract spatial feature patterns from multi-pollutant input data
- Uses **Long Short-Term Memory (LSTM)** networks to model sequential temporal dependencies
- Outperforms standalone ARIMA, SVR, CNN, and LSTM baselines across all forecasting horizons (1h, 6h, 24h)

**Key Results:**

| Metric | CNN-LSTM (Proposed) | Best Baseline |
|---|---|---|
| RMSE | **12.47 μg/m³** | 16.85 (LSTM) |
| MAE | **9.83 μg/m³** | 13.24 (LSTM) |
| R² Score | **0.941** | 0.903 (LSTM) |

---

## 🗂️ Repository Structure

```
air_pollution_forecasting/
│
├── 📄 README.md                  ← You are here
├── 📄 requirements.txt           ← Python dependencies
│
├── 🐍 main.py                    ← Master pipeline runner
├── 🐍 demo_no_tf.py              ← Lightweight demo (no TensorFlow needed)
│
├── 🐍 data_generator.py          ← Synthetic dataset generator
├── 🐍 preprocessing.py           ← Data cleaning, imputation, normalisation
├── 🐍 window_builder.py          ← Sliding window sequence builder
├── 🐍 models.py                  ← CNN, LSTM, CNN-LSTM architectures
├── 🐍 train_evaluate.py          ← Training loops + ARIMA/SVR + metrics
└── 🐍 visualise.py               ← Charts, plots, feature importance
```

---

## 🧠 Models Implemented

| Model | Type | Description |
|---|---|---|
| **ARIMA** | Statistical Baseline | Auto-Regressive Integrated Moving Average (2,1,2) |
| **SVR** | ML Baseline | Support Vector Regression with RBF kernel |
| **Standalone CNN** | Deep Learning | 1D Convolutional Network (64/128 filters) |
| **Standalone LSTM** | Deep Learning | Stacked LSTM (128/64 units) with recurrent dropout |
| **CNN-LSTM** ⭐ | **Proposed Hybrid** | CNN spatial extractor + LSTM temporal modeler |

---

## ⚙️ Methodology

```
Raw Data (PM2.5, PM10, NO2, SO2, CO, O3, Temp, Humidity, Wind)
    │
    ▼
Preprocessing
    ├── Linear interpolation  (short gaps ≤ 3 steps)
    ├── KNN Imputation        (longer gaps, k=5)
    ├── IQR×3 Outlier removal
    └── Min-Max Normalisation (fit on train only)
    │
    ▼
Sliding Window Builder  (lookback=24h, stride=1)
    │
    ▼
    ┌─────────────────────────────────────────────┐
    │           Hybrid CNN-LSTM Model             │
    │                                             │
    │  Input (24, 9)                              │
    │    → Conv1D(64, k=3) + BN + Dropout(0.2)   │
    │    → Conv1D(128,k=3) + BN + MaxPool         │
    │    → LSTM(128, return_seq=True)             │
    │    → LSTM(64,  return_seq=False)            │
    │    → Dense(32) → Dense(16) → Output         │
    └─────────────────────────────────────────────┘
    │
    ▼
Evaluation: RMSE · MAE · R²  (Horizons: 1h · 6h · 24h)
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install tensorflow statsmodels scikit-learn pandas numpy matplotlib
```

### 2. Run Full Pipeline
```bash
python main.py
```

### 3. Run Lightweight Demo (no TensorFlow required)
```bash
python demo_no_tf.py
```

### 4. Use Real Kaggle Dataset
1. Download from [Kaggle — Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
2. Open `main.py` and set:
```python
DATA_PATH = "path/to/city_day.csv"
```

---

## 📊 Output Files

After running the pipeline, the `outputs/` folder contains:

| File | Description |
|---|---|
| `comparison_bar.png` | RMSE / MAE / R² grouped bar charts for all models |
| `predictions_h1.png` | Actual vs Predicted PM2.5 overlay (H = 1h) |
| `horizon_comparison.png` | Performance vs forecasting horizon line plot |
| `loss_curves.png` | Training vs validation loss curves (deep models) |
| `feature_importance.png` | Permutation-based feature importance ranking |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-API-D00000?style=flat&logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.1+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.23+-013243?style=flat&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.5+-150458?style=flat&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.6+-11557C?style=flat)

---

## 📚 Key References

1. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780.
2. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature, 521*, 436–444.
3. Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR 2015*.
4. WHO (2021). Global Air Quality Guidelines. World Health Organization Press.

---

## 📬 Contact

For queries related to this research, please reach out to any team member via your institution's official communication channels.

---

<div align="center">

**© 2024 — Karan Arora · Sidhant Singh Bhadauriya · Harsh Kumar Sahu · Bhargav Singla**

*Research Paper | Conference Accepted | AI & Environmental Science*

</div>
