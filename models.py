"""
models.py
All model architectures described in Section 4.3 of the paper.

  ┌─────────────────────────────────────────────────────┐
  │  1. build_cnn()       – Standalone 1-D CNN          │
  │  2. build_lstm()      – Standalone Stacked LSTM      │
  │  3. build_cnn_lstm()  – Hybrid CNN-LSTM (proposed)   │
  └─────────────────────────────────────────────────────┘

ARIMA and SVR are handled in train_evaluate.py because they do not
follow the Keras compile/fit pattern.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, LSTM, Dense, Dropout,
    BatchNormalization, Flatten, Input, Reshape,
    TimeDistributed
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


# ════════════════════════════════════════════════════════════════════════════
# 1. Standalone CNN
# ════════════════════════════════════════════════════════════════════════════
def build_cnn(lookback: int, n_features: int, horizon: int = 1) -> Model:
    """
    1-D Convolutional Network for air quality time-series.

    Architecture
    ────────────
    Input (lookback, n_features)
      → Conv1D(64, k=3, ReLU) + BN + Dropout(0.2)
      → Conv1D(128, k=3, ReLU) + BN
      → MaxPool1D(2) + Dropout(0.2)
      → Flatten
      → Dense(64, ReLU) + Dropout(0.2)
      → Dense(32, ReLU)
      → Dense(horizon, linear)
    """
    model = Sequential(name="Standalone_CNN")
    model.add(Input(shape=(lookback, n_features)))

    # Block 1
    model.add(Conv1D(64, kernel_size=3, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.20))

    # Block 2
    model.add(Conv1D(128, kernel_size=3, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.20))

    # Dense head
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.20))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(horizon))           # linear output

    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse",
                  metrics=["mae"])
    return model


# ════════════════════════════════════════════════════════════════════════════
# 2. Standalone LSTM
# ════════════════════════════════════════════════════════════════════════════
def build_lstm(lookback: int, n_features: int, horizon: int = 1) -> Model:
    """
    Stacked LSTM network (Section 4.3.2).

    Architecture
    ────────────
    Input (lookback, n_features)
      → LSTM(128, return_sequences=True) + Dropout(0.30)
      → LSTM(64,  return_sequences=False) + Dropout(0.30)
      → Dense(32, ReLU) + Dropout(0.20)
      → Dense(16, ReLU)
      → Dense(horizon, linear)
    """
    model = Sequential(name="Standalone_LSTM")
    model.add(Input(shape=(lookback, n_features)))

    model.add(LSTM(128, return_sequences=True,
                   recurrent_dropout=0.0,
                   dropout=0.30))
    model.add(LSTM(64,  return_sequences=False,
                   recurrent_dropout=0.0,
                   dropout=0.30))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.20))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(horizon))

    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse",
                  metrics=["mae"])
    return model


# ════════════════════════════════════════════════════════════════════════════
# 3. Hybrid CNN-LSTM  (Proposed Model)
# ════════════════════════════════════════════════════════════════════════════
def build_cnn_lstm(lookback: int, n_features: int, horizon: int = 1) -> Model:
    """
    Hybrid CNN-LSTM as proposed in Section 4.3.

    CNN sub-network extracts spatial/cross-variable feature patterns;
    LSTM sub-network learns long-range temporal dependencies on the
    CNN-enriched representations.

    Architecture
    ────────────
    Input (lookback, n_features)
      ── CNN sub-network ──────────────────────────────────
      → Conv1D(64,  k=3, ReLU) + BN + Dropout(0.20)
      → Conv1D(128, k=3, ReLU) + BN
      → MaxPool1D(2) + Dropout(0.20)
      ── LSTM sub-network ─────────────────────────────────
      → LSTM(128, return_sequences=True, dropout=0.30)
      → LSTM(64,  return_sequences=False, dropout=0.30)
      ── Regression head ──────────────────────────────────
      → Dense(32, ReLU) + Dropout(0.20)
      → Dense(16, ReLU) + Dropout(0.20)
      → Dense(horizon, linear)

    Parameters: ~312 000 (for lookback=24, n_features=9, horizon=1)
    """
    inputs = Input(shape=(lookback, n_features), name="Input")

    # ── CNN sub-network ───────────────────────────────────────────────────
    x = Conv1D(64, kernel_size=3, padding="same",
               activation="relu", name="Conv1")(inputs)
    x = BatchNormalization(name="BN1")(x)
    x = Dropout(0.20, name="Drop_CNN1")(x)

    x = Conv1D(128, kernel_size=3, padding="same",
               activation="relu", name="Conv2")(x)
    x = BatchNormalization(name="BN2")(x)
    x = MaxPooling1D(pool_size=2, name="MaxPool")(x)
    x = Dropout(0.20, name="Drop_CNN2")(x)

    # ── LSTM sub-network ──────────────────────────────────────────────────
    x = LSTM(128, return_sequences=True,  dropout=0.30, name="LSTM1")(x)
    x = LSTM(64,  return_sequences=False, dropout=0.30, name="LSTM2")(x)

    # ── Regression head ───────────────────────────────────────────────────
    x = Dense(32, activation="relu", name="Dense1")(x)
    x = Dropout(0.20, name="Drop_Dense1")(x)
    x = Dense(16, activation="relu", name="Dense2")(x)
    x = Dropout(0.20, name="Drop_Dense2")(x)
    outputs = Dense(horizon, name="Output")(x)

    model = Model(inputs, outputs, name="Hybrid_CNN_LSTM")
    model.compile(optimizer=Adam(learning_rate=1e-3,
                                 decay=0.0),         # decay handled by callback
                  loss="mse",
                  metrics=["mae"])
    return model


# ════════════════════════════════════════════════════════════════════════════
# Utility: print model summaries
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    lookback, n_features, horizon = 24, 9, 1
    print("\n── Standalone CNN ────────────────────────────────────────────")
    build_cnn(lookback, n_features, horizon).summary()
    print("\n── Standalone LSTM ───────────────────────────────────────────")
    build_lstm(lookback, n_features, horizon).summary()
    print("\n── Hybrid CNN-LSTM ───────────────────────────────────────────")
    build_cnn_lstm(lookback, n_features, horizon).summary()
