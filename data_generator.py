"""
data_generator.py
Generates a synthetic air pollution dataset that mirrors the structure of
the "Air Pollution in India" Kaggle dataset so the pipeline is fully
self-contained and runnable without a Kaggle account.

The signals include:
  - Realistic diurnal cycles (morning / evening rush-hour peaks)
  - Seasonal variation (winter smog intensification)
  - Cross-pollutant correlations (PM2.5 ↔ PM10, CO ↔ NO2)
  - Meteorological covariation (wind disperses PM, humidity raises PM)
  - ~8% random missing values (for imputation testing)
"""

import numpy as np
import pandas as pd


def generate_synthetic_dataset(n_hours: int = 17520, seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame with n_hours rows, hourly timestamped."""
    rng = np.random.default_rng(seed)
    t   = np.arange(n_hours)

    # ── Meteorological signals ────────────────────────────────────────────
    # Temperature: seasonal + diurnal  (°C)
    temp = (20
            + 12 * np.sin(2 * np.pi * t / (24 * 365))        # seasonal
            +  6 * np.sin(2 * np.pi * t / 24 - 2)            # diurnal peak ~14:00
            + rng.normal(0, 1.5, n_hours))

    # Humidity: inverse seasonal, higher in monsoon  (%)
    humidity = (65
                - 20 * np.sin(2 * np.pi * t / (24 * 365))
                +  8 * np.sin(2 * np.pi * t / 24 + 1)
                + rng.normal(0, 4, n_hours)).clip(20, 99)

    # Wind speed: higher in summer, lower in winter  (m/s)
    wind = (3.5
            + 1.5 * np.sin(2 * np.pi * t / (24 * 365) + np.pi)
            + rng.exponential(0.8, n_hours)).clip(0.1, 20)

    # ── Base pollution from meteorology ───────────────────────────────────
    dispersion = 1 / (wind + 0.5)           # high wind → low pollution
    winter     = 0.5 * (1 - np.sin(2 * np.pi * t / (24 * 365)))

    # Diurnal emission pattern (rush hours 08:00 and 18:00)
    hour_of_day = t % 24
    diurnal = (1.0
               + 0.6 * np.exp(-0.5 * ((hour_of_day - 8)  / 1.5) ** 2)
               + 0.5 * np.exp(-0.5 * ((hour_of_day - 18) / 1.5) ** 2))

    # ── Pollutant concentrations ──────────────────────────────────────────
    base = 60 * dispersion * (1 + 0.4 * winter) * diurnal

    PM25 = (base * 0.9
            + 0.3 * humidity
            + rng.normal(0, 5, n_hours)).clip(1, 500)

    PM10 = (PM25 * 1.6
            + rng.normal(0, 8, n_hours)).clip(1, 800)

    NO2  = (base * 0.5
            + rng.normal(0, 3, n_hours)).clip(1, 200)

    SO2  = (base * 0.25
            + rng.normal(0, 2, n_hours)).clip(1, 100)

    CO   = (NO2 * 0.8
            + rng.normal(0, 1, n_hours)).clip(0.1, 50)

    # Ozone: photochemical; peaks in summer afternoons
    O3   = (40
            + 20 * np.sin(2 * np.pi * t / (24 * 365))
            + 15 * np.sin(2 * np.pi * t / 24 - np.pi / 2)
            - 0.3 * NO2
            + rng.normal(0, 4, n_hours)).clip(1, 200)

    # ── Build DataFrame ───────────────────────────────────────────────────
    dates = pd.date_range("2020-01-01", periods=n_hours, freq="h")
    df = pd.DataFrame({
        "Date"       : dates,
        "PM2.5"      : PM25,
        "PM10"       : PM10,
        "NO2"        : NO2,
        "SO2"        : SO2,
        "CO"         : CO,
        "O3"         : O3,
        "Temperature": temp,
        "Humidity"   : humidity,
        "Wind_Speed" : wind,
    })

    # ── Inject ~8% missing values (non-random) ────────────────────────────
    for col in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3",
                "Temperature", "Humidity", "Wind_Speed"]:
        mask = rng.random(n_hours) < 0.08
        df.loc[mask, col] = np.nan

    return df
