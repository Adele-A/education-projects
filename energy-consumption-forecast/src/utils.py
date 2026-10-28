"""Reusable helpers for the SYNTHETIC household energy forecast.

Contains feature preparation, a chronological train/test split and metric
functions. They are kept free of printing and file output so they can be
unit tested.
"""
import numpy as np
import pandas as pd

# Forecast set-up: predict every hour of day D using information from day D-1
# and earlier (lags of 24 h and 168 h), plus calendar and temperature inputs.
LAG_DAY = 24
LAG_WEEK = 168
HEATING_THRESHOLD_C = 16.0  # below this temperature electric heating switches on

FEATURE_COLUMNS = [
    "lag_24", "lag_168", "prev_day_mean",
    "hour", "hour_sin", "hour_cos", "day_of_week", "is_weekend",
    "temperature_c", "cold_degrees", "heating_x_cold",
    "n_residents", "floor_area_m2", "has_electric_heating",
]
TARGET = "kwh"


def build_dataset(households: pd.DataFrame, weather: pd.DataFrame,
                  energy: pd.DataFrame) -> pd.DataFrame:
    """Join the three tables, sorted by household and time."""
    df = energy.merge(households, on="household_id", how="left")
    df = df.merge(weather, on="timestamp", how="left")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["household_id", "timestamp"]).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, temperature and lag features.

    Lags use row shifts inside each household, which is valid because every
    household has a complete hourly series (checked in the EDA step).
    The temperature column is used as if it were a weather forecast for the
    day being predicted.
    """
    df = df.sort_values(["household_id", "timestamp"]).reset_index(drop=True).copy()
    df["date"] = df["timestamp"].dt.normalize()
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["cold_degrees"] = (HEATING_THRESHOLD_C - df["temperature_c"]).clip(lower=0)
    df["heating_x_cold"] = df["cold_degrees"] * df["has_electric_heating"]

    grouped = df.groupby("household_id")["kwh"]
    df["lag_24"] = grouped.shift(LAG_DAY)
    df["lag_168"] = grouped.shift(LAG_WEEK)
    # Mean of the previous calendar day: the daily mean shifted back by 24 rows
    daily_mean = df.groupby(["household_id", "date"])["kwh"].transform("mean")
    df["prev_day_mean"] = daily_mean.groupby(df["household_id"]).shift(LAG_DAY)
    return df


def drop_incomplete(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where all lag features exist (drops the first week)."""
    return df.dropna(subset=["lag_24", "lag_168", "prev_day_mean"]).reset_index(drop=True)


def chronological_split(df: pd.DataFrame, test_days: int):
    """Split by time: the last `test_days` calendar days form the test set."""
    last_day = df["date"].max()
    cutoff = last_day - pd.Timedelta(days=test_days - 1)
    train = df[df["date"] < cutoff].reset_index(drop=True)
    test = df[df["date"] >= cutoff].reset_index(drop=True)
    return train, test


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error: sum|error| / sum(actual).

    Raises ValueError if the total actual consumption is zero, because the
    ratio is then undefined.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    total_actual = np.sum(y_true)
    if total_actual == 0:
        raise ValueError("WAPE is undefined when the total actual consumption is zero")
    return float(np.sum(np.abs(y_true - y_pred)) / total_actual)


def r2(y_true, y_pred) -> float:
    """Coefficient of determination.

    Raises ValueError if the actual values are constant, because the total
    sum of squares is then zero and R2 is undefined.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        raise ValueError("R2 is undefined when all actual values are identical")
    return float(1 - ss_res / ss_tot)


def evaluate(y_true, y_pred) -> dict:
    """All metrics in one dictionary."""
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "wape": wape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


def improvement_pct(baseline_value: float, model_value: float) -> float:
    """Relative error reduction of a model versus a baseline, in percent.

    Raises ValueError if the baseline value is zero, because the reduction
    is then undefined.
    """
    if baseline_value == 0:
        raise ValueError("improvement is undefined for a zero baseline value")
    return float((baseline_value - model_value) / baseline_value * 100)
