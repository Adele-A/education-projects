"""Reusable helpers for the Delivery Time Analysis project.

Contains data loading/merging, feature preparation and regression metric
functions. Kept free of plotting and printing so it is easy to unit test.
The data is SYNTHETIC (see src/data_generator.py).
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

TARGET = "delivery_time_min"

# Reference categories are dropped in the dummy encoding
WEATHER_LEVELS = ["clear", "cloudy", "rain", "snow"]
VEHICLE_LEVELS = ["bike", "scooter", "car"]

NUMERIC_FEATURES = [
    "distance_km",
    "courier_active_orders",
    "prep_time_min",
    "is_peak_hour",
    "experience_months",
    "avg_rating",
]


def load_merged_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load orders and couriers from CSV and join them on courier_id.

    Args:
        data_dir: Folder containing orders.csv and couriers.csv.

    Returns:
        One row per order with the courier attributes attached.

    Raises:
        FileNotFoundError: If a CSV file is missing.
    """
    orders_path = Path(data_dir) / "orders.csv"
    couriers_path = Path(data_dir) / "couriers.csv"
    if not orders_path.exists() or not couriers_path.exists():
        raise FileNotFoundError(
            "CSV files not found in data/. Run: python src/data_generator.py"
        )
    orders = pd.read_csv(orders_path, parse_dates=["order_datetime"])
    couriers = pd.read_csv(couriers_path)
    return orders.merge(couriers, on="courier_id", how="left", validate="m:1")


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build the model matrix X and target y from the merged table.

    Weather and vehicle type are one-hot encoded with fixed category lists
    (reference levels: clear weather and bike), so the columns are always
    the same regardless of which categories appear in a given sample.

    Args:
        df: Merged orders/couriers table.

    Returns:
        A tuple (X, y) where X is a numeric DataFrame and y the delivery time.

    Raises:
        ValueError: If a weather or vehicle value is unknown.
    """
    unknown_weather = set(df["weather"]) - set(WEATHER_LEVELS)
    unknown_vehicle = set(df["vehicle_type"]) - set(VEHICLE_LEVELS)
    if unknown_weather or unknown_vehicle:
        raise ValueError(
            f"Unknown categories: weather={unknown_weather}, vehicle={unknown_vehicle}"
        )

    X = df[NUMERIC_FEATURES].astype(float).copy()
    for level in WEATHER_LEVELS[1:]:
        X[f"weather_{level}"] = (df["weather"] == level).astype(float)
    for level in VEHICLE_LEVELS[1:]:
        X[f"vehicle_{level}"] = (df["vehicle_type"] == level).astype(float)
    y = df[TARGET].astype(float)
    return X, y


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred) -> float:
    """Coefficient of determination (1 - SS_res / SS_tot).

    Returns 0.0 if the true values are constant (SS_tot is zero).
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return MAE, RMSE and R2 as a dictionary of plain floats."""
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }
