"""Reusable helpers for the air quality analysis (SYNTHETIC data).

Contains data loading, feature preparation, the time-based split and metric
functions, so they can be unit tested separately from the analysis script.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

POLLUTANTS = ["pm25", "pm10", "no2", "o3"]
WEATHER = ["temperature_c", "humidity_pct", "wind_speed_ms", "precipitation_mm", "pressure_hpa"]

# Fixed category list so train and test always get identical dummy columns.
# "suburban" is the baseline category (no dummy column).
STATION_TYPES = ["suburban", "traffic", "urban_background", "industrial"]


def load_data(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Load both CSV tables and return measurements merged with station metadata.

    Args:
        data_dir: Path to the data directory containing stations.csv and daily_measurements.csv.

    Returns:
        DataFrame with measurements and station metadata merged on station_id.

    Raises:
        FileNotFoundError: If data files do not exist in data_dir.
    """
    data_dir = Path(data_dir)
    stations_path = data_dir / "stations.csv"
    meas_path = data_dir / "daily_measurements.csv"
    if not stations_path.exists() or not meas_path.exists():
        raise FileNotFoundError(
            f"Data files not found in {data_dir}. Run: python src/data_generator.py"
        )
    stations = pd.read_csv(stations_path)
    meas = pd.read_csv(meas_path, parse_dates=["date"])
    return meas.merge(stations, on="station_id", how="left")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model feature matrix from merged measurements.

    Features: raw weather variables, heating degrees (how far the temperature is
    below 15 C), a rainy-day flag, a weekend flag, cyclic month encoding and
    station type dummies.

    Args:
        df: DataFrame with measurements and station metadata.

    Returns:
        Feature matrix with index matching df.
    """
    X = pd.DataFrame(index=df.index)
    for col in WEATHER:
        X[col] = df[col]
    X["heating_degrees"] = (15.0 - df["temperature_c"]).clip(lower=0)
    X["rainy"] = (df["precipitation_mm"] > 1.0).astype(float)
    X["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(float)
    month = df["date"].dt.month
    X["month_sin"] = np.sin(2 * np.pi * month / 12)
    X["month_cos"] = np.cos(2 * np.pi * month / 12)
    for t in STATION_TYPES[1:]:
        X[f"type_{t}"] = (df["station_type"] == t).astype(float)
    return X


def prepare_model_data(
    df: pd.DataFrame, target: str = "pm25"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Drop rows with a missing target or features; return (rows, X, y).

    Args:
        df: DataFrame with measurements and station metadata.
        target: Name of the target column (typically "pm25").

    Returns:
        Tuple of (filtered measurements DataFrame, feature matrix, target Series).
    """
    X = build_features(df)
    mask = df[target].notna() & X.notna().all(axis=1)
    return df.loc[mask], X.loc[mask], df.loc[mask, target]


def time_split(df: pd.DataFrame, cutoff: str) -> tuple[pd.Series, pd.Series]:
    """Split by date: rows before the cutoff are train, the rest are test.

    Args:
        df: DataFrame with a 'date' column.
        cutoff: Cutoff date as a string in YYYY-MM-DD format.

    Returns:
        Tuple of (train_mask, test_mask) as boolean Series.
    """
    cutoff = pd.Timestamp(cutoff)
    train_mask = df["date"] < cutoff
    return train_mask, ~train_mask


def regression_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series
) -> dict[str, float]:
    """Compute regression metrics (MAE, RMSE, R2).

    Args:
        y_true: True target values.
        y_pred: Predicted target values.

    Returns:
        Dictionary with keys 'mae', 'rmse', 'r2' and float values.
    """
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def smearing_factor(log_residuals: np.ndarray | pd.Series) -> float:
    """Compute the Duan smearing factor for log-scale bias correction.

    The smearing factor is the mean of exp(log_residuals) and is used to adjust
    predictions made on the log scale when back-transformed to the original scale,
    correcting for Jensen's inequality bias.

    Args:
        log_residuals: Residuals on the log scale (log(actual) - log(predicted)).

    Returns:
        Smearing factor as a float; equals 1.0 if residuals are zero.
    """
    return float(np.mean(np.exp(np.asarray(log_residuals))))
