"""Tests for the helper functions in src/utils.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

import utils


def make_merged() -> pd.DataFrame:
    """Build a tiny merged table covering several weather/vehicle values."""
    return pd.DataFrame({
        "distance_km": [1.0, 2.5, 4.0, 6.0],
        "courier_active_orders": [1, 2, 3, 4],
        "prep_time_min": [10.0, 12.0, 8.0, 15.0],
        "is_peak_hour": [0, 1, 0, 1],
        "experience_months": [5, 20, 40, 80],
        "avg_rating": [4.0, 4.5, 4.2, 4.8],
        "weather": ["clear", "rain", "snow", "cloudy"],
        "vehicle_type": ["bike", "scooter", "car", "bike"],
        "delivery_time_min": [20.0, 30.0, 35.0, 45.0],
    })


def test_prepare_features_columns_and_encoding():
    X, y = utils.prepare_features(make_merged())
    expected_cols = utils.NUMERIC_FEATURES + [
        "weather_cloudy", "weather_rain", "weather_snow",
        "vehicle_scooter", "vehicle_car",
    ]
    assert list(X.columns) == expected_cols
    assert len(X) == len(y) == 4
    # reference levels (clear, bike) have all dummies equal to zero
    assert X.loc[0, ["weather_cloudy", "weather_rain", "weather_snow"]].sum() == 0
    assert X.loc[0, ["vehicle_scooter", "vehicle_car"]].sum() == 0
    assert X.loc[1, "weather_rain"] == 1.0
    assert X.loc[2, "vehicle_car"] == 1.0
    assert y.tolist() == [20.0, 30.0, 35.0, 45.0]


def test_prepare_features_rejects_unknown_category():
    df = make_merged()
    df.loc[0, "weather"] = "hail"
    with pytest.raises(ValueError):
        utils.prepare_features(df)


def test_error_metrics_known_values():
    y_true = [10.0, 20.0, 30.0]
    y_pred = [12.0, 18.0, 33.0]
    assert utils.mae(y_true, y_pred) == pytest.approx((2 + 2 + 3) / 3)
    assert utils.rmse(y_true, y_pred) == pytest.approx(np.sqrt((4 + 4 + 9) / 3))


def test_r2_perfect_mean_and_constant_target():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert utils.r2(y, y) == pytest.approx(1.0)
    assert utils.r2(y, np.full(4, y.mean())) == pytest.approx(0.0)
    # constant true values: SS_tot is zero, function returns 0.0
    assert utils.r2([5.0, 5.0, 5.0], [4.0, 5.0, 6.0]) == 0.0


def test_regression_metrics_keys():
    result = utils.regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
    assert set(result) == {"mae", "rmse", "r2"}
    assert all(isinstance(v, float) for v in result.values())


def test_load_merged_data_joins_couriers(tmp_path):
    pd.DataFrame({
        "courier_id": [1, 2],
        "vehicle_type": ["bike", "car"],
        "experience_months": [10, 50],
        "avg_rating": [4.1, 4.6],
    }).to_csv(tmp_path / "couriers.csv", index=False)
    pd.DataFrame({
        "order_id": [1, 2, 3],
        "courier_id": [1, 2, 2],
        "order_datetime": ["2025-01-01 10:00:00"] * 3,
        "delivery_time_min": [20.0, 25.0, 30.0],
    }).to_csv(tmp_path / "orders.csv", index=False)

    merged = utils.load_merged_data(tmp_path)
    assert len(merged) == 3
    assert merged["vehicle_type"].tolist() == ["bike", "car", "car"]
    assert pd.api.types.is_datetime64_any_dtype(merged["order_datetime"])


def test_load_merged_data_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        utils.load_merged_data(tmp_path)
