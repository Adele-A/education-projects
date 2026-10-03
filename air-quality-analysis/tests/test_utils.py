"""Unit tests for src/utils.py (small hand-made frames, no data files needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils import build_features, regression_metrics, time_split  # noqa: E402


def make_df():
    return pd.DataFrame({
        "date": pd.to_datetime(["2023-12-30", "2023-12-31", "2024-01-01", "2024-01-02"]),
        "station_type": ["traffic", "suburban", "industrial", "urban_background"],
        "temperature_c": [-2.0, 20.0, 5.0, 10.0],
        "humidity_pct": [80.0, 50.0, 70.0, 60.0],
        "wind_speed_ms": [2.0, 4.0, 3.0, 5.0],
        "precipitation_mm": [0.0, 5.0, 0.0, 2.0],
        "pressure_hpa": [1010.0, 1015.0, 1000.0, 1020.0],
    })


def test_build_features_columns_and_values():
    X = build_features(make_df())
    assert len(X) == 4
    assert not X.isna().any().any()
    assert list(X["heating_degrees"]) == [17.0, 0.0, 10.0, 5.0]
    assert list(X["rainy"]) == [0.0, 1.0, 0.0, 1.0]
    assert "type_suburban" not in X.columns  # baseline category
    assert X.loc[0, "type_traffic"] == 1.0


def test_time_split_has_no_overlap():
    df = make_df()
    train, test = time_split(df, "2024-01-01")
    assert train.sum() == 2 and test.sum() == 2
    assert df.loc[train, "date"].max() < df.loc[test, "date"].min()


def test_regression_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    m = regression_metrics(y, y)
    assert m["mae"] == 0.0 and m["rmse"] == 0.0 and m["r2"] == 1.0
