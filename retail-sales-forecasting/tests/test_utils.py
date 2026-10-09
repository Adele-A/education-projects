"""Tests for the helper functions in src/utils.py.

Small hand-made series are used, so the tests are fast and deterministic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

import utils


def make_series(values, start="2022-01-03"):
    """Build a weekly Monday-indexed series."""
    index = pd.date_range(start, periods=len(values), freq="W-MON")
    return pd.Series(values, index=index, dtype=float)


def test_weekly_category_series_sums_over_stores():
    sales = pd.DataFrame({
        "week_start": pd.to_datetime(["2022-01-03"] * 4 + ["2022-01-10"] * 4),
        "store_id": ["S01", "S01", "S02", "S02"] * 2,
        "category": ["A", "B"] * 4,
        "units_sold": [1, 10, 2, 20, 3, 30, 4, 40],
    })
    wide = utils.weekly_category_series(sales)
    assert list(wide.columns) == ["A", "B"]
    assert len(wide) == 2
    assert wide["A"].tolist() == [3, 7]
    assert wide["B"].tolist() == [30, 70]
    assert wide.index.freqstr == "W-MON"


def test_time_split_keeps_order_and_validates_horizon():
    series = make_series(range(10))
    train, test = utils.time_split(series, 3)
    assert len(train) == 7 and len(test) == 3
    assert train.index.max() < test.index.min()
    with pytest.raises(ValueError):
        utils.time_split(series, 0)
    with pytest.raises(ValueError):
        utils.time_split(series, 10)


def test_naive_and_seasonal_naive_forecasts():
    train = make_series([1, 2, 3, 4, 5, 6, 7, 8])
    assert utils.naive_forecast(train, 3).tolist() == [8.0, 8.0, 8.0]
    # Season of 4: repeats the last four values, wrapping around
    pred = utils.seasonal_naive_forecast(train, 6, season=4)
    assert pred.tolist() == [5, 6, 7, 8, 5, 6]
    with pytest.raises(ValueError):
        utils.seasonal_naive_forecast(make_series([1, 2]), 2, season=4)


def test_error_metrics_known_values():
    actual = [100, 200]
    forecast = [110, 180]
    assert utils.mae(actual, forecast) == pytest.approx(15.0)
    assert utils.rmse(actual, forecast) == pytest.approx(np.sqrt(250))
    assert utils.mape(actual, forecast) == pytest.approx(10.0)
    result = utils.evaluate_forecast(actual, forecast)
    assert set(result) == {"mae", "rmse", "mape_pct"}
    # Zero actuals are skipped in MAPE
    assert utils.mape([0, 100], [5, 110]) == pytest.approx(10.0)
    assert np.isnan(utils.mape([0, 0], [1, 1]))


def test_component_strength_bounds():
    t = np.arange(60)
    component = pd.Series(np.exp(np.sin(t / 5.0)))
    ones = pd.Series(np.ones(60))
    noise = pd.Series(np.exp(np.random.default_rng(0).normal(0, 0.1, 60)))
    # No residual variation: component explains everything
    assert utils.component_strength(component, ones) == pytest.approx(1.0)
    # Constant component: nothing is explained
    assert utils.component_strength(ones, noise) == pytest.approx(0.0, abs=1e-9)
    # Mixed case stays within [0, 1]
    assert 0.0 <= utils.component_strength(component, noise) <= 1.0


def test_holt_winters_forecast_shape_and_validation():
    season = 4
    pattern = np.tile([90, 100, 120, 110], 10)  # 40 weeks, positive values
    trend = np.linspace(1.0, 1.2, 40)
    train = make_series(pattern * trend)
    pred = utils.holt_winters_forecast(train, 6, season=season)
    assert pred.shape == (6,)
    assert np.isfinite(pred).all()
    assert (pred > 0).all()
    with pytest.raises(ValueError):
        utils.holt_winters_forecast(make_series([1.0] * 6), 2, season=4)
