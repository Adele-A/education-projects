"""Unit tests for the helper functions in src/utils.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from utils import (  # noqa: E402
    build_exog,
    forecast_metrics,
    fourier_terms,
    interval_coverage,
    mape,
    rmse,
    seasonal_naive_forecast,
    seasonal_profile,
    temperature_climatology,
    time_split,
)


def test_seasonal_naive_repeats_last_cycle():
    y = np.arange(104)
    fc = seasonal_naive_forecast(y, 52)
    assert np.array_equal(fc, y[-52:])
    assert len(seasonal_naive_forecast(y, 60)) == 60


def test_seasonal_naive_requires_full_season():
    with pytest.raises(ValueError):
        seasonal_naive_forecast(np.arange(10), 5)


def test_time_split_has_no_overlap():
    dates = pd.date_range("2020-01-06", periods=20, freq="W-MON")
    df = pd.DataFrame({"week_start": dates, "x": range(20)})
    train, test = time_split(df, 5)
    assert len(test) == 5 and len(train) == 15
    assert train["week_start"].max() < test["week_start"].min()


def test_time_split_rejects_invalid_test_size():
    dates = pd.date_range("2020-01-06", periods=10, freq="W-MON")
    df = pd.DataFrame({"week_start": dates})
    with pytest.raises(ValueError):
        time_split(df, 0)
    with pytest.raises(ValueError):
        time_split(df, 10)


def test_metrics_known_values():
    assert mape([100, 200], [110, 180]) == pytest.approx(10.0)
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0
    assert interval_coverage([1, 5], [0, 0], [2, 2]) == 0.5
    m = forecast_metrics([10, 20], [12, 18])
    assert set(m) == {"mae", "rmse", "mape"}
    assert m["mae"] == pytest.approx(2.0)


def test_build_exog_columns_and_shape():
    dates = pd.date_range("2020-01-06", periods=10, freq="W-MON")
    exog = build_exog(dates, np.zeros(10), np.full(10, 10.0), dates[0], k=2)
    assert list(exog.columns) == ["promo", "temp_c", "t_years", "sin1", "cos1", "sin2", "cos2"]
    assert len(exog) == 10
    assert fourier_terms(dates, 3).shape == (10, 6)


def test_temperature_climatology_constant_series():
    train_dates = pd.date_range("2020-01-06", periods=104, freq="W-MON")
    train = pd.DataFrame({"week_start": train_dates, "avg_temp_c": 10.0})
    target = pd.date_range("2022-01-03", periods=12, freq="W-MON")
    clim = temperature_climatology(train, target)
    assert len(clim) == 12
    assert np.allclose(clim, 10.0)


def test_seasonal_profile_detects_known_peak():
    # Clean yearly cycle peaking around day 20 (January), trough around July
    rng = np.random.default_rng(0)
    dates = pd.date_range("2019-01-07", periods=208, freq="W-MON")
    angle = 2 * np.pi * (dates.dayofyear.to_numpy() - 20) / 365.25
    values = 1000 * np.exp(0.5 * np.cos(angle) + rng.normal(0, 0.01, len(dates)))
    prof = seasonal_profile(pd.Series(values, index=dates))
    assert prof["strength"] > 0.8
    assert prof["peak_to_trough"] > 1.5
    assert prof["peak_month"] in (12, 1, 2)
    assert prof["trough_month"] in (6, 7, 8)
