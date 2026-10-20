"""Unit tests for the helper functions in src/utils.py."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils import (  # noqa: E402
    build_exog,
    fourier_terms,
    interval_coverage,
    mape,
    rmse,
    seasonal_naive_forecast,
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


def test_metrics_known_values():
    assert mape([100, 200], [110, 180]) == pytest.approx(10.0)
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0
    assert interval_coverage([1, 5], [0, 0], [2, 2]) == 0.5


def test_build_exog_columns_and_shape():
    dates = pd.date_range("2020-01-06", periods=10, freq="W-MON")
    exog = build_exog(dates, np.zeros(10), np.full(10, 10.0), dates[0], k=2)
    assert list(exog.columns) == ["promo", "temp_c", "t_years", "sin1", "cos1", "sin2", "cos2"]
    assert len(exog) == 10
    assert fourier_terms(dates, 3).shape == (10, 6)
