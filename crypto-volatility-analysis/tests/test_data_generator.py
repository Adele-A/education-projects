"""Tests for the SYNTHETIC data generator (src/data_generator.py).

The tests build the tables in memory with build_tables(), so they do not
read or write any files and run in well under a second.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

import data_generator as dg


@pytest.fixture(scope="module")
def tables() -> dict:
    """Generate the tables once for all tests in this module."""
    return dg.build_tables()


def test_columns_and_row_counts(tables):
    """Each table has the documented columns and the expected number of rows."""
    n_assets = len(dg.ASSETS)
    assert list(tables["assets"].columns) == [
        "asset", "asset_name", "category", "start_price_usd"
    ]
    assert list(tables["market_regimes"].columns) == ["date", "regime"]
    assert list(tables["daily_prices"].columns) == [
        "date", "asset", "close_usd", "log_return", "volume_usd"
    ]
    assert len(tables["assets"]) == n_assets
    assert len(tables["market_regimes"]) == dg.N_DAYS
    assert len(tables["daily_prices"]) == dg.N_DAYS * n_assets


def test_value_ranges_and_keys(tables):
    """Prices and volumes are positive, labels are valid, keys are unique."""
    prices = tables["daily_prices"]
    regimes = tables["market_regimes"]

    assert not prices.isna().any().any()
    assert (prices["close_usd"] > 0).all()
    assert (prices["volume_usd"] > 0).all()
    assert np.isfinite(prices["log_return"]).all()
    assert set(regimes["regime"]) <= {"calm", "turbulent"}
    assert not prices.duplicated(["date", "asset"]).any()
    assert regimes["date"].is_unique
    assert set(prices["asset"]) == set(tables["assets"]["asset"])
    assert set(prices["date"]) == set(regimes["date"])


def test_generation_is_reproducible():
    """The fixed seed gives identical tables on every call."""
    first = dg.build_tables()
    second = dg.build_tables()
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])


def test_first_close_matches_start_price_and_return(tables):
    """First close = start price * exp(first log return), up to rounding."""
    prices = tables["daily_prices"]
    first_day = prices["date"].min()
    first = prices[prices["date"] == first_day].set_index("asset")
    start = tables["assets"].set_index("asset")["start_price_usd"]
    expected = start.loc[first.index] * np.exp(first["log_return"])
    np.testing.assert_allclose(first["close_usd"], expected, rtol=1e-3)
