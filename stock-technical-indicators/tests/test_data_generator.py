"""Tests for the synthetic data generator (src/data_generator.py).

All data is SYNTHETIC. The tests call generate_data() directly in memory,
so they do not need any files on disk and do not depend on figures.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import data_generator  # noqa: E402


@pytest.fixture(scope="module")
def generated():
    """Generate the tables once and share them between the tests."""
    return data_generator.generate_data()


def test_columns(generated):
    tickers, prices = generated
    assert list(tickers.columns) == [
        "ticker", "sector", "annual_drift", "annual_volatility", "start_price",
    ]
    assert list(prices.columns) == [
        "date", "ticker", "open", "high", "low", "close", "volume", "volatility_regime",
    ]


def test_row_counts(generated):
    tickers, prices = generated
    n_tickers = len(data_generator.TICKER_PARAMS)
    assert len(tickers) == n_tickers
    assert len(prices) == n_tickers * data_generator.N_DAYS
    counts = prices.groupby("ticker").size()
    assert (counts == data_generator.N_DAYS).all()
    assert not prices.duplicated(["date", "ticker"]).any()
    assert not prices.isna().any().any()


def test_price_values_are_consistent(generated):
    _, prices = generated
    ohlc = prices[["open", "high", "low", "close"]]
    assert (ohlc > 0).all().all()
    assert (prices["high"] >= ohlc.max(axis=1)).all()
    assert (prices["low"] <= ohlc.min(axis=1)).all()


def test_volume_and_regime_ranges(generated):
    _, prices = generated
    assert (prices["volume"] >= 1).all()
    assert set(prices["volatility_regime"].unique()) <= {"calm", "stressed"}


def test_ticker_parameter_ranges(generated):
    tickers, _ = generated
    assert tickers["annual_drift"].between(0.05, 0.12).all()
    assert tickers["annual_volatility"].between(0.16, 0.32).all()
    assert tickers["start_price"].between(45.0, 100.0).all()


def test_generation_is_reproducible(generated):
    tickers_a, prices_a = generated
    tickers_b, prices_b = data_generator.generate_data()
    pd.testing.assert_frame_equal(tickers_a, tickers_b)
    pd.testing.assert_frame_equal(prices_a, prices_b)
