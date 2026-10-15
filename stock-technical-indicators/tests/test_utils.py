"""Tests for the indicator, signal and backtest helpers in src/utils.py.

All inputs are small hand-made or seeded SYNTHETIC series, so the tests are
fast and do not depend on data files or figures.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import utils  # noqa: E402


def make_random_walk(n=300, seed=0):
    """Seeded synthetic close prices."""
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))


def test_sma_known_values():
    result = utils.sma(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    assert result.iloc[:2].isna().all()
    assert result.iloc[2:].tolist() == [2.0, 3.0, 4.0]


def test_ema_constant_series_and_warmup():
    result = utils.ema(pd.Series([10.0] * 20), 5)
    assert result.iloc[:4].isna().all()
    assert np.allclose(result.iloc[4:], 10.0)


def test_rsi_extremes_and_range():
    rising = pd.Series(np.arange(1.0, 60.0))
    falling = pd.Series(np.arange(60.0, 1.0, -1.0))
    assert utils.rsi(rising).iloc[-1] == pytest.approx(100.0)
    assert utils.rsi(falling).iloc[-1] == pytest.approx(0.0)

    values = utils.rsi(make_random_walk()).dropna()
    assert ((values >= 0) & (values <= 100)).all()


def test_macd_columns_and_histogram():
    m = utils.macd(make_random_walk())
    assert list(m.columns) == ["macd", "macd_signal", "macd_hist"]
    valid = m.dropna()
    assert len(valid) > 0
    assert np.allclose(valid["macd_hist"], valid["macd"] - valid["macd_signal"])


def test_positions_are_binary_and_flat_during_warmup():
    close = make_random_walk()
    sma_pos = utils.sma_crossover_position(close)
    assert set(sma_pos.unique()) <= {0, 1}
    assert (sma_pos.iloc[:49] == 0).all()  # SMA(50) needs 50 observations

    rsi_pos = utils.rsi_reversion_position(close)
    assert set(rsi_pos.unique()) <= {0, 1}
    assert (rsi_pos.iloc[:14] == 0).all()

    macd_pos = utils.macd_crossover_position(close)
    assert set(macd_pos.unique()) <= {0, 1}

    assert (utils.buy_and_hold_position(close) == 1).all()


def test_sma_crossover_long_in_steady_uptrend():
    close = pd.Series(np.arange(1.0, 101.0))
    pos = utils.sma_crossover_position(close)
    assert (pos.iloc[49:] == 1).all()


def test_backtest_applies_position_with_one_day_lag_and_costs():
    log_ret = pd.Series([0.01] * 5)
    position = pd.Series([1] * 5)
    cost = 0.001
    bt = utils.backtest(log_ret, position, cost_rate=cost)
    # Position decided on day 0 only earns from day 1 (no look-ahead)
    assert bt["position"].tolist() == [0.0, 1.0, 1.0, 1.0, 1.0]
    # Four days of return and a single entry cost
    assert bt["equity"].iloc[-1] == pytest.approx(np.exp(0.04 - cost))
    assert utils.count_trades(bt["position"]) == 1


def test_max_drawdown_and_sharpe_edge_cases():
    equity = pd.Series([1.0, 2.0, 1.0, 1.5])
    assert utils.max_drawdown(equity) == pytest.approx(-0.5)
    assert utils.sharpe_ratio(pd.Series([0.0, 0.0, 0.0])) == 0.0
    assert utils.sharpe_ratio(pd.Series([0.01, 0.02, 0.03])) > 0
