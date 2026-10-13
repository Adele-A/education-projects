"""Reusable helpers for the technical-indicator analysis.

All data used with these functions is SYNTHETIC (see src/data_generator.py).

Contains:
    - indicator functions (SMA, EMA, RSI, MACD)
    - rule-based position generators (SMA crossover, RSI mean reversion, MACD crossover)
    - a simple long/flat backtest and performance metrics
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def sma(series, window):
    """Simple moving average; NaN until `window` observations are available."""
    return series.rolling(window=window, min_periods=window).mean()


def ema(series, span):
    """Exponential moving average (adjust=False); NaN until `span` observations."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(series, window=14):
    """Relative Strength Index using Wilder smoothing (range 0-100)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss  # avg_loss == 0 gives inf -> RSI of 100
    return 100.0 - 100.0 / (1.0 + rs)


def macd(series, fast=12, slow=26, signal=9):
    """Return a DataFrame with the MACD line, its signal line and the histogram."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": macd_line - signal_line,
        }
    )


# ---------------------------------------------------------------------------
# Signals: each function returns a position series (1 = long, 0 = flat)
# decided at the close of day t. The backtest applies it from day t+1.
# ---------------------------------------------------------------------------
def sma_crossover_position(close, short=20, long=50):
    """Long while the short SMA is above the long SMA."""
    return (sma(close, short) > sma(close, long)).astype(int)


def rsi_reversion_position(close, window=14, entry=30.0, exit_level=50.0):
    """Mean reversion: enter long when RSI < entry, exit when RSI > exit_level."""
    values = rsi(close, window).to_numpy()
    pos = np.zeros(len(values), dtype=int)
    state = 0
    for i, v in enumerate(values):
        if np.isnan(v):
            state = 0
        elif v < entry:
            state = 1
        elif v > exit_level:
            state = 0
        pos[i] = state
    return pd.Series(pos, index=close.index)


def macd_crossover_position(close, fast=12, slow=26, signal=9):
    """Long while the MACD line is above its signal line."""
    m = macd(close, fast, slow, signal)
    return (m["macd"] > m["macd_signal"]).astype(int)


def buy_and_hold_position(close):
    """Always long."""
    return pd.Series(1, index=close.index)


# ---------------------------------------------------------------------------
# Backtest and metrics
# ---------------------------------------------------------------------------
def backtest(log_returns, position, cost_rate=0.001):
    """Backtest a long/flat position series.

    The position decided at the close of day t earns the return of day t+1
    (shift by one day, so there is no look-ahead). Every change of position
    costs `cost_rate` (as a fraction of traded value).

    Returns a DataFrame with the effective position, strategy log return and
    equity curve (starting at 1.0).
    """
    log_returns = log_returns.fillna(0.0)
    effective = position.shift(1).fillna(0).astype(float)
    costs = effective.diff().abs().fillna(effective.abs()) * cost_rate
    strat = effective * log_returns - costs
    return pd.DataFrame(
        {
            "position": effective,
            "strategy_log_return": strat,
            "equity": np.exp(strat.cumsum()),
        }
    )


def count_trades(effective_position):
    """Number of entries into a long position (0 -> 1 transitions)."""
    prev = effective_position.shift(1).fillna(0)
    return int(((effective_position == 1) & (prev == 0)).sum())


def max_drawdown(equity):
    """Maximum peak-to-trough decline of an equity curve (a negative number)."""
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def sharpe_ratio(log_returns, periods=TRADING_DAYS):
    """Annualised Sharpe ratio with a zero risk-free rate."""
    std = log_returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(log_returns.mean() / std * np.sqrt(periods))


def performance_metrics(bt):
    """Summarise a backtest DataFrame into a dict of metrics."""
    r = bt["strategy_log_return"]
    n_years = len(r) / TRADING_DAYS
    total_return = float(bt["equity"].iloc[-1] - 1.0)
    return {
        "total_return": total_return,
        "annualised_return": float(bt["equity"].iloc[-1] ** (1.0 / n_years) - 1.0),
        "annualised_volatility": float(r.std() * np.sqrt(TRADING_DAYS)),
        "sharpe_ratio": sharpe_ratio(r),
        "max_drawdown": max_drawdown(bt["equity"]),
        "n_trades": count_trades(bt["position"]),
        "time_in_market": float(bt["position"].mean()),
    }
