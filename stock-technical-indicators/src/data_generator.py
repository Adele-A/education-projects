"""Generate synthetic daily stock prices for the technical-indicators project.

All data is SYNTHETIC. Prices follow a geometric random walk with fat-tailed
(Student-t) shocks and a two-state volatility regime (calm / stressed).

Outputs (relative to the project root):
    data/tickers.csv  - one row per fictional stock
    data/prices.csv   - daily OHLCV rows for every stock

Run from the project root:
    python src/data_generator.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

SEED = 42
N_DAYS = 1000                 # business days per ticker (4 tickers -> 4000 rows)
START_DATE = "2021-01-04"
TRADING_DAYS = 252

# Volatility regime: probability of switching state on any given day
P_CALM_TO_STRESS = 0.02
P_STRESS_TO_CALM = 0.08
STRESS_VOL_MULTIPLIER = 2.0

# Fictional stocks and their assumed annual parameters
TICKER_PARAMS = [
    # ticker, sector, annual_drift, annual_vol, start_price, base_volume
    ("ALFA", "Technology", 0.12, 0.32, 100.0, 2_000_000),
    ("BRVO", "Utilities", 0.05, 0.16, 60.0, 1_200_000),
    ("CHRL", "Energy", 0.08, 0.28, 45.0, 1_800_000),
    ("DLTA", "Healthcare", 0.10, 0.22, 80.0, 1_500_000),
]


def simulate_ticker(rng, dates, annual_drift, annual_vol, start_price, base_volume):
    """Simulate OHLCV data for one ticker and return it as a DataFrame."""
    n = len(dates)
    daily_drift = annual_drift / TRADING_DAYS
    daily_vol = annual_vol / np.sqrt(TRADING_DAYS)

    # Student-t shocks (df=5) rescaled to unit variance give fat tails
    df = 5
    shocks = rng.standard_t(df, size=n) / np.sqrt(df / (df - 2))

    # Two-state Markov chain for the volatility regime
    stressed = np.zeros(n, dtype=bool)
    state = False
    switch_draws = rng.random(n)
    for i in range(n):
        if state:
            state = not (switch_draws[i] < P_STRESS_TO_CALM)
        else:
            state = switch_draws[i] < P_CALM_TO_STRESS
        stressed[i] = state
    vol_mult = np.where(stressed, STRESS_VOL_MULTIPLIER, 1.0)

    # Log returns of close-to-close moves (with Ito correction for the drift)
    sigma_t = daily_vol * vol_mult
    log_ret = daily_drift - 0.5 * sigma_t**2 + sigma_t * shocks
    close = start_price * np.exp(np.cumsum(log_ret))

    # Open = previous close with a small overnight gap
    prev_close = np.concatenate(([start_price], close[:-1]))
    open_ = prev_close * np.exp(rng.normal(0.0, 0.003, size=n))

    # High / Low extend beyond the open-close range by a random intraday amount
    wick_scale = 0.4 * sigma_t
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.0, wick_scale)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.0, wick_scale)))

    # Volume: lognormal noise, higher on days with large absolute returns
    volume = base_volume * np.exp(rng.normal(0.0, 0.25, size=n))
    volume = volume * (1 + 20 * np.abs(log_ret))
    volume = np.maximum(volume.round(), 1).astype(np.int64)

    out = pd.DataFrame(
        {
            "date": dates,
            "open": open_.round(2),
            "high": high.round(2),
            "low": low.round(2),
            "close": close.round(2),
            "volume": volume,
            "volatility_regime": np.where(stressed, "stressed", "calm"),
        }
    )
    # Rounding can break the OHLC ordering; enforce it explicitly
    out["high"] = out[["open", "high", "low", "close"]].max(axis=1)
    out["low"] = out[["open", "high", "low", "close"]].min(axis=1)
    return out


def generate_data():
    """Build the tickers and prices tables."""
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range(START_DATE, periods=N_DAYS)

    tickers_rows = []
    price_frames = []
    for ticker, sector, drift, vol, start_price, base_volume in TICKER_PARAMS:
        tickers_rows.append(
            {
                "ticker": ticker,
                "sector": sector,
                "annual_drift": drift,
                "annual_volatility": vol,
                "start_price": start_price,
            }
        )
        frame = simulate_ticker(rng, dates, drift, vol, start_price, base_volume)
        frame.insert(1, "ticker", ticker)
        price_frames.append(frame)

    tickers = pd.DataFrame(tickers_rows)
    prices = pd.concat(price_frames, ignore_index=True)
    prices["date"] = prices["date"].dt.strftime("%Y-%m-%d")
    return tickers, prices


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    tickers, prices = generate_data()
    tickers.to_csv(DATA_DIR / "tickers.csv", index=False)
    prices.to_csv(DATA_DIR / "prices.csv", index=False)
    print(f"Saved {len(tickers)} rows to {DATA_DIR / 'tickers.csv'}")
    print(f"Saved {len(prices)} rows to {DATA_DIR / 'prices.csv'}")


if __name__ == "__main__":
    main()
