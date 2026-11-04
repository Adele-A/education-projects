"""Generate SYNTHETIC crypto price data.

Creates three CSV tables in the data/ folder of the project:
  - assets.csv          : static asset information
  - market_regimes.csv  : daily volatility regime of the whole market
  - daily_prices.csv    : daily close, log return and volume per asset

The data is simulated (fixed seed 42) and is NOT real market data.

Run from the project root:
    python src/data_generator.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths and global settings
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

SEED = 42
N_DAYS = 730                 # crypto trades every day, so ~2 calendar years
START_DATE = "2023-01-01"
DAYS_PER_YEAR = 365

# Regime settings: 0 = calm, 1 = turbulent
P_CALM_TO_TURBULENT = 0.02   # daily probability of switching calm -> turbulent
P_TURBULENT_TO_CALM = 0.06   # daily probability of switching turbulent -> calm
VOL_MULTIPLIER = {0: 0.8, 1: 2.0}   # scales each asset's base volatility
LOADING_BOOST_TURBULENT = 0.10      # extra market-factor loading in turbulence
MAX_LOADING = 0.95

T_DF = 5                     # degrees of freedom of the Student-t shocks

# Asset definitions (all values are made up for the simulation)
ASSETS = pd.DataFrame(
    {
        "asset": ["BTC", "ETH", "SOL", "BNB", "ADA", "DOGE"],
        "asset_name": ["Bitcoin-like", "Ether-like", "Solana-like",
                       "BNB-like", "Cardano-like", "Dogecoin-like"],
        "category": ["Store of value", "Smart contract platform",
                     "Smart contract platform", "Exchange token",
                     "Smart contract platform", "Meme"],
        "start_price_usd": [16500.0, 1200.0, 10.0, 245.0, 0.32, 0.07],
        "annual_drift": [0.35, 0.40, 0.60, 0.25, 0.10, 0.05],
        "annual_volatility": [0.55, 0.70, 1.10, 0.60, 0.90, 1.20],
        "market_loading": [0.85, 0.80, 0.70, 0.75, 0.65, 0.55],
        "base_daily_volume_usd": [2.0e10, 9.0e9, 1.5e9, 1.2e9, 4.0e8, 6.0e8],
    }
)


def simulate_regimes(rng: np.random.Generator, n_days: int) -> np.ndarray:
    """Simulate a two-state Markov chain (0 = calm, 1 = turbulent)."""
    states = np.zeros(n_days, dtype=int)
    uniforms = rng.random(n_days)
    for t in range(1, n_days):
        if states[t - 1] == 0:
            states[t] = 1 if uniforms[t] < P_CALM_TO_TURBULENT else 0
        else:
            states[t] = 0 if uniforms[t] < P_TURBULENT_TO_CALM else 1
    return states


def student_t_unit_variance(rng: np.random.Generator, size) -> np.ndarray:
    """Student-t shocks rescaled to have variance 1 (requires df > 2)."""
    return rng.standard_t(T_DF, size=size) * np.sqrt((T_DF - 2) / T_DF)


def simulate_prices(rng: np.random.Generator, regimes: np.ndarray) -> pd.DataFrame:
    """Simulate correlated daily log returns, prices and volume for all assets."""
    n_days = len(regimes)
    n_assets = len(ASSETS)

    # Shared market shock and idiosyncratic shocks (unit variance, fat tails)
    market_shock = student_t_unit_variance(rng, n_days)
    idio_shock = student_t_unit_variance(rng, (n_days, n_assets))

    # Market-factor loading per day and asset (higher in turbulent regime)
    base_loading = ASSETS["market_loading"].to_numpy()
    loading = np.where(
        regimes[:, None] == 1,
        np.minimum(base_loading + LOADING_BOOST_TURBULENT, MAX_LOADING),
        base_loading,
    )
    # Combined shock keeps unit variance: b*M + sqrt(1-b^2)*e
    shock = loading * market_shock[:, None] + np.sqrt(1 - loading**2) * idio_shock

    # Daily volatility and drift per day and asset
    base_daily_vol = ASSETS["annual_volatility"].to_numpy() / np.sqrt(DAYS_PER_YEAR)
    multiplier = np.where(regimes == 1, VOL_MULTIPLIER[1], VOL_MULTIPLIER[0])
    daily_vol = base_daily_vol[None, :] * multiplier[:, None]
    daily_drift = ASSETS["annual_drift"].to_numpy() / DAYS_PER_YEAR

    # Log returns with Ito correction, then cumulative prices
    log_returns = (daily_drift[None, :] - 0.5 * daily_vol**2) + daily_vol * shock
    start_prices = ASSETS["start_price_usd"].to_numpy()
    prices = start_prices[None, :] * np.exp(np.cumsum(log_returns, axis=0))

    # Volume: lognormal noise, rises with the absolute size of the move
    base_volume = ASSETS["base_daily_volume_usd"].to_numpy()
    noise = np.exp(rng.normal(0.0, 0.3, size=(n_days, n_assets)))
    scaled_move = np.abs(log_returns) / base_daily_vol[None, :]
    volume = base_volume[None, :] * noise * (1 + 0.5 * scaled_move)

    # Assemble the long-format table
    dates = pd.date_range(START_DATE, periods=n_days, freq="D")
    frames = []
    for j, asset in enumerate(ASSETS["asset"]):
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "asset": asset,
                    "close_usd": np.round(prices[:, j], 6),
                    "log_return": np.round(log_returns[:, j], 6),
                    "volume_usd": np.round(volume[:, j], 0),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def build_tables() -> dict:
    """Generate all tables and return them as a dict of DataFrames."""
    rng = np.random.default_rng(SEED)

    regimes = simulate_regimes(rng, N_DAYS)
    prices = simulate_prices(rng, regimes)

    dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")
    market_regimes = pd.DataFrame(
        {
            "date": dates,
            "regime": np.where(regimes == 1, "turbulent", "calm"),
        }
    )

    assets = ASSETS[["asset", "asset_name", "category", "start_price_usd"]].copy()

    # Dates are stored as ISO strings (YYYY-MM-DD) for easy loading into SQL
    prices["date"] = prices["date"].dt.strftime("%Y-%m-%d")
    market_regimes["date"] = market_regimes["date"].dt.strftime("%Y-%m-%d")

    return {
        "assets": assets,
        "market_regimes": market_regimes,
        "daily_prices": prices,
    }


def main() -> None:
    """Generate the tables, save them to data/ and print a short summary."""
    os.makedirs(DATA_DIR, exist_ok=True)
    tables = build_tables()

    for name, df in tables.items():
        path = DATA_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"Saved {name}.csv: {df.shape[0]} rows x {df.shape[1]} columns -> {path}")

    regimes = tables["market_regimes"]["regime"].value_counts()
    print("\nDays per regime:")
    print(regimes.to_string())

    print("\nDaily prices preview:")
    print(tables["daily_prices"].head().to_string(index=False))


if __name__ == "__main__":
    main()
