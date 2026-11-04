# Data description

All data is **synthetic**. It is produced by `src/data_generator.py` (numpy, seed 42) and saved to the `data/` folder. Value ranges below describe what the simulation allows by construction. Actual observed minima and maxima depend on the simulated path and should be measured in the EDA step.

## assets.csv

One row per simulated asset (6 rows).

| Column | Type | Meaning | Value range |
|---|---|---|---|
| `asset` | string | Ticker-like asset code (primary key) | BTC, ETH, SOL, BNB, ADA, DOGE |
| `asset_name` | string | Descriptive name of the fictional asset | 6 distinct text values |
| `category` | string | Asset category | Store of value, Smart contract platform, Exchange token, Meme |
| `start_price_usd` | float | Reference price (USD) before the first simulated day; the first close is this value times the first day's return factor | 0.07 to 16500.0 |

## market_regimes.csv

One row per calendar day (730 rows). The regime is the hidden state that drives volatility in the simulation, provided as ground truth.

| Column | Type | Meaning | Value range |
|---|---|---|---|
| `date` | string (YYYY-MM-DD) | Calendar day (primary key) | 2023-01-01 to 2024-12-30 |
| `regime` | string | Market volatility regime for that day | `calm` or `turbulent` |

## daily_prices.csv

One row per asset per day (6 assets x 730 days = 4380 rows). Key: (`date`, `asset`).

| Column | Type | Meaning | Value range |
|---|---|---|---|
| `date` | string (YYYY-MM-DD) | Calendar day | 2023-01-01 to 2024-12-30 |
| `asset` | string | Asset code, foreign key to `assets.asset` | BTC, ETH, SOL, BNB, ADA, DOGE |
| `close_usd` | float | Simulated daily close price in USD, rounded to 6 decimals | strictly positive |
| `log_return` | float | Natural log return of the close versus the previous close (for the first day, versus `start_price_usd`), rounded to 6 decimals | real number, fat-tailed, centred near zero |
| `volume_usd` | float | Simulated traded volume in USD, rounded to whole dollars; higher on days with large absolute returns | strictly positive |

## Simulation notes

- Shocks follow a Student-t distribution (5 degrees of freedom), rescaled to unit variance, to produce fat tails.
- Each day is either `calm` (volatility multiplier 0.8) or `turbulent` (multiplier 2.0), with regime switching governed by a two-state Markov chain.
- All assets share a common market shock. The loading on it is higher on turbulent days, so correlations rise under stress.
- There are no intentionally missing values in these tables.
