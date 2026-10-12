# Data description

All data is **synthetic**. It is produced by `src/data_generator.py` with a fixed random seed (42). The tickers are fictional and the values do not represent real markets.

Ranges below describe how the generator is built. Exact observed minimums and maximums should be taken from the generated files.

## Table `tickers` (`data/tickers.csv`)

One row per fictional stock (4 rows).

| Column | Type | Meaning | Value range |
|---|---|---|---|
| ticker | string | Fictional ticker symbol (primary key) | ALFA, BRVO, CHRL, DLTA |
| sector | string | Assigned sector label | Technology, Utilities, Energy, Healthcare |
| annual_drift | float | Assumed annualised expected return used in the simulation | 0.05 to 0.12 |
| annual_volatility | float | Assumed annualised volatility in calm regime | 0.16 to 0.32 |
| start_price | float | Price level the simulation starts from | 45.0 to 100.0 |

## Table `prices` (`data/prices.csv`)

One row per ticker per business day (4 tickers x 1000 days = 4000 rows). Dates start at 2021-01-04 and use a Monday-Friday calendar without holidays.

| Column | Type | Meaning | Value range |
|---|---|---|---|
| date | string (YYYY-MM-DD) | Trading date | 1000 consecutive business days from 2021-01-04 |
| ticker | string | Fictional ticker symbol (foreign key to `tickers.ticker`) | ALFA, BRVO, CHRL, DLTA |
| open | float | Opening price, previous close plus a small overnight gap | > 0, rounded to 2 decimals |
| high | float | Highest price of the day | > 0, always >= open, close and low |
| low | float | Lowest price of the day | > 0, always <= open, close and high |
| close | float | Closing price | > 0, rounded to 2 decimals |
| volume | integer | Number of shares traded | >= 1, typically around 1 to 3 million, higher on days with large moves |
| volatility_regime | string | Hidden simulation state used for that day, kept for EDA and validation | calm, stressed |

### Notes

- Close-to-close log returns have fat tails (Student-t shocks with 5 degrees of freedom).
- In the `stressed` regime daily volatility is twice the calm level. The regime switches randomly: calm to stressed with probability 0.02 per day, stressed to calm with probability 0.08 per day.
- No missing values are generated.
