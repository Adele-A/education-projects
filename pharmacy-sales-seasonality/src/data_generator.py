"""Generate SYNTHETIC weekly pharmacy sales data.

Creates two tables in the data/ folder (relative to the project root):
  - categories.csv   : reference table of medicine categories
  - weekly_sales.csv : weekly sales per category

All data is synthetic and produced with a fixed seed (42).
Run from the project root:  python src/data_generator.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

SEED = 42
START_DATE = "2019-01-07"  # a Monday
N_WEEKS = 260              # 5 years of weekly data
MAX_ROWS = 5000            # project limit per table

PROMO_PROB = 0.10          # share of weeks with a promotion
PROMO_UPLIFT = 1.15        # demand multiplier during a promotion
PROMO_DISCOUNT = 0.90      # price multiplier during a promotion
ANNUAL_INFLATION = 0.03    # yearly price growth

# Category definitions.
# amp / amp2: amplitude of the first / second yearly harmonic of log-demand
# (the second harmonic makes the peak sharper), peak_doy: day of year of the peak,
# growth: yearly demand growth, price: base unit price.
CATEGORIES = [
    dict(category="cold_and_flu", pattern="winter peak", base=1800, amp=0.55,
         amp2=0.10, peak_doy=20, growth=0.02, price=6.5),
    dict(category="allergy", pattern="spring peak", base=900, amp=0.30,
         amp2=0.35, peak_doy=115, growth=0.04, price=9.0),
    dict(category="vitamins", pattern="mild winter peak", base=1200, amp=0.20,
         amp2=0.05, peak_doy=30, growth=0.03, price=12.0),
    dict(category="sun_care", pattern="summer peak", base=600, amp=0.70,
         amp2=0.15, peak_doy=190, growth=0.03, price=14.5),
    dict(category="digestive", pattern="weak summer peak", base=1000, amp=0.10,
         amp2=0.00, peak_doy=200, growth=0.01, price=7.5),
    dict(category="pain_relief", pattern="almost flat", base=1500, amp=0.05,
         amp2=0.00, peak_doy=15, growth=0.01, price=5.0),
    dict(category="first_aid", pattern="weak summer peak", base=700, amp=0.12,
         amp2=0.05, peak_doy=185, growth=0.00, price=4.0),
    dict(category="cardio_chronic", pattern="flat with upward trend", base=1300,
         amp=0.03, amp2=0.00, peak_doy=10, growth=0.05, price=18.0),
]


def build_categories_table() -> pd.DataFrame:
    """Reference table with one row per category."""
    return pd.DataFrame(
        {
            "category": [c["category"] for c in CATEGORIES],
            "seasonality_pattern": [c["pattern"] for c in CATEGORIES],
            "base_weekly_units": [c["base"] for c in CATEGORIES],
            "base_unit_price": [c["price"] for c in CATEGORIES],
        }
    )


def generate_temperature(weeks: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Average weekly air temperature (deg C) with a yearly cycle and noise."""
    doy = weeks.dayofyear.to_numpy()
    temp = 12 + 11 * np.sin(2 * np.pi * (doy - 110) / 365.25) + rng.normal(0, 2.0, len(weeks))
    return np.clip(temp, -10, 38).round(1)


def generate_weekly_sales(rng: np.random.Generator) -> pd.DataFrame:
    """Weekly units, price, promotions and revenue for every category."""
    weeks = pd.date_range(START_DATE, periods=N_WEEKS, freq="W-MON")
    doy = weeks.dayofyear.to_numpy()
    t_years = np.arange(N_WEEKS) / 52.0
    temp = generate_temperature(weeks, rng)  # same weather for all categories

    frames = []
    for c in CATEGORIES:
        angle = 2 * np.pi * (doy - c["peak_doy"]) / 365.25
        log_season = c["amp"] * np.cos(angle) + c["amp2"] * np.cos(2 * angle)
        trend = (1 + c["growth"]) ** t_years

        promo = (rng.random(N_WEEKS) < PROMO_PROB).astype(int)
        promo_factor = np.where(promo == 1, PROMO_UPLIFT, 1.0)
        noise = rng.lognormal(mean=0.0, sigma=0.05, size=N_WEEKS)

        expected = c["base"] * trend * np.exp(log_season) * promo_factor * noise
        units = rng.poisson(expected)

        price = (
            c["price"]
            * (1 + ANNUAL_INFLATION) ** t_years
            * rng.uniform(0.98, 1.02, N_WEEKS)
            * np.where(promo == 1, PROMO_DISCOUNT, 1.0)
        ).round(2)

        frames.append(
            pd.DataFrame(
                {
                    "week_start": weeks,
                    "category": c["category"],
                    "units_sold": units,
                    "unit_price": price,
                    "promo_flag": promo,
                    "revenue": (units * price).round(2),
                    "avg_temp_c": temp,
                }
            )
        )

    sales = pd.concat(frames, ignore_index=True)
    return sales.sort_values(["category", "week_start"]).reset_index(drop=True)


def main() -> None:
    rng = np.random.default_rng(SEED)
    os.makedirs(DATA_DIR, exist_ok=True)

    categories = build_categories_table()
    sales = generate_weekly_sales(rng)

    for name, df in [("categories", categories), ("weekly_sales", sales)]:
        assert len(df) <= MAX_ROWS, f"{name} exceeds {MAX_ROWS} rows"
        path = DATA_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"Saved {path.relative_to(PROJECT_DIR)}: {len(df)} rows, {df.shape[1]} columns")


if __name__ == "__main__":
    main()
