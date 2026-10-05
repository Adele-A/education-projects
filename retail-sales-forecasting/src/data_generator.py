"""Generate synthetic weekly retail sales data.

Creates two tables in the data/ folder of the project:
    - stores.csv        (store attributes)
    - weekly_sales.csv  (weekly sales per store and category)

All data is synthetic. A fixed seed (42) makes the output reproducible.
Run from the project root: python src/data_generator.py
"""

import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

SEED = 42
START_DATE = date(2022, 1, 3)  # a Monday
N_WEEKS = 156                  # three years of weekly data

# Category settings:
#   base_units  - typical weekly units for an average-sized store
#   base_price  - typical unit price
#   season_amp  - amplitude of yearly seasonality (share of the base level)
#   peak_doy    - day of year at which the seasonal pattern peaks
#   holiday_up  - extra uplift in holiday weeks (share of the base level)
CATEGORIES = {
    "Grocery":     {"base_units": 2200, "base_price": 4.5,  "season_amp": 0.05,
                    "peak_doy": 355, "holiday_up": 0.20},
    "Electronics": {"base_units": 300,  "base_price": 180.0, "season_amp": 0.20,
                    "peak_doy": 340, "holiday_up": 0.35},
    "Clothing":    {"base_units": 700,  "base_price": 32.0, "season_amp": 0.25,
                    "peak_doy": 120, "holiday_up": 0.25},
    "Garden":      {"base_units": 450,  "base_price": 18.0, "season_amp": 0.45,
                    "peak_doy": 180, "holiday_up": 0.05},
}

ANNUAL_GROWTH = 0.03      # yearly trend growth of the base level
PROMO_PROB = 0.12         # chance that a row is a promotion week
NOISE_SIGMA = 0.06        # sigma of multiplicative log-normal noise


def build_stores(rng: np.random.Generator) -> pd.DataFrame:
    """Create a small table of stores with region, size and opening year."""
    n_stores = 6
    regions = ["North", "South", "East", "West", "North", "South"]
    sizes = rng.integers(800, 3001, size=n_stores)
    opened = rng.integers(2005, 2021, size=n_stores)
    return pd.DataFrame({
        "store_id": [f"S{i + 1:02d}" for i in range(n_stores)],
        "region": regions,
        "size_sqm": sizes,
        "opened_year": opened,
    })


def thanksgiving(year: int) -> date:
    """Return the date of US Thanksgiving (fourth Thursday of November)."""
    first = date(year, 11, 1)
    first_thursday = first + timedelta(days=(3 - first.weekday()) % 7)
    return first_thursday + timedelta(weeks=3)


def is_holiday_week(week_start: date) -> int:
    """Flag weeks (Mon-Sun) that contain Christmas Eve/Day or Black Friday."""
    days = {week_start + timedelta(days=i) for i in range(7)}
    black_friday = thanksgiving(week_start.year) + timedelta(days=1)
    christmas = {date(week_start.year, 12, 24), date(week_start.year, 12, 25)}
    return int(black_friday in days or bool(christmas & days))


def build_weekly_sales(stores: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Create weekly sales for every store x category x week combination."""
    weeks = [START_DATE + timedelta(weeks=i) for i in range(N_WEEKS)]
    week_index = np.arange(N_WEEKS)
    doy = np.array([w.timetuple().tm_yday for w in weeks])
    holiday = np.array([is_holiday_week(w) for w in weeks])
    mean_size = stores["size_sqm"].mean()

    frames = []
    for _, store in stores.iterrows():
        size_factor = store["size_sqm"] / mean_size
        # Store-level price differences (a few percent)
        price_factor = rng.uniform(0.95, 1.05)

        for category, cfg in CATEGORIES.items():
            # Slightly different level for each store/category pair
            level = cfg["base_units"] * size_factor * rng.uniform(0.9, 1.1)

            trend = 1 + ANNUAL_GROWTH * week_index / 52.0
            seasonal = 1 + cfg["season_amp"] * np.cos(
                2 * np.pi * (doy - cfg["peak_doy"]) / 365.25
            )
            holiday_effect = 1 + cfg["holiday_up"] * holiday

            promo = (rng.random(N_WEEKS) < PROMO_PROB).astype(int)
            promo_effect = 1 + promo * rng.uniform(0.10, 0.30, size=N_WEEKS)

            noise = rng.lognormal(mean=0.0, sigma=NOISE_SIGMA, size=N_WEEKS)

            expected = level * trend * seasonal * holiday_effect * promo_effect * noise
            units = np.maximum(np.round(expected), 0).astype(int)

            # Promotions give a ~10% price cut; small random price noise
            unit_price = (
                cfg["base_price"] * price_factor
                * np.where(promo == 1, 0.90, 1.0)
                * rng.normal(1.0, 0.01, size=N_WEEKS)
            )
            unit_price = np.round(unit_price, 2)

            frames.append(pd.DataFrame({
                "week_start": pd.to_datetime(weeks),
                "store_id": store["store_id"],
                "category": category,
                "units_sold": units,
                "unit_price": unit_price,
                "promo_flag": promo,
                "holiday_week": holiday,
                "revenue": np.round(units * unit_price, 2),
            }))

    sales = pd.concat(frames, ignore_index=True)
    return sales.sort_values(["week_start", "store_id", "category"]).reset_index(drop=True)


def main() -> None:
    """Generate both tables and save them to data/."""
    rng = np.random.default_rng(SEED)
    os.makedirs(DATA_DIR, exist_ok=True)

    stores = build_stores(rng)
    sales = build_weekly_sales(stores, rng)

    stores.to_csv(DATA_DIR / "stores.csv", index=False)
    sales.to_csv(DATA_DIR / "weekly_sales.csv", index=False)

    print(f"Saved {len(stores)} rows to {DATA_DIR / 'stores.csv'}")
    print(f"Saved {len(sales)} rows to {DATA_DIR / 'weekly_sales.csv'}")
    print(f"Period: {sales['week_start'].min().date()} to {sales['week_start'].max().date()}")


if __name__ == "__main__":
    main()
