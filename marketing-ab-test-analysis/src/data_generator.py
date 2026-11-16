"""Generate a SYNTHETIC landing page A/B test dataset.

Creates two tables in <project root>/data/:
    users.csv  - one row per visitor (assignment, attributes, conversion flag)
    orders.csv - one row per order, only for visitors who converted

All data is simulated with numpy using a fixed seed (42). It does not
represent real people or a real company.

Run from the project root:
    python src/data_generator.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Paths are computed relative to the project root, not the working directory
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

SEED = 42
N_USERS = 5000
EXPERIMENT_START = pd.Timestamp("2025-03-03")
EXPERIMENT_DAYS = 14

# Simulation parameters (inputs of the simulation, not analysis results)
BASE_CONVERSION = 0.10
VARIANT_MULTIPLIER = 1.18  # relative effect of the new page on conversion probability

DEVICES = ["mobile", "desktop", "tablet"]
DEVICE_PROBS = [0.58, 0.35, 0.07]
DEVICE_MULT = {"mobile": 0.85, "desktop": 1.15, "tablet": 1.00}

SOURCES = ["organic", "paid_search", "social", "email", "direct"]
SOURCE_PROBS = [0.30, 0.25, 0.20, 0.10, 0.15]
SOURCE_MULT = {
    "organic": 1.00,
    "paid_search": 0.90,
    "social": 0.75,
    "email": 1.50,
    "direct": 1.20,
}

COUNTRIES = ["US", "UK", "DE", "FR", "CA"]
COUNTRY_PROBS = [0.45, 0.15, 0.15, 0.13, 0.12]


def generate_users(rng: np.random.Generator) -> pd.DataFrame:
    """Create the visitor table with random assignment and simulated conversion."""
    user_id = np.arange(100001, 100001 + N_USERS)

    # Random 50/50 assignment
    group = np.where(rng.random(N_USERS) < 0.5, "control", "variant")

    visit_offset = rng.integers(0, EXPERIMENT_DAYS, size=N_USERS)
    visit_date = EXPERIMENT_START + pd.to_timedelta(visit_offset, unit="D")

    device = rng.choice(DEVICES, size=N_USERS, p=DEVICE_PROBS)
    traffic_source = rng.choice(SOURCES, size=N_USERS, p=SOURCE_PROBS)
    country = rng.choice(COUNTRIES, size=N_USERS, p=COUNTRY_PROBS)
    is_returning = rng.random(N_USERS) < 0.22

    # Engagement metrics
    page_views = np.clip(rng.poisson(3.2, size=N_USERS) + 1, 1, 15)
    time_on_page_sec = np.clip(
        rng.lognormal(mean=3.8, sigma=0.7, size=N_USERS), 5, 600
    ).round(1)

    # Conversion probability depends on segment, returning status and group
    p = np.full(N_USERS, BASE_CONVERSION)
    p *= pd.Series(device).map(DEVICE_MULT).to_numpy()
    p *= pd.Series(traffic_source).map(SOURCE_MULT).to_numpy()
    p *= np.where(is_returning, 1.40, 1.00)
    p *= np.where(group == "variant", VARIANT_MULTIPLIER, 1.00)
    p = np.clip(p, 0.01, 0.95)
    converted = (rng.random(N_USERS) < p).astype(int)

    return pd.DataFrame(
        {
            "user_id": user_id,
            "group": group,
            "visit_date": visit_date.strftime("%Y-%m-%d"),
            "device": device,
            "traffic_source": traffic_source,
            "country": country,
            "is_returning": is_returning.astype(int),
            "page_views": page_views,
            "time_on_page_sec": time_on_page_sec,
            "converted": converted,
        }
    )


def generate_orders(users: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Create one order per converted visitor."""
    buyers = users.loc[users["converted"] == 1, ["user_id", "visit_date"]].reset_index(
        drop=True
    )
    n_orders = len(buyers)

    items = np.clip(rng.poisson(1.0, size=n_orders) + 1, 1, 5)
    # Order value grows with the number of items; lognormal noise for realism
    order_value = np.clip(
        items * rng.lognormal(mean=3.6, sigma=0.45, size=n_orders), 10, 500
    ).round(2)

    # Most orders happen on the visit day, some within the next two days
    delay_days = rng.choice([0, 1, 2], size=n_orders, p=[0.80, 0.14, 0.06])
    order_date = pd.to_datetime(buyers["visit_date"]) + pd.to_timedelta(
        delay_days, unit="D"
    )

    return pd.DataFrame(
        {
            "order_id": np.arange(500001, 500001 + n_orders),
            "user_id": buyers["user_id"].to_numpy(),
            "order_date": order_date.dt.strftime("%Y-%m-%d"),
            "items": items,
            "order_value": order_value,
        }
    )


def main() -> None:
    rng = np.random.default_rng(SEED)
    os.makedirs(DATA_DIR, exist_ok=True)

    users = generate_users(rng)
    orders = generate_orders(users, rng)

    users.to_csv(DATA_DIR / "users.csv", index=False)
    orders.to_csv(DATA_DIR / "orders.csv", index=False)

    print(f"Saved {len(users)} rows to {DATA_DIR / 'users.csv'}")
    print(f"Saved {len(orders)} rows to {DATA_DIR / 'orders.csv'}")


if __name__ == "__main__":
    main()
