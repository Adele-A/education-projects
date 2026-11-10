"""Generate SYNTHETIC customer and transaction data for the CLV project.

Purchase behavior follows a BG/NBD-like process:
  * each customer has a purchase rate (lambda) drawn from a Gamma distribution;
  * after every purchase the customer becomes inactive with probability p,
    drawn per customer from a Beta distribution;
  * while active, the time between purchases is exponential with rate lambda.

All data is synthetic. Run from the project root:
    python src/data_generator.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Project paths (relative to the project root, independent of the working directory)
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

SEED = 42
N_CUSTOMERS = 1200
MAX_ROWS_PER_TABLE = 5000

ACQUISITION_START = pd.Timestamp("2024-01-01")
ACQUISITION_END = pd.Timestamp("2025-06-30")
OBSERVATION_END = pd.Timestamp("2025-12-31")

# Acquisition channels: sampling probability and a multiplier on purchase rate
CHANNELS = {
    "organic": (0.35, 1.10),
    "paid_search": (0.25, 0.95),
    "social": (0.20, 0.85),
    "email": (0.10, 1.25),
    "referral": (0.10, 1.15),
}
COUNTRIES = {"US": 0.45, "UK": 0.20, "DE": 0.15, "FR": 0.12, "ES": 0.08}


def generate_customers(rng: np.random.Generator) -> pd.DataFrame:
    """Create the customers table (one row per customer)."""
    n_days = (ACQUISITION_END - ACQUISITION_START).days + 1
    acq_offsets = rng.integers(0, n_days, size=N_CUSTOMERS)
    acquisition_dates = ACQUISITION_START + pd.to_timedelta(acq_offsets, unit="D")

    channel_names = list(CHANNELS)
    channel_probs = [CHANNELS[c][0] for c in channel_names]
    channels = rng.choice(channel_names, size=N_CUSTOMERS, p=channel_probs)
    countries = rng.choice(list(COUNTRIES), size=N_CUSTOMERS, p=list(COUNTRIES.values()))

    customers = pd.DataFrame(
        {
            "customer_id": np.arange(1, N_CUSTOMERS + 1),
            "acquisition_date": acquisition_dates,
            "acquisition_channel": channels,
            "country": countries,
        }
    )
    return customers.sort_values("acquisition_date").reset_index(drop=True).assign(
        customer_id=lambda d: np.arange(1, len(d) + 1)
    )


def generate_transactions(customers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Simulate orders for each customer with a BG/NBD-like process."""
    rows = []
    for cust in customers.itertuples(index=False):
        channel_mult = CHANNELS[cust.acquisition_channel][1]

        # Latent customer traits (not saved to the CSV files)
        weekly_rate = rng.gamma(shape=1.5, scale=0.04) * channel_mult  # purchases per week
        daily_rate = max(weekly_rate / 7.0, 1e-4)
        dropout_prob = rng.beta(1.0, 4.0)  # chance of churning after each purchase
        avg_value = rng.lognormal(mean=np.log(45.0), sigma=0.4)  # typical order value

        horizon_days = (OBSERVATION_END - cust.acquisition_date).days

        # The first order happens on the acquisition date
        t = 0.0
        while True:
            # Order value = customer average with order-level noise (mean 1)
            value = max(5.0, avg_value * rng.gamma(shape=8.0, scale=1 / 8.0))
            rows.append(
                (
                    cust.customer_id,
                    cust.acquisition_date + pd.Timedelta(days=int(t)),
                    round(float(value), 2),
                )
            )
            # Possible churn after this purchase
            if rng.random() < dropout_prob:
                break
            # Waiting time until the next purchase
            t += rng.exponential(1.0 / daily_rate)
            if t > horizon_days:
                break

    transactions = pd.DataFrame(rows, columns=["customer_id", "transaction_date", "order_value"])
    transactions = transactions.sort_values(["transaction_date", "customer_id"]).reset_index(drop=True)
    transactions.insert(0, "transaction_id", np.arange(1, len(transactions) + 1))
    return transactions


def main() -> None:
    rng = np.random.default_rng(SEED)
    customers = generate_customers(rng)
    transactions = generate_transactions(customers, rng)

    # Safety check: keep tables within the allowed size
    for name, table in (("customers", customers), ("transactions", transactions)):
        if len(table) > MAX_ROWS_PER_TABLE:
            raise ValueError(f"{name} has {len(table)} rows, limit is {MAX_ROWS_PER_TABLE}")

    os.makedirs(DATA_DIR, exist_ok=True)
    customers.to_csv(DATA_DIR / "customers.csv", index=False)
    transactions.to_csv(DATA_DIR / "transactions.csv", index=False)

    print(f"Saved {len(customers)} rows to {DATA_DIR / 'customers.csv'}")
    print(f"Saved {len(transactions)} rows to {DATA_DIR / 'transactions.csv'}")


if __name__ == "__main__":
    main()
