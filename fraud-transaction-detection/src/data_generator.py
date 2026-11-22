"""Generate SYNTHETIC customers and card transactions for fraud detection.

All data is produced by this script with a fixed random seed (42).
It is not real data. Run from the project root:

    python src/data_generator.py

Output:
    data/customers.csv
    data/transactions.csv
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Paths are relative to the project root, not to the current working directory
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

SEED = 42
N_CUSTOMERS = 1000
N_TRANSACTIONS = 5000
N_FRAUD = 100  # exactly 2% of transactions are fraudulent
START_DATE = pd.Timestamp("2025-01-01")
N_DAYS = 181  # 2025-01-01 .. 2025-06-30

COUNTRIES = ["US", "UK", "DE", "FR", "ES"]
COUNTRY_P = [0.40, 0.20, 0.15, 0.15, 0.10]

MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "fuel", "retail",
    "travel", "electronics", "entertainment", "health",
]
LEGIT_CATEGORY_P = [0.28, 0.18, 0.12, 0.14, 0.06, 0.05, 0.10, 0.07]
FRAUD_CATEGORY_P = [0.03, 0.04, 0.03, 0.20, 0.22, 0.34, 0.08, 0.06]

CHANNELS = ["pos", "online", "atm"]
LEGIT_CHANNEL_P = [0.55, 0.38, 0.07]
FRAUD_CHANNEL_P = [0.10, 0.85, 0.05]

# Relative weight of each hour of day (0..23); fraud skews to the night
LEGIT_HOUR_W = np.array([1, 1, 1, 1, 1, 2, 3, 5, 6, 6, 6, 7,
                         8, 7, 6, 6, 6, 7, 8, 8, 7, 5, 3, 2], dtype=float)
FRAUD_HOUR_W = np.array([8, 8, 8, 7, 6, 4, 2, 2, 2, 2, 2, 2,
                         2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 5, 6], dtype=float)


def sample_by_label(rng, is_fraud, values, legit_p, fraud_p):
    """Sample categorical values using different probabilities per class."""
    values = np.asarray(values)
    out = np.empty(len(is_fraud), dtype=values.dtype)
    legit_mask = is_fraud == 0
    out[legit_mask] = rng.choice(values, size=legit_mask.sum(), p=legit_p)
    out[~legit_mask] = rng.choice(values, size=(~legit_mask).sum(), p=fraud_p)
    return out


def generate_customers(rng):
    """Create the customers table."""
    n = N_CUSTOMERS
    age = np.clip(rng.normal(42, 14, n), 18, 80).round().astype(int)
    account_age_days = rng.integers(30, 3650, n)
    country = rng.choice(COUNTRIES, size=n, p=COUNTRY_P)
    monthly_income = np.clip(rng.lognormal(mean=8.0, sigma=0.5, size=n), 1000, 20000)
    monthly_income = monthly_income.round(2)
    # Credit limit is a multiple of income, rounded to the nearest 100
    credit_limit = np.maximum(
        (monthly_income * rng.uniform(1.5, 4.0, n)).round(-2), 500
    ).astype(int)

    return pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "age": age,
        "country": country,
        "account_age_days": account_age_days,
        "monthly_income": monthly_income,
        "credit_limit": credit_limit,
    })


def generate_transactions(rng):
    """Create the transactions table with an is_fraud label."""
    n = N_TRANSACTIONS

    # Fraud label: exactly N_FRAUD randomly chosen rows
    is_fraud = np.zeros(n, dtype=int)
    is_fraud[rng.choice(n, size=N_FRAUD, replace=False)] = 1
    legit = is_fraud == 0

    customer_id = rng.integers(1, N_CUSTOMERS + 1, n)

    # Timestamp: random day, hour depends on the label, random minute/second
    day = rng.integers(0, N_DAYS, n)
    hour = np.empty(n, dtype=int)
    hour[legit] = rng.choice(24, size=legit.sum(), p=LEGIT_HOUR_W / LEGIT_HOUR_W.sum())
    hour[~legit] = rng.choice(24, size=(~legit).sum(), p=FRAUD_HOUR_W / FRAUD_HOUR_W.sum())
    minute = rng.integers(0, 60, n)
    second = rng.integers(0, 60, n)
    timestamp = (
        START_DATE
        + pd.to_timedelta(day, unit="D")
        + pd.to_timedelta(hour, unit="h")
        + pd.to_timedelta(minute, unit="m")
        + pd.to_timedelta(second, unit="s")
    )

    # Amount: log-normal, larger for fraud
    amount = np.empty(n)
    amount[legit] = rng.lognormal(3.5, 0.9, legit.sum())
    amount[~legit] = rng.lognormal(5.0, 1.0, (~legit).sum())
    amount = np.clip(amount, 1, 5000).round(2)

    merchant_category = sample_by_label(
        rng, is_fraud, MERCHANT_CATEGORIES, LEGIT_CATEGORY_P, FRAUD_CATEGORY_P
    )
    channel = sample_by_label(
        rng, is_fraud, CHANNELS, LEGIT_CHANNEL_P, FRAUD_CHANNEL_P
    )

    # Foreign transaction flag
    is_foreign = np.where(is_fraud == 1,
                          rng.random(n) < 0.45,
                          rng.random(n) < 0.04).astype(int)

    # New device flag only makes sense for online transactions
    new_device = np.where(is_fraud == 1,
                          rng.random(n) < 0.50,
                          rng.random(n) < 0.05).astype(int)
    new_device[channel != "online"] = 0

    # Distance from the customer's home; foreign transactions are much farther
    distance = np.empty(n)
    distance[legit] = rng.exponential(8, legit.sum())
    distance[~legit] = rng.exponential(60, (~legit).sum())
    distance = distance + is_foreign * rng.uniform(300, 6000, n)
    distance = np.clip(distance, 0, 8000).round(1)

    df = pd.DataFrame({
        "timestamp": timestamp,
        "customer_id": customer_id,
        "amount": amount,
        "merchant_category": merchant_category,
        "channel": channel,
        "is_foreign": is_foreign,
        "new_device": new_device,
        "distance_from_home_km": distance,
        "is_fraud": is_fraud,
    })

    # Order by time and assign ids so ids follow chronology
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "transaction_id", np.arange(1, n + 1))
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(DATA_DIR, exist_ok=True)

    customers = generate_customers(rng)
    transactions = generate_transactions(rng)

    customers.to_csv(DATA_DIR / "customers.csv", index=False)
    transactions.to_csv(DATA_DIR / "transactions.csv", index=False)

    print("Synthetic data generated (seed=%d)" % SEED)
    print("customers.csv:    %d rows, %d columns" % customers.shape)
    print("transactions.csv: %d rows, %d columns" % transactions.shape)
    print("Fraud transactions: %d" % transactions["is_fraud"].sum())
    print("Saved to:", DATA_DIR)


if __name__ == "__main__":
    main()
