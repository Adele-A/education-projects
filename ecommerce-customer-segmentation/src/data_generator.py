"""Generate synthetic e-commerce data for the customer segmentation project.

All data is SYNTHETIC. It is produced here with numpy using a fixed seed (42).
Customers come from four hidden behavioral groups (the group label is NOT saved),
so that RFM analysis and clustering have realistic structure to find.

Outputs (relative to the project root):
    data/customers.csv
    data/orders.csv

Run from the project root:
    python src/data_generator.py
"""

import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

SEED = 42
N_CUSTOMERS = 1000
MAX_ORDERS = 5000  # hard limit of rows for the orders table

SIGNUP_START = pd.Timestamp("2023-01-01")
SIGNUP_END = pd.Timestamp("2025-09-30")
DATA_END = pd.Timestamp("2025-12-31")  # last possible order date

COUNTRIES = ["United States", "United Kingdom", "Germany", "France", "Canada", "Spain"]
COUNTRY_P = [0.35, 0.18, 0.15, 0.12, 0.12, 0.08]
CHANNELS = ["organic_search", "paid_search", "social_media", "email", "referral"]
CHANNEL_P = [0.30, 0.25, 0.20, 0.15, 0.10]
PAYMENT_METHODS = ["credit_card", "paypal", "debit_card", "gift_card"]
PAYMENT_P = [0.50, 0.28, 0.17, 0.05]

# Hidden behavioral groups:
# share of customers, extra orders (Poisson mean), median unit price,
# range (min, max) of days since the last active day relative to DATA_END
SEGMENTS: dict[str, dict[str, float | tuple[int, int]]] = {
    "champions": {"share": 0.15, "extra_orders": 8.0, "unit_price": 40.0, "lapse": (0, 30)},
    "regulars": {"share": 0.30, "extra_orders": 4.0, "unit_price": 30.0, "lapse": (0, 90)},
    "occasional": {"share": 0.35, "extra_orders": 1.2, "unit_price": 25.0, "lapse": (30, 300)},
    "lapsed": {"share": 0.20, "extra_orders": 2.0, "unit_price": 28.0, "lapse": (200, 500)},
}


def generate_customers(rng: np.random.Generator) -> tuple[pd.DataFrame, npt.NDArray[np.str_]]:
    """Create the customers table and return it with each customer's hidden group.

    Args:
        rng: Seeded numpy random generator used for all draws in this function.

    Returns:
        A tuple of (customers, groups):
        - customers: DataFrame with one row per customer (N_CUSTOMERS rows) and
          columns customer_id, signup_date, country, age, acquisition_channel,
          newsletter_subscriber.
        - groups: array of length N_CUSTOMERS with the hidden behavioral group
          name of each customer (a key of SEGMENTS). This array is not saved to disk.
    """
    names = list(SEGMENTS)
    shares = [SEGMENTS[n]["share"] for n in names]
    groups = rng.choice(names, size=N_CUSTOMERS, p=shares)

    signup_days = (SIGNUP_END - SIGNUP_START).days
    signup_dates = SIGNUP_START + pd.to_timedelta(
        rng.integers(0, signup_days + 1, N_CUSTOMERS), unit="D"
    )
    age = np.clip(rng.normal(37, 11, N_CUSTOMERS).round(), 18, 75).astype(int)

    customers = pd.DataFrame(
        {
            "customer_id": [f"C{i:04d}" for i in range(1, N_CUSTOMERS + 1)],
            "signup_date": signup_dates,
            "country": rng.choice(COUNTRIES, size=N_CUSTOMERS, p=COUNTRY_P),
            "age": age,
            "acquisition_channel": rng.choice(CHANNELS, size=N_CUSTOMERS, p=CHANNEL_P),
            "newsletter_subscriber": rng.binomial(1, 0.45, N_CUSTOMERS),
        }
    )
    return customers, groups


def generate_orders(
    customers: pd.DataFrame, groups: npt.NDArray[np.str_], rng: np.random.Generator
) -> pd.DataFrame:
    """Create the orders table; every customer has at least one order.

    Each customer gets 1 + Poisson(extra_orders) orders, placed between the
    signup date and the customer's last active day. If the total exceeds
    MAX_ORDERS, random extra orders are dropped (first orders are always kept).

    Args:
        customers: Customers table from generate_customers, with customer_id and
            signup_date columns.
        groups: Hidden behavioral group of each customer, aligned with the rows
            of ``customers``.
        rng: Seeded numpy random generator used for all draws in this function.

    Returns:
        DataFrame with one row per order, sorted by order_date and customer_id,
        with columns order_id, customer_id, order_date, order_value, items_count,
        payment_method and discount_used. At most MAX_ORDERS rows.
    """
    rows = []
    for cust_id, signup, group in zip(customers["customer_id"], customers["signup_date"], groups):
        params = SEGMENTS[group]
        n_orders = 1 + rng.poisson(params["extra_orders"])

        # The last active day depends on how long ago the customer went quiet
        lapse = rng.integers(params["lapse"][0], params["lapse"][1] + 1)
        active_end = max(DATA_END - pd.Timedelta(days=int(lapse)), signup)
        span = (active_end - signup).days
        offsets = np.sort(rng.integers(0, span + 1, n_orders))

        # Order value = items * unit price, with log-normal noise
        items = 1 + rng.poisson(1.3, n_orders)
        unit_price = rng.lognormal(np.log(params["unit_price"]), 0.45, n_orders)
        values = np.clip(items * unit_price, 5, 2000).round(2)

        for off, n_items, value in zip(offsets, items, values):
            rows.append((cust_id, signup + pd.Timedelta(days=int(off)), value, int(n_items)))

    orders = pd.DataFrame(rows, columns=["customer_id", "order_date", "order_value", "items_count"])

    # Enforce the row limit: keep each customer's first order, drop random extra ones
    excess = len(orders) - MAX_ORDERS
    if excess > 0:
        extra_idx = orders.index[orders.groupby("customer_id").cumcount() > 0]
        drop_idx = rng.choice(extra_idx, size=excess, replace=False)
        orders = orders.drop(index=drop_idx)

    orders = orders.sort_values(["order_date", "customer_id"]).reset_index(drop=True)
    orders.insert(0, "order_id", [f"O{i:06d}" for i in range(1, len(orders) + 1)])
    orders["payment_method"] = rng.choice(PAYMENT_METHODS, size=len(orders), p=PAYMENT_P)
    orders["discount_used"] = rng.binomial(1, 0.25, len(orders))
    return orders


def main() -> None:
    """Generate both tables and save them to data/."""
    rng = np.random.default_rng(SEED)
    customers, groups = generate_customers(rng)
    orders = generate_orders(customers, groups, rng)

    os.makedirs(DATA_DIR, exist_ok=True)
    customers.to_csv(DATA_DIR / "customers.csv", index=False, date_format="%Y-%m-%d")
    orders.to_csv(DATA_DIR / "orders.csv", index=False, date_format="%Y-%m-%d")

    print(f"Saved {len(customers)} rows to {DATA_DIR / 'customers.csv'}")
    print(f"Saved {len(orders)} rows to {DATA_DIR / 'orders.csv'}")


if __name__ == "__main__":
    main()
