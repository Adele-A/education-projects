"""Synthetic data generator for the Customer Churn Prediction project.

Creates two CSV tables in <project root>/data/:
  * customers.csv        - one row per customer, includes the target `churned`
  * support_tickets.csv  - one row per support ticket

All data is SYNTHETIC. The random seed is fixed, so output is reproducible.
Run from the project root:  python src/data_generator.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

SEED = 42
N_CUSTOMERS = 4000
N_TICKETS = 4500
SNAPSHOT_DATE = pd.Timestamp("2024-12-31")  # date the data is "extracted"

PLAN_BASE_PRICE = {"Basic": 9.99, "Standard": 14.99, "Premium": 24.99}


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic function."""
    return 1.0 / (1.0 + np.exp(-x))


def generate_customers(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate the customers table.

    Returns the table and a hidden "frustration" latent variable. The latent
    variable is NOT saved; it only links churn risk to the number of support
    tickets, so the ticket table carries real signal.
    """
    n = N_CUSTOMERS

    # Tenure in months (right-skewed, 1..60)
    tenure = np.clip(np.round(rng.gamma(2.0, 12.0, n)), 1, 60).astype(int)
    signup_date = SNAPSHOT_DATE - pd.to_timedelta(tenure * 30, unit="D")

    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20])
    plan = rng.choice(["Basic", "Standard", "Premium"], size=n, p=[0.40, 0.40, 0.20])
    payment = rng.choice(
        ["Credit card", "Debit card", "PayPal", "Bank transfer"],
        size=n, p=[0.35, 0.25, 0.25, 0.15],
    )
    country = rng.choice(
        ["US", "UK", "Germany", "France", "Canada", "Australia"],
        size=n, p=[0.35, 0.15, 0.15, 0.12, 0.13, 0.10],
    )
    age = np.clip(np.round(rng.normal(38, 12, n)), 18, 80).astype(int)

    discount = rng.binomial(1, 0.20, n)
    auto_renew = rng.binomial(1, 0.70, n)

    base_price = np.array([PLAN_BASE_PRICE[p] for p in plan])
    monthly_price = np.round(base_price * (1 - 0.15 * discount), 2)

    # Usage: Premium users log in a bit more often
    plan_bump = np.where(plan == "Premium", 1.5, np.where(plan == "Standard", 0.5, 0.0))
    avg_weekly_logins = np.round(np.clip(rng.gamma(3.0, 1.5, n) + plan_bump, 0, 30), 1)
    avg_session_minutes = np.round(np.clip(rng.normal(25, 8, n), 3, 90), 1)

    # Billing behaviour: late payments in the last 6 months
    late_rate = np.where(payment == "Bank transfer", 0.6, 0.35)
    late_payments_6m = np.clip(rng.poisson(late_rate), 0, 6).astype(int)

    # Hidden customer frustration (drives churn and support tickets)
    frustration = rng.normal(0, 1, n)

    # Churn probability from a logistic model with known effects
    contract_effect = np.select(
        [contract == "Month-to-month", contract == "One year"], [0.9, -0.2], default=-1.0
    )
    logit = (
        0.0
        + contract_effect
        - 0.03 * tenure
        + 0.04 * (monthly_price - 15)
        - 0.12 * avg_weekly_logins
        + 0.35 * late_payments_6m
        - 0.30 * discount
        - 0.50 * auto_renew
        + 0.50 * frustration
    )
    churned = rng.binomial(1, sigmoid(logit))

    customers = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "signup_date": signup_date.strftime("%Y-%m-%d"),
        "tenure_months": tenure,
        "age": age,
        "country": country,
        "contract_type": contract,
        "plan": plan,
        "payment_method": payment,
        "auto_renew": auto_renew,
        "discount_flag": discount,
        "monthly_price": monthly_price,
        "avg_weekly_logins": avg_weekly_logins,
        "avg_session_minutes": avg_session_minutes,
        "late_payments_6m": late_payments_6m,
        "churned": churned,
    })
    return customers, frustration


def generate_tickets(rng: np.random.Generator, customers: pd.DataFrame,
                     frustration: np.ndarray) -> pd.DataFrame:
    """Generate support tickets; more frustrated customers file more tickets."""
    weights = np.exp(0.6 * frustration)
    weights = weights / weights.sum()
    customer_ids = rng.choice(customers["customer_id"].to_numpy(), size=N_TICKETS, p=weights)

    # Tickets were created within the 180 days before the snapshot date
    days_ago = rng.integers(0, 181, N_TICKETS)
    created = SNAPSHOT_DATE - pd.to_timedelta(days_ago, unit="D")

    category = rng.choice(
        ["Billing", "Technical", "Account", "Feature request"],
        size=N_TICKETS, p=[0.30, 0.40, 0.20, 0.10],
    )
    priority = rng.choice(["Low", "Medium", "High"], size=N_TICKETS, p=[0.45, 0.40, 0.15])

    resolution_hours = np.round(np.clip(rng.gamma(2.0, 10.0, N_TICKETS), 0.5, 240), 1)

    # Satisfaction (1-5) gets worse as resolution takes longer; ~15% not answered
    sat_raw = np.round(4.4 - 0.04 * resolution_hours + rng.normal(0, 0.8, N_TICKETS))
    satisfaction = pd.array(np.clip(sat_raw, 1, 5).astype(int), dtype="Int64")
    missing = rng.random(N_TICKETS) < 0.15
    satisfaction[missing] = pd.NA

    tickets = pd.DataFrame({
        "ticket_id": np.arange(1, N_TICKETS + 1),
        "customer_id": customer_ids,
        "created_date": created.strftime("%Y-%m-%d"),
        "category": category,
        "priority": priority,
        "resolution_hours": resolution_hours,
        "satisfaction_score": satisfaction,
    })
    return tickets


def main() -> None:
    """Generate both tables and save them to data/."""
    rng = np.random.default_rng(SEED)
    os.makedirs(DATA_DIR, exist_ok=True)

    customers, frustration = generate_customers(rng)
    tickets = generate_tickets(rng, customers, frustration)

    customers.to_csv(DATA_DIR / "customers.csv", index=False)
    tickets.to_csv(DATA_DIR / "support_tickets.csv", index=False)

    print(f"Saved {len(customers)} rows to {DATA_DIR / 'customers.csv'}")
    print(f"Saved {len(tickets)} rows to {DATA_DIR / 'support_tickets.csv'}")
    print(f"Churn rate in generated data: {customers['churned'].mean():.3f}")
    print(f"Missing satisfaction scores: {tickets['satisfaction_score'].isna().mean():.3f}")


if __name__ == "__main__":
    main()
