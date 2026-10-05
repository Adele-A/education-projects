"""Generate a synthetic SaaS subscription dataset.

Outputs (relative to the project root):
    data/customers.csv
    data/subscription_events.csv

All data is SYNTHETIC. It is produced by simulating each customer month by
month: customers sign up, may expand (add seats or upgrade plan), contract
(remove seats or downgrade) and eventually churn.

Run from the project root:
    python src/data_generator.py
"""
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

SEED = 42
N_CUSTOMERS = 700
N_MONTHS = 24          # 2023-01 .. 2024-12
START_YEAR = 2023
MAX_ROWS = 5000        # cap per table

# Plans ordered from lowest to highest tier, with price per seat per month.
PLANS = ["Starter", "Growth", "Enterprise"]
PRICE_PER_SEAT = {"Starter": 12.0, "Growth": 25.0, "Enterprise": 45.0}

COUNTRIES = ["US", "UK", "DE", "FR", "CA", "AU", "IN", "BR"]
COUNTRY_P = [0.35, 0.12, 0.12, 0.08, 0.08, 0.07, 0.10, 0.08]

CHANNELS = ["organic", "paid_ads", "referral", "outbound_sales"]
CHANNEL_P = [0.35, 0.30, 0.20, 0.15]
# Relative churn risk by acquisition channel.
CHANNEL_CHURN_MULT = {"organic": 1.0, "paid_ads": 1.4, "referral": 0.7, "outbound_sales": 0.8}

SIZES = ["small", "medium", "large"]
SIZE_P = [0.60, 0.30, 0.10]
SEAT_RANGE = {"small": (1, 6), "medium": (5, 26), "large": (25, 101)}  # [low, high)
# Initial plan probabilities by company size (Starter, Growth, Enterprise).
PLAN_P_BY_SIZE = {
    "small": [0.75, 0.22, 0.03],
    "medium": [0.30, 0.55, 0.15],
    "large": [0.05, 0.40, 0.55],
}

# Base monthly probabilities by plan.
CHURN_P = {"Starter": 0.045, "Growth": 0.025, "Enterprise": 0.012}
EXPANSION_P = 0.04
CONTRACTION_P = 0.015


def month_start_index_to_date(m: int, rng: np.random.Generator) -> date:
    """Return a random day (1-28) inside month index m (0 = January 2023)."""
    year = START_YEAR + m // 12
    month = m % 12 + 1
    return date(year, month, int(rng.integers(1, 29)))


def simulate_customer(cid: int, signup_m: int, rng: np.random.Generator):
    """Simulate one customer; return the customer row and a list of events."""
    country = rng.choice(COUNTRIES, p=COUNTRY_P)
    channel = rng.choice(CHANNELS, p=CHANNEL_P)
    size = rng.choice(SIZES, p=SIZE_P)
    plan = rng.choice(PLANS, p=PLAN_P_BY_SIZE[size])
    low, high = SEAT_RANGE[size]
    seats = int(rng.integers(low, high))

    signup_date = month_start_index_to_date(signup_m, rng)
    customer = {
        "customer_id": cid,
        "signup_date": signup_date,
        "country": country,
        "acquisition_channel": channel,
        "company_size": size,
        "initial_plan": plan,
    }

    mrr = seats * PRICE_PER_SEAT[plan]
    events = [{
        "customer_id": cid, "event_date": signup_date, "event_type": "new",
        "plan": plan, "seats": seats, "mrr_before": 0.0, "mrr_after": mrr,
    }]

    for m in range(signup_m + 1, N_MONTHS):
        churn_p = CHURN_P[plan] * CHANNEL_CHURN_MULT[channel]
        u = rng.random()
        event_date = month_start_index_to_date(m, rng)
        old_mrr = mrr

        if u < churn_p:
            events.append({
                "customer_id": cid, "event_date": event_date, "event_type": "churn",
                "plan": plan, "seats": 0, "mrr_before": old_mrr, "mrr_after": 0.0,
            })
            break
        elif u < churn_p + EXPANSION_P:
            idx = PLANS.index(plan)
            if idx < len(PLANS) - 1 and rng.random() < 0.35:
                plan = PLANS[idx + 1]                      # plan upgrade
            else:
                seats += max(1, int(np.ceil(seats * rng.uniform(0.1, 0.4))))
            etype = "expansion"
        elif u < churn_p + EXPANSION_P + CONTRACTION_P:
            idx = PLANS.index(plan)
            if seats > 1:
                seats -= int(rng.integers(1, max(2, seats // 4 + 1)))
                seats = max(seats, 1)
            elif idx > 0:
                plan = PLANS[idx - 1]                      # plan downgrade
            else:
                continue                                   # nothing to reduce
            etype = "contraction"
        else:
            continue

        mrr = seats * PRICE_PER_SEAT[plan]
        if mrr == old_mrr:       # no real change, skip the event
            continue
        # Direction must match the event type (a plan change can offset a seat change).
        if (etype == "expansion") != (mrr > old_mrr):
            continue
        events.append({
            "customer_id": cid, "event_date": event_date, "event_type": etype,
            "plan": plan, "seats": seats, "mrr_before": old_mrr, "mrr_after": mrr,
        })
    return customer, events


def generate() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the customers and subscription_events tables."""
    rng = np.random.default_rng(SEED)

    # Signups grow over time: later months get more weight.
    weights = np.linspace(1.0, 2.5, N_MONTHS)
    weights = weights / weights.sum()
    signup_months = np.sort(rng.choice(N_MONTHS, size=N_CUSTOMERS, p=weights))

    customers, events = [], []
    for cid, m in enumerate(signup_months, start=1):
        cust, evs = simulate_customer(cid, int(m), rng)
        customers.append(cust)
        events.extend(evs)

    customers_df = pd.DataFrame(customers)
    events_df = pd.DataFrame(events)
    events_df["event_date"] = pd.to_datetime(events_df["event_date"])
    events_df = events_df.sort_values(["event_date", "customer_id"]).reset_index(drop=True)
    events_df.insert(0, "event_id", np.arange(1, len(events_df) + 1))
    events_df["mrr_change"] = (events_df["mrr_after"] - events_df["mrr_before"]).round(2)
    events_df["mrr_before"] = events_df["mrr_before"].round(2)
    events_df["mrr_after"] = events_df["mrr_after"].round(2)

    customers_df["signup_date"] = pd.to_datetime(customers_df["signup_date"])
    return customers_df, events_df


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    customers_df, events_df = generate()

    assert len(customers_df) <= MAX_ROWS and len(events_df) <= MAX_ROWS

    customers_df.to_csv(DATA_DIR / "customers.csv", index=False)
    events_df.to_csv(DATA_DIR / "subscription_events.csv", index=False)

    print(f"customers.csv: {len(customers_df)} rows")
    print(f"subscription_events.csv: {len(events_df)} rows")
    print(events_df["event_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
