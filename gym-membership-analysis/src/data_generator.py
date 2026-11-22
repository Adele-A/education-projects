"""Generate a SYNTHETIC gym membership dataset.

Creates two CSV tables in the data/ folder of the project:
    - members.csv  : one row per member
    - upgrades.csv : one row per plan upgrade event

All data is artificial and produced with a fixed random seed.
Run from the project root:  python src/data_generator.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

SEED = 42
N_MEMBERS = 3000
SNAPSHOT_DATE = pd.Timestamp("2026-10-31")
FIRST_SIGNUP = pd.Timestamp("2023-01-01")
LAST_SIGNUP = pd.Timestamp("2026-08-31")

PLANS = ["Basic", "Standard", "Premium"]
FEES = {"Basic": 29.99, "Standard": 49.99, "Premium": 79.99}
NEXT_PLAN = {"Basic": "Standard", "Standard": "Premium"}
CITIES = ["Springfield", "Riverton", "Lakeside", "Hillcrest", "Brookfield"]


def sigmoid(x):
    """Logistic function."""
    return 1.0 / (1.0 + np.exp(-x))


def generate_members(rng):
    """Create the members table (without final plan info) and helper arrays."""
    n = N_MEMBERS

    # --- Sign-up dates -------------------------------------------------
    span_days = (LAST_SIGNUP - FIRST_SIGNUP).days
    signup_date = FIRST_SIGNUP + pd.to_timedelta(rng.integers(0, span_days + 1, n), unit="D")

    # --- Demographics --------------------------------------------------
    age = np.clip(rng.normal(34, 11, n), 16, 70).round().astype(int)
    gender = rng.choice(["Female", "Male", "Other"], size=n, p=[0.48, 0.49, 0.03])
    city = rng.choice(CITIES, size=n, p=[0.30, 0.25, 0.20, 0.15, 0.10])
    distance = np.clip(rng.lognormal(mean=1.0, sigma=0.6, size=n), 0.1, 30).round(1)

    # --- Contract and services ----------------------------------------
    initial_plan = rng.choice(PLANS, size=n, p=[0.50, 0.35, 0.15])
    contract_type = rng.choice(["Month-to-month", "Annual"], size=n, p=[0.60, 0.40])
    payment_method = rng.choice(["Card", "Direct debit", "Cash"], size=n, p=[0.45, 0.45, 0.10])
    trainer = (rng.random(n) < np.where(initial_plan == "Premium", 0.45, 0.10)).astype(int)
    classes = (rng.random(n) < np.where(initial_plan == "Basic", 0.20, 0.45)).astype(int)

    # --- Behaviour: visits per week over the last 90 days -------------
    visits = rng.gamma(shape=2.2, scale=0.9, size=n)
    visits *= 1 + 0.15 * trainer + 0.10 * classes
    visits *= np.exp(-0.03 * distance)
    visits = np.clip(visits, 0, 7).round(2)

    # More frequent visitors have a shorter gap since their last visit
    gap_scale = 3.0 / np.maximum(visits, 0.2)
    days_since_last = np.clip(rng.exponential(gap_scale), 0, 120).round().astype(int)

    # --- Churn (depends on behaviour, plan, contract, distance) -------
    plan_effect = np.select(
        [initial_plan == "Basic", initial_plan == "Premium"], [0.4, -0.3], default=0.0
    )
    logit = (
        -1.2
        - 0.6 * (visits - 2.0)
        + 0.03 * (days_since_last - 7)
        + 0.08 * distance
        - 0.4 * trainer
        - 0.3 * classes
        + plan_effect
        + np.where(contract_type == "Month-to-month", 0.5, -0.3)
    )
    is_churned = (rng.random(n) < sigmoid(logit)).astype(int)

    # Churn date: at least 45 days after sign-up and not after the snapshot
    min_churn = signup_date + pd.Timedelta(days=45)
    room = np.maximum((SNAPSHOT_DATE - min_churn).days, 0)
    churn_offset = np.floor(rng.random(n) * (room + 1)).astype(int)
    churn_date = pd.Series(min_churn + pd.to_timedelta(churn_offset, unit="D"))
    churn_date = churn_date.where(is_churned == 1, pd.NaT)

    end_date = churn_date.fillna(SNAPSHOT_DATE)
    tenure_months = ((end_date - pd.Series(signup_date)).dt.days / 30.4375).round(1)

    members = pd.DataFrame({
        "member_id": [f"M{i:05d}" for i in range(1, n + 1)],
        "signup_date": pd.Series(signup_date).dt.date,
        "age": age,
        "gender": gender,
        "city": city,
        "initial_plan": initial_plan,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "distance_to_gym_km": distance,
        "has_personal_trainer": trainer,
        "uses_group_classes": classes,
        "avg_visits_per_week": visits,
        "days_since_last_visit": days_since_last,
        "tenure_months": tenure_months,
        "is_churned": is_churned,
        "churn_date": churn_date.dt.date,
    })
    return members, pd.Series(signup_date), end_date


def generate_upgrades(rng, members, signup, end_date):
    """Create upgrade events and set the current plan / fee on members."""
    n = len(members)
    initial = members["initial_plan"].to_numpy()
    visits = members["avg_visits_per_week"].to_numpy()

    tenure_days = (end_date - signup).dt.days.to_numpy()
    eligible = (initial != "Premium") & (tenure_days >= 90)

    # Active visitors are more likely to move up
    p_up = np.where(
        initial == "Basic",
        sigmoid(-2.2 + 0.5 * (visits - 2.0)),
        sigmoid(-2.8 + 0.4 * (visits - 3.0)),
    )
    upgraded = eligible & (rng.random(n) < p_up)

    # Upgrade happens between day 60 after sign-up and the end of membership
    offset = 60 + np.floor(rng.random(n) * np.maximum(tenure_days - 60, 1)).astype(int)
    upgrade_date = signup + pd.to_timedelta(offset, unit="D")

    to_plan = np.array([NEXT_PLAN.get(p, p) for p in initial])
    current_plan = np.where(upgraded, to_plan, initial)

    upgrades = pd.DataFrame({
        "member_id": members["member_id"][upgraded].to_numpy(),
        "upgrade_date": upgrade_date[upgraded].dt.date.to_numpy(),
        "from_plan": initial[upgraded],
        "to_plan": to_plan[upgraded],
    })
    upgrades["monthly_fee_before"] = upgrades["from_plan"].map(FEES)
    upgrades["monthly_fee_after"] = upgrades["to_plan"].map(FEES)
    upgrades = upgrades.sort_values(["upgrade_date", "member_id"]).reset_index(drop=True)
    upgrades.insert(0, "upgrade_id", [f"U{i:05d}" for i in range(1, len(upgrades) + 1)])

    # Add the current plan and its fee to the members table
    members = members.copy()
    members.insert(6, "current_plan", current_plan)
    members.insert(7, "monthly_fee", pd.Series(current_plan).map(FEES).to_numpy())
    return members, upgrades


def main():
    rng = np.random.default_rng(SEED)
    members, signup, end_date = generate_members(rng)
    members, upgrades = generate_upgrades(rng, members, signup, end_date)

    os.makedirs(DATA_DIR, exist_ok=True)
    members.to_csv(DATA_DIR / "members.csv", index=False)
    upgrades.to_csv(DATA_DIR / "upgrades.csv", index=False)

    print(f"Saved members.csv:  {len(members)} rows, {members.shape[1]} columns")
    print(f"Saved upgrades.csv: {len(upgrades)} rows, {upgrades.shape[1]} columns")
    print(f"Output folder: {DATA_DIR}")


if __name__ == "__main__":
    main()
