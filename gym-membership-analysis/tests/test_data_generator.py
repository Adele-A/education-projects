"""Tests for the SYNTHETIC data generator (src/data_generator.py).

The data is generated in memory with the project seed, so no CSV files
and no figures are needed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import data_generator as dg  # noqa: E402

EXPECTED_MEMBER_COLUMNS = [
    "member_id", "signup_date", "age", "gender", "city", "initial_plan",
    "current_plan", "monthly_fee", "contract_type", "payment_method",
    "distance_to_gym_km", "has_personal_trainer", "uses_group_classes",
    "avg_visits_per_week", "days_since_last_visit", "tenure_months",
    "is_churned", "churn_date",
]
EXPECTED_UPGRADE_COLUMNS = [
    "upgrade_id", "member_id", "upgrade_date", "from_plan", "to_plan",
    "monthly_fee_before", "monthly_fee_after",
]


@pytest.fixture(scope="module")
def data():
    """Generate both tables once for all tests in this file."""
    rng = np.random.default_rng(dg.SEED)
    members, signup, end_date = dg.generate_members(rng)
    members, upgrades = dg.generate_upgrades(rng, members, signup, end_date)
    return members, upgrades


def test_columns_and_row_counts(data):
    members, upgrades = data
    assert list(members.columns) == EXPECTED_MEMBER_COLUMNS
    assert list(upgrades.columns) == EXPECTED_UPGRADE_COLUMNS
    assert len(members) == dg.N_MEMBERS
    assert members["member_id"].is_unique
    assert upgrades["upgrade_id"].is_unique
    # Each member has at most one upgrade
    assert upgrades["member_id"].is_unique


def test_member_value_ranges(data):
    members, _ = data
    assert members["age"].between(16, 70).all()
    assert members["distance_to_gym_km"].between(0.1, 30).all()
    assert members["avg_visits_per_week"].between(0, 7).all()
    assert members["days_since_last_visit"].between(0, 120).all()
    assert set(members["is_churned"].unique()) <= {0, 1}
    assert set(members["monthly_fee"].unique()) <= set(dg.FEES.values())
    assert set(members["initial_plan"].unique()) <= set(dg.PLANS)
    signup = pd.to_datetime(members["signup_date"])
    assert signup.min() >= dg.FIRST_SIGNUP
    assert signup.max() <= dg.LAST_SIGNUP
    assert members.drop(columns=["churn_date"]).notna().all().all()


def test_churn_dates_are_consistent(data):
    members, _ = data
    churn = pd.to_datetime(members["churn_date"])
    signup = pd.to_datetime(members["signup_date"])
    churned = members["is_churned"] == 1
    # Churn date is present exactly for churned members
    assert churn[churned].notna().all()
    assert churn[~churned].isna().all()
    # Cancellation is at least 45 days after sign-up and not after the snapshot
    assert ((churn[churned] - signup[churned]).dt.days >= 45).all()
    assert (churn[churned] <= dg.SNAPSHOT_DATE).all()


def test_upgrades_are_consistent_with_members(data):
    members, upgrades = data
    assert len(upgrades) > 0
    assert upgrades["member_id"].isin(members["member_id"]).all()
    assert (upgrades["from_plan"] != "Premium").all()
    assert (upgrades["to_plan"] == upgrades["from_plan"].map(dg.NEXT_PLAN)).all()
    assert (upgrades["monthly_fee_after"] > upgrades["monthly_fee_before"]).all()

    merged = upgrades.merge(members, on="member_id")
    # The upgrade starts from the initial plan and ends in the current plan
    assert (merged["from_plan"] == merged["initial_plan"]).all()
    assert (merged["to_plan"] == merged["current_plan"]).all()
    # Upgrade happens at least 60 days after sign-up
    days = (pd.to_datetime(merged["upgrade_date"]) - pd.to_datetime(merged["signup_date"])).dt.days
    assert (days >= 60).all()
    # Members without an upgrade keep their initial plan
    no_upgrade = members[~members["member_id"].isin(upgrades["member_id"])]
    assert (no_upgrade["current_plan"] == no_upgrade["initial_plan"]).all()
