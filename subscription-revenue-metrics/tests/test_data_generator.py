"""Tests for the synthetic data generator (src/data_generator.py).

All data is SYNTHETIC. The tests call generate() directly, so they do not
need the CSV files or the figures.

Run from the project root:
    python -m pytest -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import data_generator as dg  # noqa: E402


@pytest.fixture(scope="module")
def tables():
    """Generate the data once and share it between the tests of this file."""
    return dg.generate()


def test_columns_and_row_counts(tables):
    customers, events = tables
    assert list(customers.columns) == [
        "customer_id", "signup_date", "country", "acquisition_channel",
        "company_size", "initial_plan",
    ]
    assert list(events.columns) == [
        "event_id", "customer_id", "event_date", "event_type", "plan",
        "seats", "mrr_before", "mrr_after", "mrr_change",
    ]
    assert len(customers) == dg.N_CUSTOMERS
    assert 0 < len(events) <= dg.MAX_ROWS
    assert customers["customer_id"].is_unique
    assert events["event_id"].is_unique
    assert customers.isna().sum().sum() == 0
    assert events.isna().sum().sum() == 0


def test_categorical_values_and_date_range(tables):
    customers, events = tables
    assert set(customers["country"]) <= set(dg.COUNTRIES)
    assert set(customers["acquisition_channel"]) <= set(dg.CHANNELS)
    assert set(customers["company_size"]) <= set(dg.SIZES)
    assert set(customers["initial_plan"]) <= set(dg.PLANS)
    assert set(events["event_type"]) <= {"new", "expansion", "contraction", "churn"}
    assert set(events["plan"]) <= set(dg.PLANS)

    low, high = pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")
    assert customers["signup_date"].between(low, high).all()
    assert events["event_date"].between(low, high).all()


def test_mrr_and_seat_ranges(tables):
    _, events = tables
    assert (events["mrr_before"] >= 0).all()
    assert (events["mrr_after"] >= 0).all()
    churn = events["event_type"] == "churn"
    assert (events.loc[churn, "mrr_after"] == 0).all()
    assert (events.loc[churn, "seats"] == 0).all()
    assert (events.loc[~churn, "seats"] >= 1).all()

    # MRR after a non-churn event equals seats * price per seat of the plan.
    active = events[~churn]
    expected = active["seats"] * active["plan"].map(dg.PRICE_PER_SEAT)
    assert (active["mrr_after"] - expected).abs().max() < 0.01


def test_event_consistency(tables):
    customers, events = tables
    diff = events["mrr_after"] - events["mrr_before"] - events["mrr_change"]
    assert diff.abs().max() < 0.01

    positive = events["event_type"].isin(["new", "expansion"])
    assert (events.loc[positive, "mrr_change"] > 0).all()
    assert (events.loc[~positive, "mrr_change"] < 0).all()

    # Exactly one 'new' event per customer, on the signup date.
    new = events[events["event_type"] == "new"]
    assert len(new) == len(customers)
    assert new["customer_id"].is_unique
    merged = new.merge(customers[["customer_id", "signup_date"]], on="customer_id")
    assert (merged["event_date"] == merged["signup_date"]).all()

    # A churn event, when present, is unique and the last event of the customer.
    churn = events[events["event_type"] == "churn"]
    assert churn["customer_id"].is_unique
    last_id = events.groupby("customer_id")["event_id"].max()
    assert (last_id.loc[churn["customer_id"]].to_numpy() == churn["event_id"].to_numpy()).all()


def test_generation_is_reproducible(tables):
    customers, events = tables
    customers2, events2 = dg.generate()
    pd.testing.assert_frame_equal(customers, customers2)
    pd.testing.assert_frame_equal(events, events2)
