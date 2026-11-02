"""Tests for src/data_generator.py (SYNTHETIC data generation).

The tests generate the tables in memory with a fixed seed, so they do not
need the CSV files in data/ and do not write anything to disk.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import data_generator as dg  # noqa: E402


def make_tables(seed: int = dg.SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate both tables in memory with the given seed."""
    rng = np.random.default_rng(seed)
    customers, frustration = dg.generate_customers(rng)
    tickets = dg.generate_tickets(rng, customers, frustration)
    return customers, tickets


@pytest.fixture(scope="module")
def tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the tables once and share them between tests."""
    return make_tables()


def test_customers_columns_and_row_count(tables):
    customers, _ = tables
    expected = [
        "customer_id", "signup_date", "tenure_months", "age", "country",
        "contract_type", "plan", "payment_method", "auto_renew",
        "discount_flag", "monthly_price", "avg_weekly_logins",
        "avg_session_minutes", "late_payments_6m", "churned",
    ]
    assert list(customers.columns) == expected
    assert len(customers) == dg.N_CUSTOMERS
    assert customers["customer_id"].is_unique
    assert customers.drop(columns=["signup_date"]).isna().sum().sum() == 0


def test_customer_value_ranges_and_categories(tables):
    customers, _ = tables
    assert customers["tenure_months"].between(1, 60).all()
    assert customers["age"].between(18, 80).all()
    assert customers["avg_weekly_logins"].between(0, 30).all()
    assert customers["avg_session_minutes"].between(3, 90).all()
    assert customers["late_payments_6m"].between(0, 6).all()
    assert customers["monthly_price"].between(8.49, 24.99).all()
    for col in ["auto_renew", "discount_flag", "churned"]:
        assert set(customers[col].unique()) <= {0, 1}
    assert set(customers["contract_type"]) <= {"Month-to-month", "One year", "Two year"}
    assert set(customers["plan"]) <= set(dg.PLAN_BASE_PRICE)


def test_price_matches_plan_and_discount(tables):
    customers, _ = tables
    base = customers["plan"].map(dg.PLAN_BASE_PRICE)
    expected = (base * (1 - 0.15 * customers["discount_flag"])).round(2)
    assert np.allclose(customers["monthly_price"], expected)


def test_tickets_columns_and_row_count(tables):
    customers, tickets = tables
    expected = ["ticket_id", "customer_id", "created_date", "category",
                "priority", "resolution_hours", "satisfaction_score"]
    assert list(tickets.columns) == expected
    assert len(tickets) == dg.N_TICKETS
    assert tickets["ticket_id"].is_unique
    # Every ticket must point to an existing customer
    assert tickets["customer_id"].isin(customers["customer_id"]).all()


def test_ticket_value_ranges(tables):
    _, tickets = tables
    assert tickets["resolution_hours"].between(0.5, 240).all()
    scores = tickets["satisfaction_score"].dropna()
    assert scores.between(1, 5).all()
    missing_share = tickets["satisfaction_score"].isna().mean()
    assert 0.05 < missing_share < 0.30  # about 15% by design
    dates = pd.to_datetime(tickets["created_date"])
    assert dates.max() <= dg.SNAPSHOT_DATE
    assert dates.min() >= dg.SNAPSHOT_DATE - pd.Timedelta(days=180)


def test_generator_is_reproducible(tables):
    customers, tickets = tables
    customers2, tickets2 = make_tables(dg.SEED)
    pd.testing.assert_frame_equal(customers, customers2)
    pd.testing.assert_frame_equal(tickets, tickets2)
