"""Tests for src/data_generator.py (SYNTHETIC data generation).

The tables are generated in memory with the generator's own seed, so the
tests do not need the CSV files or any figures.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data_generator as dg  # noqa: E402


def _generate(seed=dg.SEED):
    """Generate both tables in the same order as dg.main()."""
    rng = np.random.default_rng(seed)
    customers = dg.generate_customers(rng)
    transactions = dg.generate_transactions(rng)
    return customers, transactions


@pytest.fixture(scope="module")
def tables():
    return _generate()


def test_customers_columns_and_ranges(tables):
    customers, _ = tables
    assert list(customers.columns) == [
        "customer_id", "age", "country", "account_age_days",
        "monthly_income", "credit_limit",
    ]
    assert len(customers) == dg.N_CUSTOMERS
    assert customers["customer_id"].is_unique
    assert customers["customer_id"].min() == 1
    assert customers["customer_id"].max() == dg.N_CUSTOMERS
    assert customers["age"].between(18, 80).all()
    assert customers["account_age_days"].between(30, 3649).all()
    assert customers["monthly_income"].between(1000, 20000).all()
    assert (customers["credit_limit"] >= 500).all()
    assert set(customers["country"]) <= set(dg.COUNTRIES)
    assert not customers.isna().any().any()


def test_transactions_columns_and_row_counts(tables):
    _, tx = tables
    assert list(tx.columns) == [
        "transaction_id", "timestamp", "customer_id", "amount",
        "merchant_category", "channel", "is_foreign", "new_device",
        "distance_from_home_km", "is_fraud",
    ]
    assert len(tx) == dg.N_TRANSACTIONS
    assert tx["transaction_id"].is_unique
    assert list(tx["transaction_id"]) == list(range(1, dg.N_TRANSACTIONS + 1))
    assert tx["is_fraud"].sum() == dg.N_FRAUD
    assert not tx.isna().any().any()


def test_transactions_value_ranges_and_categories(tables):
    _, tx = tables
    assert tx["amount"].between(1, 5000).all()
    assert tx["distance_from_home_km"].between(0, 8000).all()
    for col in ["is_foreign", "new_device", "is_fraud"]:
        assert set(tx[col]) <= {0, 1}
    assert set(tx["merchant_category"]) <= set(dg.MERCHANT_CATEGORIES)
    assert set(tx["channel"]) <= set(dg.CHANNELS)
    # A new device only makes sense for online transactions
    assert (tx.loc[tx["channel"] != "online", "new_device"] == 0).all()


def test_timestamps_in_range_and_ordered(tables):
    _, tx = tables
    ts = pd.to_datetime(tx["timestamp"], format="%Y-%m-%d %H:%M:%S")
    assert ts.min() >= pd.Timestamp("2025-01-01 00:00:00")
    assert ts.max() <= pd.Timestamp("2025-06-30 23:59:59")
    # Ids follow chronology
    assert ts.is_monotonic_increasing


def test_foreign_keys_match_customers(tables):
    customers, tx = tables
    assert tx["customer_id"].isin(customers["customer_id"]).all()


def test_generation_is_reproducible_and_seed_dependent():
    c1, t1 = _generate(dg.SEED)
    c2, t2 = _generate(dg.SEED)
    pd.testing.assert_frame_equal(c1, c2)
    pd.testing.assert_frame_equal(t1, t2)
    _, t3 = _generate(dg.SEED + 1)
    assert not t1["amount"].equals(t3["amount"])


def test_fraud_amounts_are_larger_by_design(tables):
    _, tx = tables
    medians = tx.groupby("is_fraud")["amount"].median()
    assert medians[1] > medians[0]
