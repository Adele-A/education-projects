"""Tests for the synthetic data generator (src/data_generator.py).

All data is SYNTHETIC. The tests generate the tables in memory with the project's
seed, so they do not need files in data/ and do not touch figures.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import data_generator as dg  # noqa: E402


@pytest.fixture(scope="module")
def generated():
    """Generate customers, hidden groups and orders once for all tests."""
    rng = np.random.default_rng(dg.SEED)
    customers, groups = dg.generate_customers(rng)
    orders = dg.generate_orders(customers, groups, rng)
    return customers, groups, orders


def test_customers_columns_and_row_count(generated):
    customers, groups, _ = generated
    expected = [
        "customer_id",
        "signup_date",
        "country",
        "age",
        "acquisition_channel",
        "newsletter_subscriber",
    ]
    assert list(customers.columns) == expected
    assert len(customers) == dg.N_CUSTOMERS
    assert len(groups) == dg.N_CUSTOMERS
    assert customers["customer_id"].is_unique
    assert set(groups) <= set(dg.SEGMENTS)


def test_customers_value_ranges(generated):
    customers, _, _ = generated
    assert customers["age"].between(18, 75).all()
    assert customers["signup_date"].between(dg.SIGNUP_START, dg.SIGNUP_END).all()
    assert set(customers["newsletter_subscriber"].unique()) <= {0, 1}
    assert set(customers["country"]) <= set(dg.COUNTRIES)
    assert set(customers["acquisition_channel"]) <= set(dg.CHANNELS)


def test_orders_columns_and_row_limit(generated):
    _, _, orders = generated
    expected = [
        "order_id",
        "customer_id",
        "order_date",
        "order_value",
        "items_count",
        "payment_method",
        "discount_used",
    ]
    assert list(orders.columns) == expected
    assert 0 < len(orders) <= dg.MAX_ORDERS
    assert orders["order_id"].is_unique


def test_orders_value_ranges(generated):
    _, _, orders = generated
    assert orders["order_value"].between(5, 2000).all()
    assert (orders["items_count"] >= 1).all()
    assert set(orders["discount_used"].unique()) <= {0, 1}
    assert set(orders["payment_method"]) <= set(dg.PAYMENT_METHODS)
    assert orders["order_date"].max() <= dg.DATA_END


def test_orders_integrity_with_customers(generated):
    customers, _, orders = generated
    # Every customer has at least one order and no order has an unknown customer
    assert set(orders["customer_id"]) == set(customers["customer_id"])
    # No order is dated before the customer's signup
    merged = orders.merge(customers[["customer_id", "signup_date"]], on="customer_id")
    assert (merged["order_date"] >= merged["signup_date"]).all()


def test_generation_is_reproducible():
    def build():
        rng = np.random.default_rng(dg.SEED)
        customers, groups = dg.generate_customers(rng)
        return dg.generate_orders(customers, groups, rng)

    pd.testing.assert_frame_equal(build(), build())
