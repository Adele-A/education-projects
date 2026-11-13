"""Tests for the synthetic data generator (columns, row counts, value ranges).

The tests call the generator functions directly with the project seed and
never write files, so they are fast and independent of data/ and figures/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

import data_generator as dg

SUPPLIER_COLUMNS = [
    "supplier_id", "supplier_name", "country", "region", "category",
    "transport_mode", "reliability_tier", "contract_lead_time_days",
    "price_index",
]
ORDER_COLUMNS = [
    "po_id", "supplier_id", "order_date", "promised_date", "delivery_date",
    "promised_lead_time_days", "actual_lead_time_days", "delay_days",
    "on_time", "order_qty", "unit_price", "is_expedited", "order_value",
]


@pytest.fixture(scope="module")
def tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate both tables once with the project seed."""
    rng = np.random.default_rng(dg.SEED)
    suppliers = dg.generate_suppliers(rng)
    orders = dg.generate_orders(rng, suppliers)
    return suppliers, orders


def test_suppliers_columns_counts_and_ranges(tables):
    suppliers, _ = tables
    assert list(suppliers.columns) == SUPPLIER_COLUMNS
    assert len(suppliers) == dg.N_SUPPLIERS
    assert suppliers["supplier_id"].is_unique
    assert suppliers["reliability_tier"].isin(["A", "B", "C"]).all()
    assert suppliers["country"].isin(list(dg.COUNTRIES)).all()
    assert suppliers["category"].isin(list(dg.CATEGORIES)).all()
    assert suppliers["price_index"].between(0.7, 1.4).all()
    assert suppliers["contract_lead_time_days"].between(3, 45).all()
    # Region must match the country definition
    expected_region = suppliers["country"].map(lambda c: dg.COUNTRIES[c][0])
    assert (suppliers["region"] == expected_region).all()


def test_orders_columns_keys_and_ranges(tables):
    suppliers, orders = tables
    assert list(orders.columns) == ORDER_COLUMNS
    assert len(orders) == dg.N_ORDERS
    assert orders["po_id"].is_unique
    assert orders["supplier_id"].isin(suppliers["supplier_id"]).all()
    assert orders.isna().sum().sum() == 0

    order_date = pd.to_datetime(orders["order_date"])
    assert order_date.min() >= dg.START_DATE
    assert order_date.max() <= dg.END_DATE
    assert order_date.is_monotonic_increasing

    assert orders["order_qty"].between(10, 5000).all()
    assert orders["actual_lead_time_days"].between(1, 150).all()
    assert (orders["promised_lead_time_days"] >= 1).all()
    assert (orders["unit_price"] > 0).all()
    assert (orders["order_value"] > 0).all()
    assert set(orders["on_time"].unique()) <= {0, 1}
    assert set(orders["is_expedited"].unique()) <= {0, 1}


def test_orders_date_logic_and_reproducibility(tables):
    suppliers, orders = tables
    order_date = pd.to_datetime(orders["order_date"])
    promised = pd.to_datetime(orders["promised_date"])
    delivered = pd.to_datetime(orders["delivery_date"])

    assert (promised > order_date).all()
    assert (delivered > order_date).all()
    assert ((delivered - promised).dt.days == orders["delay_days"]).all()
    assert ((orders["delay_days"] <= 0).astype(int) == orders["on_time"]).all()

    # Same seed must give identical tables
    rng = np.random.default_rng(dg.SEED)
    suppliers2 = dg.generate_suppliers(rng)
    orders2 = dg.generate_orders(rng, suppliers2)
    pd.testing.assert_frame_equal(suppliers, suppliers2)
    pd.testing.assert_frame_equal(orders, orders2)
