"""Tests for the SYNTHETIC data generator (columns, row counts, value ranges)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

import data_generator as dg


def make_tables(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate both tables in memory (nothing is written to disk)."""
    rng = np.random.default_rng(seed)
    users = dg.generate_users(rng)
    orders = dg.generate_orders(users, rng)
    return users, orders


def test_users_columns_rows_and_ranges():
    users, _ = make_tables()
    expected_cols = [
        "user_id", "group", "visit_date", "device", "traffic_source",
        "country", "is_returning", "page_views", "time_on_page_sec", "converted",
    ]
    assert list(users.columns) == expected_cols
    assert len(users) == dg.N_USERS
    assert users["user_id"].is_unique
    assert set(users["group"]) == {"control", "variant"}
    assert set(users["device"]) <= set(dg.DEVICES)
    assert set(users["traffic_source"]) <= set(dg.SOURCES)
    assert set(users["country"]) <= set(dg.COUNTRIES)
    assert set(users["is_returning"]) <= {0, 1}
    assert set(users["converted"]) <= {0, 1}
    assert users["page_views"].between(1, 15).all()
    assert users["time_on_page_sec"].between(5, 600).all()
    assert users["visit_date"].between("2025-03-03", "2025-03-16").all()
    assert users.isna().sum().sum() == 0


def test_orders_are_consistent_with_users():
    users, orders = make_tables()
    expected_cols = ["order_id", "user_id", "order_date", "items", "order_value"]
    assert list(orders.columns) == expected_cols
    # exactly one order per converted visitor
    assert len(orders) == users["converted"].sum()
    assert orders["order_id"].is_unique
    assert orders["user_id"].is_unique
    assert set(orders["user_id"]) == set(users.loc[users["converted"] == 1, "user_id"])
    assert orders["items"].between(1, 5).all()
    assert orders["order_value"].between(10, 500).all()
    # orders happen 0 to 2 days after the visit
    visit = users.set_index("user_id").loc[orders["user_id"], "visit_date"]
    delay = (pd.to_datetime(orders["order_date"]).to_numpy()
             - pd.to_datetime(visit).to_numpy()) / np.timedelta64(1, "D")
    assert ((delay >= 0) & (delay <= 2)).all()


def test_generation_is_reproducible_with_same_seed():
    users_a, orders_a = make_tables(seed=7)
    users_b, orders_b = make_tables(seed=7)
    pd.testing.assert_frame_equal(users_a, users_b)
    pd.testing.assert_frame_equal(orders_a, orders_b)
