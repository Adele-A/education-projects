"""Tests for the synthetic data generator (src/data_generator.py).

The tables are generated in memory with the project's fixed seed, so no
files are written and the tests do not depend on data/ or figures/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

import data_generator as dg


@pytest.fixture(scope="module")
def tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the couriers and orders tables once for all tests."""
    rng = np.random.default_rng(dg.SEED)
    couriers = dg.generate_couriers(rng)
    orders = dg.generate_orders(rng, couriers)
    return couriers, orders


def test_couriers_columns_and_rows(tables):
    couriers, _ = tables
    assert list(couriers.columns) == [
        "courier_id", "vehicle_type", "experience_months", "avg_rating"
    ]
    assert len(couriers) == dg.N_COURIERS
    assert couriers["courier_id"].is_unique
    assert couriers.isna().sum().sum() == 0


def test_couriers_value_ranges(tables):
    couriers, _ = tables
    assert set(couriers["vehicle_type"]) <= set(dg.VEHICLE_SPEED)
    assert couriers["experience_months"].between(1, 120).all()
    assert couriers["avg_rating"].between(3.0, 5.0).all()


def test_orders_columns_and_rows(tables):
    _, orders = tables
    expected = [
        "order_id", "courier_id", "order_datetime", "is_peak_hour",
        "distance_km", "weather", "courier_active_orders",
        "prep_time_min", "delivery_time_min",
    ]
    assert list(orders.columns) == expected
    assert len(orders) == dg.N_ORDERS
    assert orders["order_id"].tolist() == list(range(1, dg.N_ORDERS + 1))
    assert orders.isna().sum().sum() == 0


def test_orders_value_ranges(tables):
    _, orders = tables
    assert orders["distance_km"].between(0.3, 15.0).all()
    assert orders["courier_active_orders"].between(1, 6).all()
    assert orders["prep_time_min"].between(3.0, 35.0).all()
    assert orders["delivery_time_min"].between(8.0, 120.0).all()
    assert set(orders["is_peak_hour"]) <= {0, 1}
    assert set(orders["weather"]) <= set(dg.WEATHER_PROBS)


def test_orders_reference_existing_couriers(tables):
    couriers, orders = tables
    assert set(orders["courier_id"]) <= set(couriers["courier_id"])


def test_orders_time_and_peak_flag_consistent(tables):
    _, orders = tables
    # order ids follow time, so datetimes must be non-decreasing
    assert orders["order_datetime"].is_monotonic_increasing
    start = pd.Timestamp(dg.START_DATE)
    assert orders["order_datetime"].min() >= start
    assert orders["order_datetime"].max() < start + pd.Timedelta(days=dg.N_DAYS)
    expected_peak = orders["order_datetime"].dt.hour.isin(dg.PEAK_HOURS).astype(int)
    assert (orders["is_peak_hour"] == expected_peak).all()


def test_generation_is_reproducible():
    def build():
        rng = np.random.default_rng(dg.SEED)
        c = dg.generate_couriers(rng)
        return c, dg.generate_orders(rng, c)

    c1, o1 = build()
    c2, o2 = build()
    pd.testing.assert_frame_equal(c1, c2)
    pd.testing.assert_frame_equal(o1, o2)
