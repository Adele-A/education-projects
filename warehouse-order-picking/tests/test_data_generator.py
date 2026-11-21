"""Tests for the SYNTHETIC data generator (src/data_generator.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

import data_generator as dg


def make_tables(seed: int = dg.SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate both tables in memory with the given seed."""
    rng = np.random.default_rng(seed)
    pickers = dg.generate_pickers(rng)
    orders = dg.generate_orders(rng, pickers)
    return pickers, orders


@pytest.fixture(scope="module")
def tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the tables once for all tests in this module."""
    return make_tables()


def test_pickers_schema_and_ranges(tables):
    pickers, _ = tables
    assert list(pickers.columns) == ["picker_id", "experience_months", "contract_type"]
    assert len(pickers) == dg.N_PICKERS
    assert pickers["picker_id"].is_unique
    assert pickers["experience_months"].between(1, 120).all()
    assert set(pickers["contract_type"]) <= {"full_time", "part_time", "temp"}
    # Temp workers are capped at 24 months of experience
    temps = pickers[pickers["contract_type"] == "temp"]
    assert (temps["experience_months"] <= 24).all()


def test_orders_schema_and_row_count(tables):
    pickers, orders = tables
    expected = [
        "order_id", "picker_id", "order_date", "shift", "priority", "n_lines",
        "n_units", "total_weight_kg", "n_zones", "distance_m", "pick_time_min",
    ]
    assert list(orders.columns) == expected
    assert len(orders) == dg.N_ORDERS
    assert orders["order_id"].is_unique
    assert orders.isna().sum().sum() == 0
    # Every order refers to an existing picker
    assert set(orders["picker_id"]) <= set(pickers["picker_id"])


def test_orders_value_ranges(tables):
    _, orders = tables
    assert orders["n_lines"].between(1, 40).all()
    assert orders["n_units"].between(1, 600).all()
    assert (orders["n_units"] >= orders["n_lines"]).all()
    assert orders["total_weight_kg"].between(0.1, 500.0).all()
    assert orders["n_zones"].between(1, 8).all()
    assert (orders["n_zones"] <= orders["n_lines"]).all()
    assert orders["distance_m"].between(10.0, 600.0).all()
    assert orders["pick_time_min"].between(0.5, 240.0).all()


def test_orders_categories_and_dates(tables):
    _, orders = tables
    assert set(orders["shift"]) <= {"morning", "afternoon", "night"}
    assert set(orders["priority"]) <= {"standard", "express"}
    dates = pd.to_datetime(orders["order_date"])
    assert dates.min() >= pd.Timestamp("2025-01-01")
    assert dates.max() <= pd.Timestamp("2025-03-31")


def test_generator_is_reproducible(tables):
    pickers, orders = tables
    pickers2, orders2 = make_tables()
    pd.testing.assert_frame_equal(pickers, pickers2)
    pd.testing.assert_frame_equal(orders, orders2)


def test_different_seed_changes_data(tables):
    _, orders = tables
    _, other = make_tables(seed=123)
    assert not orders["pick_time_min"].equals(other["pick_time_min"])
