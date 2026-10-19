"""Tests for the synthetic data generation (src/data_generator.py).

The tests build the tables in memory with the fixed project seed, so they do
not need the CSV files and do not write anything to disk.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

import data_generator as dg

EXPECTED_COLUMNS = [
    "session_id", "session_date", "traffic_source", "device", "country",
    "user_type", "pages_viewed", "session_duration_sec", "viewed_product",
    "added_to_cart", "reached_checkout", "purchased", "order_value",
]


def make_sessions() -> pd.DataFrame:
    """Generate the sessions table with the project seed."""
    return dg.generate_sessions(np.random.default_rng(dg.SEED))


def test_sessions_columns_and_row_count():
    df = make_sessions()
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) == dg.N_SESSIONS
    assert df["session_id"].is_unique
    assert df.isna().sum().sum() == 0


def test_sessions_value_ranges_and_categories():
    df = make_sessions()
    assert set(df["traffic_source"]) <= set(dg.SOURCES)
    assert set(df["device"]) <= set(dg.DEVICES)
    assert set(df["country"]) <= set(dg.COUNTRIES)
    assert set(df["user_type"]) <= {"new", "returning"}
    assert (df["pages_viewed"] >= 1).all()
    assert df["session_duration_sec"].between(3, 3600).all()
    dates = pd.to_datetime(df["session_date"])
    assert dates.min() >= pd.Timestamp(dg.START_DATE)
    assert dates.max() <= pd.Timestamp(dg.START_DATE) + pd.Timedelta(days=dg.N_DAYS - 1)
    assert dates.is_monotonic_increasing


def test_funnel_is_nested_and_order_value_consistent():
    df = make_sessions()
    for col in ["viewed_product", "added_to_cart", "reached_checkout", "purchased"]:
        assert set(df[col].unique()) <= {0, 1}
    assert (df["added_to_cart"] <= df["viewed_product"]).all()
    assert (df["reached_checkout"] <= df["added_to_cart"]).all()
    assert (df["purchased"] <= df["reached_checkout"]).all()
    # Order value is positive exactly for purchasing sessions
    assert (df.loc[df["purchased"] == 0, "order_value"] == 0.0).all()
    assert (df.loc[df["purchased"] == 1, "order_value"] > 0).all()


def test_generation_is_reproducible_and_sources_table_valid():
    pd.testing.assert_frame_equal(make_sessions(), make_sessions())

    sources = dg.generate_traffic_sources()
    assert list(sources.columns) == ["traffic_source", "channel_type", "cost_per_session"]
    assert len(sources) == 6
    assert set(sources["traffic_source"]) == set(dg.SOURCES)
    assert sources["cost_per_session"].between(0.0, 1.2).all()
    assert set(sources["channel_type"]) <= {"paid", "unpaid", "owned"}
