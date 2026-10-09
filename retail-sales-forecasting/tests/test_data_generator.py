"""Tests for the synthetic data generator (src/data_generator.py).

The data is synthetic and generated in memory with the project's seed,
so the tests do not read or write any files.
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

import data_generator as dg


@pytest.fixture(scope="module")
def tables():
    """Generate both tables once with the project seed."""
    rng = np.random.default_rng(dg.SEED)
    stores = dg.build_stores(rng)
    sales = dg.build_weekly_sales(stores, rng)
    return stores, sales


def test_columns_and_row_counts(tables):
    stores, sales = tables
    assert list(stores.columns) == ["store_id", "region", "size_sqm", "opened_year"]
    assert list(sales.columns) == [
        "week_start", "store_id", "category", "units_sold",
        "unit_price", "promo_flag", "holiday_week", "revenue",
    ]
    assert len(stores) == 6
    assert len(sales) == dg.N_WEEKS * len(stores) * len(dg.CATEGORIES)
    # One row per week, store and category
    assert not sales.duplicated(["week_start", "store_id", "category"]).any()


def test_store_value_ranges(tables):
    stores, _ = tables
    assert stores["store_id"].is_unique
    assert stores["size_sqm"].between(800, 3000).all()
    assert stores["opened_year"].between(2005, 2020).all()


def test_sales_value_ranges(tables):
    stores, sales = tables
    assert sales.isna().sum().sum() == 0
    assert (sales["units_sold"] >= 0).all()
    assert (sales["unit_price"] > 0).all()
    assert set(sales["promo_flag"].unique()) <= {0, 1}
    assert set(sales["holiday_week"].unique()) <= {0, 1}
    assert set(sales["store_id"]) <= set(stores["store_id"])
    assert set(sales["category"]) == set(dg.CATEGORIES)
    # Promotion share should be near the configured probability
    assert 0.05 < sales["promo_flag"].mean() < 0.20


def test_weeks_are_consecutive_mondays(tables):
    _, sales = tables
    weeks = pd.Series(sales["week_start"].drop_duplicates().sort_values())
    assert len(weeks) == dg.N_WEEKS
    assert (weeks.dt.weekday == 0).all()
    assert (weeks.diff().dropna() == pd.Timedelta(weeks=1)).all()


def test_revenue_matches_units_times_price(tables):
    _, sales = tables
    expected = sales["units_sold"] * sales["unit_price"]
    assert np.allclose(sales["revenue"], expected, atol=0.01)


def test_generation_is_reproducible():
    def run():
        rng = np.random.default_rng(dg.SEED)
        stores = dg.build_stores(rng)
        return dg.build_weekly_sales(stores, rng)

    pd.testing.assert_frame_equal(run(), run())


def test_holiday_week_flags():
    # Week of Black Friday 2022 (Nov 25) and the week with Christmas 2022
    assert dg.thanksgiving(2022) == date(2022, 11, 24)
    assert dg.is_holiday_week(date(2022, 11, 21)) == 1
    assert dg.is_holiday_week(date(2022, 12, 19)) == 1
    # An ordinary January week
    assert dg.is_holiday_week(date(2022, 1, 3)) == 0
