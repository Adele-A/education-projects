"""Tests for the synthetic data generator in src/data_generator.py.

The generator is called in memory only: no CSV files are written and no
figures are needed, so the tests are fast.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import data_generator as dg  # noqa: E402

SALES_COLUMNS = ["week_start", "category", "units_sold", "unit_price",
                 "promo_flag", "revenue", "avg_temp_c"]
CATEGORY_COLUMNS = ["category", "seasonality_pattern", "base_weekly_units", "base_unit_price"]


@pytest.fixture(scope="module")
def sales() -> pd.DataFrame:
    """Sales table generated with the project seed."""
    return dg.generate_weekly_sales(np.random.default_rng(dg.SEED))


@pytest.fixture(scope="module")
def categories() -> pd.DataFrame:
    return dg.build_categories_table()


def test_categories_table_structure(categories):
    assert list(categories.columns) == CATEGORY_COLUMNS
    assert len(categories) == 8
    assert categories["category"].is_unique
    assert categories["base_weekly_units"].between(600, 1800).all()
    assert categories["base_unit_price"].between(4.0, 18.0).all()


def test_sales_columns_and_row_counts(sales, categories):
    assert list(sales.columns) == SALES_COLUMNS
    assert len(sales) == dg.N_WEEKS * len(categories) == 2080
    assert len(sales) <= dg.MAX_ROWS
    # Every category has all weeks, and the (week, category) key is unique
    assert (sales.groupby("category")["week_start"].nunique() == dg.N_WEEKS).all()
    assert not sales.duplicated(["week_start", "category"]).any()
    assert set(sales["category"]) == set(categories["category"])
    assert sales.isna().sum().sum() == 0


def test_sales_value_ranges(sales):
    assert (sales["units_sold"] >= 0).all()
    assert (sales["unit_price"] > 0).all()
    assert set(sales["promo_flag"].unique()) <= {0, 1}
    assert sales["avg_temp_c"].between(-10.0, 38.0).all()
    # Promotions run in roughly 10% of weeks (wide tolerance, seed is fixed anyway)
    assert 0.05 < sales["promo_flag"].mean() < 0.15
    # Revenue is units times price, rounded to 2 decimals
    expected = (sales["units_sold"] * sales["unit_price"]).round(2)
    assert np.allclose(sales["revenue"], expected, atol=0.011)


def test_sales_dates_are_consecutive_mondays(sales):
    start = pd.Timestamp(dg.START_DATE)
    weeks = pd.DatetimeIndex(sorted(sales["week_start"].unique()))
    assert weeks[0] == start
    assert weeks[-1] == start + pd.Timedelta(weeks=dg.N_WEEKS - 1)
    assert (weeks.dayofweek == 0).all()
    assert (np.diff(weeks.to_numpy()) == np.timedelta64(7, "D")).all()


def test_generation_is_reproducible_and_seed_dependent(sales):
    again = dg.generate_weekly_sales(np.random.default_rng(dg.SEED))
    pd.testing.assert_frame_equal(sales, again)
    other = dg.generate_weekly_sales(np.random.default_rng(dg.SEED + 1))
    assert not sales["units_sold"].equals(other["units_sold"])
