"""Tests for the synthetic data generator in src/data_generator.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import data_generator as dg  # noqa: E402


def build_tables(seed: int):
    """Generate both tables in memory (no files are written)."""
    rng = np.random.default_rng(seed)
    products = dg.generate_products(rng)
    sales = dg.generate_weekly_sales(products, rng)
    return products, sales


@pytest.fixture(scope="module")
def tables():
    return build_tables(dg.SEED)


def test_products_columns_and_counts(tables):
    products, _ = tables
    assert list(products.columns) == ["product_id", "category", "brand_tier", "base_price"]
    assert len(products) == len(dg.CATEGORY_ELASTICITY) * dg.N_PRODUCTS_PER_CATEGORY
    assert products["product_id"].is_unique
    assert set(products["category"]) == set(dg.CATEGORY_ELASTICITY)
    assert (products.groupby("category").size() == dg.N_PRODUCTS_PER_CATEGORY).all()


def test_products_value_ranges(tables):
    products, _ = tables
    assert products["base_price"].between(1.5, 15.0).all()
    assert set(products["brand_tier"]).issubset(set(dg.BRAND_TIERS))
    assert products.isna().sum().sum() == 0


def test_weekly_sales_columns_and_row_count(tables):
    products, sales = tables
    expected_cols = [
        "product_id", "week_start", "week_of_year", "price", "competitor_price",
        "is_promo", "is_holiday_week", "units_sold", "revenue",
    ]
    assert list(sales.columns) == expected_cols
    assert len(sales) == len(products) * dg.N_WEEKS
    assert (sales.groupby("product_id").size() == dg.N_WEEKS).all()
    assert set(sales["product_id"]) == set(products["product_id"])


def test_weekly_sales_value_ranges(tables):
    _, sales = tables
    assert (sales["price"] > 0).all()
    assert (sales["competitor_price"] > 0).all()
    assert (sales["units_sold"] >= 0).all()
    assert sales["is_promo"].isin([0, 1]).all()
    assert sales["is_holiday_week"].isin([0, 1]).all()
    assert sales["week_of_year"].between(1, dg.N_WEEKS).all()
    # Holiday flag marks exactly the configured weeks.
    holiday_weeks = set(sales.loc[sales["is_holiday_week"] == 1, "week_of_year"])
    assert holiday_weeks == dg.HOLIDAY_WEEKS
    # Promo share should be near the configured probability.
    assert 0.10 <= sales["is_promo"].mean() <= 0.20
    # Revenue equals price * units (both rounded to cents).
    assert (sales["revenue"] - sales["price"] * sales["units_sold"]).abs().max() < 0.01


def test_promo_weeks_have_lower_prices(tables):
    products, sales = tables
    df = sales.merge(products, on="product_id")
    ratio = df["price"] / df["base_price"]
    assert ratio[df["is_promo"] == 1].mean() < ratio[df["is_promo"] == 0].mean()


def test_generator_is_reproducible_and_seed_dependent():
    p1, s1 = build_tables(dg.SEED)
    p2, s2 = build_tables(dg.SEED)
    pd.testing.assert_frame_equal(p1, p2)
    pd.testing.assert_frame_equal(s1, s2)

    _, s3 = build_tables(dg.SEED + 1)
    assert not s1["units_sold"].equals(s3["units_sold"])
