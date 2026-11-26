"""Tests for the SYNTHETIC data generator (src/data_generator.py).

The tables are built in memory with the fixed seed, so the tests do not
need the CSV files in data/ and do not write any files.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

import data_generator as dg


@pytest.fixture(scope="module")
def tables():
    """Generate both tables once and share them between tests."""
    rng = np.random.default_rng(dg.SEED)
    customers = dg.generate_customers(rng)
    responses = dg.generate_responses(customers, rng)
    return customers, responses


def test_columns_and_row_counts(tables):
    customers, responses = tables
    assert list(customers.columns) == [
        "customer_id", "segment", "plan", "region",
        "acquisition_channel", "signup_date", "tenure_months",
    ]
    assert list(responses.columns) == [
        "response_id", "customer_id", "survey_date",
        "nps_score", "nps_category", "comment",
    ]
    assert len(customers) == dg.N_CUSTOMERS
    assert len(responses) == dg.N_RESPONSES


def test_customer_value_ranges_and_categories(tables):
    customers, _ = tables
    assert customers["customer_id"].is_unique
    assert customers["tenure_months"].between(1, 72).all()
    assert set(customers["segment"]) <= set(dg.SEGMENTS)
    assert set(customers["plan"]) <= set(dg.PLANS)
    assert set(customers["region"]) <= set(dg.REGIONS)
    assert set(customers["acquisition_channel"]) <= set(dg.CHANNELS)
    assert customers.drop(columns=["signup_date"]).notna().all().all()


def test_response_scores_categories_and_dates(tables):
    _, responses = tables
    assert responses["nps_score"].between(0, 10).all()
    # The category must follow the 0-6 / 7-8 / 9-10 buckets
    expected = responses["nps_score"].map(dg.nps_category)
    assert (expected == responses["nps_category"]).all()
    dates = pd.to_datetime(responses["survey_date"])
    assert dates.min() >= pd.Timestamp("2025-01-01")
    assert dates.max() <= pd.Timestamp("2025-12-31")
    # response_id is numbered in order of survey date
    assert dates.is_monotonic_increasing


def test_keys_are_consistent_between_tables(tables):
    customers, responses = tables
    assert responses["response_id"].is_unique
    assert responses["customer_id"].is_unique
    assert responses["customer_id"].isin(customers["customer_id"]).all()


def test_comment_missing_share_is_close_to_design(tables):
    _, responses = tables
    missing_share = responses["comment"].isna().mean()
    assert 0.2 < missing_share < 0.4


def test_generator_is_reproducible():
    def build():
        rng = np.random.default_rng(dg.SEED)
        customers = dg.generate_customers(rng)
        return customers, dg.generate_responses(customers, rng)

    c1, r1 = build()
    c2, r2 = build()
    pd.testing.assert_frame_equal(c1, c2)
    pd.testing.assert_frame_equal(r1, r2)
