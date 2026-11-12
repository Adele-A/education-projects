"""Tests for the synthetic data generator (src/data_generator.py).

The tests generate the tables in memory with the project's own functions, so they
do not need the CSV files, the figures or the network, and they run in about a second.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import data_generator as gen  # noqa: E402


def make_tables(seed: int = gen.SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate both tables in memory with a given seed."""
    rng = np.random.default_rng(seed)
    customers = gen.generate_customers(rng)
    transactions = gen.generate_transactions(customers, rng)
    return customers, transactions


@pytest.fixture(scope="module")
def tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tables generated once and shared by the tests in this file."""
    return make_tables()


def test_customers_table_structure_and_ranges(tables):
    customers, _ = tables
    assert list(customers.columns) == [
        "customer_id", "acquisition_date", "acquisition_channel", "country",
    ]
    assert len(customers) == gen.N_CUSTOMERS
    # ids are 1..N and assigned in order of acquisition date
    assert customers["customer_id"].tolist() == list(range(1, gen.N_CUSTOMERS + 1))
    assert customers["acquisition_date"].is_monotonic_increasing
    assert customers["acquisition_date"].between(gen.ACQUISITION_START, gen.ACQUISITION_END).all()
    assert set(customers["acquisition_channel"]) <= set(gen.CHANNELS)
    assert set(customers["country"]) <= set(gen.COUNTRIES)
    assert customers.notna().all().all()


def test_transactions_table_structure_and_ranges(tables):
    customers, transactions = tables
    assert list(transactions.columns) == [
        "transaction_id", "customer_id", "transaction_date", "order_value",
    ]
    # at least one order per customer, and within the documented size limit
    assert gen.N_CUSTOMERS <= len(transactions) <= gen.MAX_ROWS_PER_TABLE
    assert transactions["transaction_id"].is_unique
    assert transactions["order_value"].min() >= 5.0
    assert transactions.notna().all().all()

    # foreign key: every order belongs to a known customer, every customer has an order
    assert set(transactions["customer_id"]) == set(customers["customer_id"])

    # orders never happen before acquisition or after the observation window
    merged = transactions.merge(customers, on="customer_id")
    assert (merged["transaction_date"] >= merged["acquisition_date"]).all()
    assert (merged["transaction_date"] <= gen.OBSERVATION_END).all()

    # the first order of each customer is on the acquisition date
    first_order = merged.groupby("customer_id")["transaction_date"].min()
    acquisition = customers.set_index("customer_id")["acquisition_date"]
    assert (first_order == acquisition.loc[first_order.index]).all()


def test_generation_is_reproducible_with_same_seed(tables):
    customers, transactions = tables
    customers_2, transactions_2 = make_tables(gen.SEED)
    pd.testing.assert_frame_equal(customers, customers_2)
    pd.testing.assert_frame_equal(transactions, transactions_2)
