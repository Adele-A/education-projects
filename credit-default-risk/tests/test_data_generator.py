"""Tests for the SYNTHETIC data generator (src/data_generator.py).

Tables are generated in memory with a small size, so the tests are fast and
do not need any files in data/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from data_generator import SEED, generate_customers, generate_loans  # noqa: E402

N_CUST = 300
N_LOANS = 500


@pytest.fixture(scope="module")
def tables():
    """Small customers and loans tables generated with the project seed."""
    rng = np.random.default_rng(SEED)
    customers = generate_customers(rng, n=N_CUST)
    loans = generate_loans(rng, customers, n=N_LOANS)
    return customers, loans


def test_columns_and_row_counts(tables):
    customers, loans = tables
    assert list(customers.columns) == [
        "customer_id", "age", "annual_income", "employment_years",
        "home_ownership", "region", "credit_score", "existing_debt",
        "num_credit_lines"]
    assert list(loans.columns) == [
        "loan_id", "customer_id", "loan_amount", "term_months", "interest_rate",
        "loan_purpose", "application_date", "debt_to_income",
        "num_delinquencies_2y", "default"]
    assert len(customers) == N_CUST
    assert len(loans) == N_LOANS


def test_customer_value_ranges(tables):
    customers, _ = tables
    assert customers["customer_id"].is_unique
    assert customers["age"].between(18, 75).all()
    assert customers["annual_income"].between(15000, 250000).all()
    assert customers["credit_score"].between(300, 850).all()
    assert customers["num_credit_lines"].between(1, 20).all()
    assert customers["employment_years"].between(0, 45).all()
    # Employment length cannot exceed the years since age 18
    assert (customers["employment_years"] <= customers["age"] - 18 + 1e-9).all()
    # Existing debt is a share of income
    assert (customers["existing_debt"] <= customers["annual_income"]).all()
    assert set(customers["home_ownership"]) <= {"RENT", "MORTGAGE", "OWN"}
    assert set(customers["region"]) <= {"North", "South", "East", "West"}


def test_loan_value_ranges(tables):
    _, loans = tables
    assert loans["loan_id"].is_unique
    assert loans["loan_amount"].between(500, 40000).all()
    assert set(loans["term_months"]) <= {36, 60}
    assert loans["interest_rate"].between(4, 30).all()
    assert loans["debt_to_income"].between(0, 3).all()
    assert loans["num_delinquencies_2y"].between(0, 10).all()
    assert set(loans["default"]) <= {0, 1}
    dates = pd.to_datetime(loans["application_date"])
    assert dates.min() >= pd.Timestamp("2022-01-01")
    assert dates.max() <= pd.Timestamp("2023-12-31")


def test_keys_and_ordering(tables):
    customers, loans = tables
    # Every loan points to an existing customer
    assert loans["customer_id"].isin(customers["customer_id"]).all()
    # Loan ids are 1..n and follow the application date order
    assert loans["loan_id"].tolist() == list(range(1, N_LOANS + 1))
    assert pd.to_datetime(loans["application_date"]).is_monotonic_increasing


def test_defaults_are_minority_class(tables):
    _, loans = tables
    rate = loans["default"].mean()
    assert 0 < rate < 0.5


def test_generation_is_reproducible(tables):
    customers, loans = tables
    rng = np.random.default_rng(SEED)
    customers2 = generate_customers(rng, n=N_CUST)
    loans2 = generate_loans(rng, customers2, n=N_LOANS)
    pd.testing.assert_frame_equal(customers, customers2)
    pd.testing.assert_frame_equal(loans, loans2)
