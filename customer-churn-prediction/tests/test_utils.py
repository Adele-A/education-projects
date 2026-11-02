"""Tests for the helper functions in src/utils.py.

Uses small hand-made tables so the expected values are easy to verify.
All values are made up for testing and are not real customer data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import utils  # noqa: E402


@pytest.fixture
def customers() -> pd.DataFrame:
    """Three customers with all columns needed by build_features."""
    return pd.DataFrame({
        "customer_id": [1, 2, 3],
        "signup_date": ["2023-01-01", "2022-06-01", "2024-01-01"],
        "tenure_months": [24, 31, 12],
        "age": [30, 45, 52],
        "country": ["US", "UK", "Germany"],
        "contract_type": ["Month-to-month", "One year", "Two year"],
        "plan": ["Basic", "Standard", "Premium"],
        "payment_method": ["Credit card", "PayPal", "Bank transfer"],
        "auto_renew": [0, 1, 1],
        "discount_flag": [0, 1, 0],
        "monthly_price": [9.99, 12.74, 24.99],
        "avg_weekly_logins": [4.0, 10.0, 2.0],
        "avg_session_minutes": [20.0, 30.0, 10.0],
        "late_payments_6m": [1, 0, 3],
        "churned": [1, 0, 1],
    })


@pytest.fixture
def tickets() -> pd.DataFrame:
    """Customer 1 has two tickets (one Billing), customer 3 has one unrated ticket, customer 2 has none."""
    return pd.DataFrame({
        "ticket_id": [1, 2, 3],
        "customer_id": [1, 1, 3],
        "created_date": ["2024-10-01", "2024-11-01", "2024-12-01"],
        "category": ["Billing", "Technical", "Account"],
        "priority": ["Low", "High", "Medium"],
        "resolution_hours": [10.0, 20.0, 5.0],
        "satisfaction_score": [4.0, np.nan, np.nan],
    })


def test_add_ticket_features_aggregates_per_customer(customers, tickets):
    out = utils.add_ticket_features(customers, tickets)
    assert len(out) == len(customers)  # no rows lost or duplicated
    row1 = out[out["customer_id"] == 1].iloc[0]
    assert row1["n_tickets"] == 2
    assert row1["n_billing_tickets"] == 1
    assert row1["avg_resolution_hours"] == pytest.approx(15.0)
    assert row1["avg_satisfaction"] == pytest.approx(4.0)  # missing score ignored


def test_add_ticket_features_handles_customers_without_tickets(customers, tickets):
    out = utils.add_ticket_features(customers, tickets)
    row2 = out[out["customer_id"] == 2].iloc[0]
    assert row2["n_tickets"] == 0
    assert row2["n_billing_tickets"] == 0
    assert pd.isna(row2["avg_resolution_hours"])
    # Customer 3 has a ticket but no satisfaction answer
    row3 = out[out["customer_id"] == 3].iloc[0]
    assert row3["n_tickets"] == 1
    assert pd.isna(row3["avg_satisfaction"])


def test_build_features_drops_columns_and_adds_weekly_minutes(customers, tickets):
    X, y = utils.build_features(customers, tickets)
    assert utils.TARGET not in X.columns
    for col in utils.DROP_COLUMNS:
        assert col not in X.columns
    assert list(y) == [1, 0, 1]
    assert list(X["weekly_minutes"]) == [80.0, 300.0, 20.0]


def test_preprocessor_output_has_no_missing_and_handles_unknown(customers, tickets):
    X, _ = utils.build_features(customers, tickets)
    numeric, categorical = utils.split_feature_types(X)
    assert set(categorical) == set(utils.CATEGORICAL_FEATURES)
    assert "n_tickets" in numeric

    prep = utils.build_preprocessor(X)
    transformed = prep.fit_transform(X)
    assert transformed.shape[0] == len(X)
    assert not np.isnan(transformed).any()  # NaN values were imputed

    # A category not seen during fitting must not raise an error
    new = X.iloc[[0]].copy()
    new["country"] = "Spain"
    assert prep.transform(new).shape[1] == transformed.shape[1]


def test_compute_metrics_perfect_predictions():
    y_true = [0, 0, 1, 1]
    metrics = utils.compute_metrics(y_true, [0.1, 0.2, 0.8, 0.9])
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["threshold"] == 0.5


def test_compute_metrics_respects_threshold():
    y_true = [0, 0, 1, 1]
    proba = [0.1, 0.4, 0.45, 0.9]
    default = utils.compute_metrics(y_true, proba)  # threshold 0.5
    low = utils.compute_metrics(y_true, proba, threshold=0.3)
    assert default["recall"] == pytest.approx(0.5)
    assert low["recall"] == pytest.approx(1.0)
    assert low["precision"] == pytest.approx(2 / 3)
    # Threshold-free metrics do not change
    assert default["roc_auc"] == pytest.approx(low["roc_auc"])
