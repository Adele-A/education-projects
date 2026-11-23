"""Unit tests for src/utils.py (feature preparation and metrics)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils import build_features, evaluate_scores, top_k_metrics  # noqa: E402


def _tiny_tables():
    customers = pd.DataFrame({
        "customer_id": [1, 2],
        "age": [30, 50],
        "country": ["US", "UK"],
        "account_age_days": [100, 2000],
        "monthly_income": [3000.0, 5000.0],
        "credit_limit": [6000, 12000],
    })
    tx = pd.DataFrame({
        "transaction_id": [1, 2, 3],
        "timestamp": pd.to_datetime(["2025-01-01 03:00:00",
                                     "2025-01-01 12:00:00",
                                     "2025-01-02 23:30:00"]),
        "customer_id": [1, 2, 1],
        "amount": [100.0, 20.0, 600.0],
        "merchant_category": ["electronics", "grocery", "travel"],
        "channel": ["online", "pos", "online"],
        "is_foreign": [1, 0, 0],
        "new_device": [1, 0, 0],
        "distance_from_home_km": [500.0, 2.0, 10.0],
        "is_fraud": [1, 0, 0],
    })
    return customers, tx


def test_build_features_shape_and_no_label():
    customers, tx = _tiny_tables()
    X, y = build_features(tx, customers)
    assert len(X) == len(tx)
    assert "is_fraud" not in X.columns
    assert list(y) == [1, 0, 0]
    assert not X.isna().any().any()


def test_build_features_values():
    customers, tx = _tiny_tables()
    X, _ = build_features(tx, customers)
    assert list(X["is_night"]) == [1, 0, 1]
    assert np.isclose(X.loc[0, "amount_to_limit"], 100 / 6000)
    assert X.loc[0, "channel_online"] == 1 and X.loc[1, "channel_pos"] == 1


def test_top_k_metrics():
    y = np.array([1, 0, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    m = top_k_metrics(y, scores, k=2)
    assert m["true_positives"] == 1
    assert np.isclose(m["precision"], 0.5)
    assert np.isclose(m["recall"], 0.5)


def test_evaluate_scores_perfect_ranking():
    y = np.array([0] * 98 + [1] * 2)
    scores = np.linspace(0, 1, 100)  # fraud rows have the highest scores
    m = evaluate_scores(y, scores, alert_budget=0.02)
    assert np.isclose(m["pr_auc"], 1.0)
    assert np.isclose(m["roc_auc"], 1.0)
    assert np.isclose(m["precision"], 1.0) and np.isclose(m["recall"], 1.0)
