"""Tests for the helper functions in src/utils.py (SYNTHETIC data only)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from data_generator import SEED, generate_customers, generate_loans  # noqa: E402
from utils import (CATEGORICAL_FEATURES, NUMERIC_FEATURES, best_threshold,  # noqa: E402
                   build_pipeline, compute_metrics, cost_curve, expected_cost,
                   load_model_data, prepare_features)


@pytest.fixture(scope="module")
def model_data(tmp_path_factory):
    """Small synthetic tables saved to a temp folder and joined by load_model_data."""
    rng = np.random.default_rng(SEED)
    customers = generate_customers(rng, n=300)
    loans = generate_loans(rng, customers, n=600)
    folder = tmp_path_factory.mktemp("data")
    customers.to_csv(folder / "customers.csv", index=False)
    loans.to_csv(folder / "loans.csv", index=False)
    return load_model_data(folder)


def test_load_model_data_one_row_per_loan(model_data):
    assert len(model_data) == 600
    assert not model_data["credit_score"].isna().any()


def test_prepare_features(model_data):
    X, y = prepare_features(model_data)
    assert list(X.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert "default" not in X.columns
    assert len(X) == len(y)
    assert set(y.unique()) <= {0, 1}
    with pytest.raises(KeyError):
        prepare_features(model_data.drop(columns=["credit_score"]))


def test_pipeline_fit_and_predict_proba(model_data):
    X, y = prepare_features(model_data)
    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_expected_cost():
    y_true = [1, 0, 1, 0]
    y_pred = [0, 1, 1, 0]  # one missed default, one false alarm
    assert expected_cost(y_true, y_pred, cost_fn=5, cost_fp=1) == 6.0
    assert expected_cost(y_true, y_true, cost_fn=5, cost_fp=1) == 0.0


def test_cost_curve_and_best_threshold():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    curve = cost_curve(y_true, scores, cost_fn=5, cost_fp=1)
    assert list(curve.columns) == ["threshold", "cost"]
    assert len(curve) == 99
    thr = best_threshold(y_true, scores, cost_fn=5, cost_fp=1)
    # Perfectly separable scores: the chosen threshold separates the classes
    assert 0.2 < thr <= 0.8
    assert curve["cost"].min() == 0.0


def test_compute_metrics_known_values():
    y_true = [0, 0, 1, 1]
    scores = [0.1, 0.6, 0.4, 0.9]
    m = compute_metrics(y_true, scores, threshold=0.5, cost_fn=5, cost_fp=1)
    assert m["roc_auc"] == pytest.approx(0.75)
    assert (m["true_negatives"], m["false_positives"],
            m["false_negatives"], m["true_positives"]) == (1, 1, 1, 1)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["total_cost"] == pytest.approx(6.0)
