"""Additional tests for src/utils.py: data loading and feature building on
generated SYNTHETIC data, plus metric edge cases."""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data_generator as dg  # noqa: E402
from utils import (  # noqa: E402
    CHANNELS,
    MERCHANT_CATEGORIES,
    build_features,
    evaluate_scores,
    load_data,
    top_k_metrics,
)


@pytest.fixture(scope="module")
def generated():
    rng = np.random.default_rng(dg.SEED)
    customers = dg.generate_customers(rng)
    transactions = dg.generate_transactions(rng)
    return customers, transactions


def test_load_data_missing_files_exits(tmp_path):
    with pytest.raises(SystemExit):
        load_data(tmp_path)


def test_load_data_round_trip_parses_timestamp(tmp_path, generated):
    customers, transactions = generated
    customers.to_csv(tmp_path / "customers.csv", index=False)
    transactions.to_csv(tmp_path / "transactions.csv", index=False)
    cust_loaded, tx_loaded = load_data(tmp_path)
    assert len(cust_loaded) == len(customers)
    assert len(tx_loaded) == len(transactions)
    assert pd.api.types.is_datetime64_any_dtype(tx_loaded["timestamp"])


def test_build_features_on_generated_data(generated):
    customers, transactions = generated
    tx = transactions.assign(timestamp=pd.to_datetime(transactions["timestamp"]))
    X, y = build_features(tx, customers)
    assert len(X) == len(tx)
    assert not X.isna().any().any()
    assert "is_fraud" not in X.columns
    assert int(y.sum()) == dg.N_FRAUD
    # Exactly one channel and one merchant category per row
    channel_cols = ["channel_%s" % c for c in CHANNELS]
    category_cols = ["cat_%s" % c for c in MERCHANT_CATEGORIES]
    assert (X[channel_cols].sum(axis=1) == 1).all()
    assert (X[category_cols].sum(axis=1) == 1).all()


def test_top_k_metrics_clips_k_and_handles_misses():
    y = np.array([0, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    # k larger than the data is clipped to the number of rows
    m = top_k_metrics(y, scores, k=100)
    assert m["flagged"] == 4
    assert m["true_positives"] == 1
    # No fraud among the flagged rows gives F1 of zero
    miss = top_k_metrics(y, scores, k=1)
    assert miss["true_positives"] == 0
    assert miss["f1"] == 0.0


def test_evaluate_scores_flags_budget_share():
    rng = np.random.default_rng(0)
    y = np.zeros(200, dtype=int)
    y[:4] = 1
    scores = rng.random(200)
    m = evaluate_scores(y, scores, alert_budget=0.05)
    assert m["flagged"] == math.ceil(0.05 * 200)
    assert 0.0 <= m["pr_auc"] <= 1.0
    assert 0.0 <= m["roc_auc"] <= 1.0
