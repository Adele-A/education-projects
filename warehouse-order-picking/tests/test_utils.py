"""Tests for the helper functions in src/utils.py (SYNTHETIC data)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

import data_generator as dg
import utils


@pytest.fixture(scope="module")
def merged() -> pd.DataFrame:
    """Merged orders + pickers table generated in memory (no files needed)."""
    rng = np.random.default_rng(dg.SEED)
    pickers = dg.generate_pickers(rng)
    orders = dg.generate_orders(rng, pickers)
    return orders.merge(pickers, on="picker_id", how="left")


def test_load_merged_reads_and_joins(tmp_path):
    pd.DataFrame(
        {"picker_id": ["P001", "P002"], "experience_months": [10, 50],
         "contract_type": ["temp", "full_time"]}
    ).to_csv(tmp_path / "pickers.csv", index=False)
    pd.DataFrame(
        {"order_id": ["O1", "O2", "O3"], "picker_id": ["P001", "P002", "P001"],
         "order_date": ["2025-01-01"] * 3}
    ).to_csv(tmp_path / "orders.csv", index=False)

    result = utils.load_merged(tmp_path)
    assert len(result) == 3
    assert {"experience_months", "contract_type"} <= set(result.columns)
    assert result["contract_type"].notna().all()


def test_build_regression_data_shape_and_types(merged):
    X, y = utils.build_regression_data(merged)
    assert len(X) == len(y) == len(merged)
    assert X.isna().sum().sum() == 0
    assert all(np.issubdtype(t, np.floating) for t in X.dtypes)
    # drop_first=True keeps k-1 dummy columns per categorical feature
    assert "shift_night" in X.columns
    assert "priority_express" in X.columns
    assert "shift_afternoon" not in X.columns or "shift_morning" not in X.columns
    assert "pick_time_min" not in X.columns


def test_regression_metrics_known_values():
    perfect = utils.regression_metrics([1, 2, 3], [1, 2, 3])
    assert perfect == pytest.approx({"mae": 0.0, "rmse": 0.0, "r2": 1.0})

    mean_pred = utils.regression_metrics([1, 2, 3], [2, 2, 2])
    assert mean_pred["mae"] == pytest.approx(2 / 3)
    assert mean_pred["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert mean_pred["r2"] == pytest.approx(0.0)


def test_prepare_cluster_features_scaled_and_input_untouched(merged):
    before = merged[utils.CLUSTER_FEATURES].copy()
    X, scaler = utils.prepare_cluster_features(merged)
    assert X.shape == (len(merged), len(utils.CLUSTER_FEATURES))
    assert np.allclose(X.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(X.std(axis=0), 1, atol=1e-8)
    # The original data must not be modified by the log transform
    pd.testing.assert_frame_equal(merged[utils.CLUSTER_FEATURES], before)
    assert hasattr(scaler, "mean_")


def test_choose_k_picks_highest_silhouette():
    scores = pd.DataFrame(
        {"k": [3, 4, 5], "inertia": [30.0, 20.0, 15.0], "silhouette": [0.30, 0.45, 0.40]}
    )
    assert utils.choose_k(scores) == 4


def test_evaluate_k_range_and_profile_clusters(merged):
    sample = merged.head(600).reset_index(drop=True)
    X, _ = utils.prepare_cluster_features(sample)

    scores = utils.evaluate_k_range(X, range(2, 4), random_state=0)
    assert list(scores["k"]) == [2, 3]
    assert scores["silhouette"].between(-1, 1).all()
    assert (scores["inertia"] > 0).all()

    # Simple deterministic labels to test the cluster summary
    labels = np.arange(len(sample)) % 3
    profiles = utils.profile_clusters(sample, labels)
    assert profiles["orders"].sum() == len(sample)
    assert profiles["share_pct"].sum() == pytest.approx(100, abs=0.1)
    # Sorted by mean picking time
    assert profiles["pick_time_min"].is_monotonic_increasing
