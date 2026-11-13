"""Tests for the helper functions in src/utils.py using small hand-made data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

import utils


@pytest.fixture
def small_orders() -> pd.DataFrame:
    """Six orders merged with supplier attributes."""
    return pd.DataFrame(
        {
            "po_id": [f"PO-{i}" for i in range(1, 7)],
            "supplier_id": ["S1", "S1", "S1", "S2", "S2", "S2"],
            "order_date": ["2024-01-05", "2024-03-10", "2024-11-02",
                           "2024-02-01", "2024-10-15", "2024-06-20"],
            "promised_lead_time_days": [10, 10, 10, 20, 20, 20],
            "actual_lead_time_days": [8, 10, 14, 20, 25, 30],
            "delay_days": [-2, 0, 4, 0, 5, 10],
            "on_time": [1, 1, 0, 1, 0, 0],
            "order_qty": [100, 200, 300, 50, 60, 70],
            "is_expedited": [0, 0, 1, 0, 0, 0],
            "transport_mode": ["road", "road", "road", "sea", "sea", "sea"],
            "reliability_tier": ["A", "A", "A", "C", "C", "C"],
            "region": ["Europe", "Europe", "Europe", "Asia", "Asia", "Asia"],
            "category": ["MRO", "MRO", "MRO", "Packaging", "Packaging",
                         "Packaging"],
        }
    )


def test_coefficient_of_variation():
    # mean 4, sample std 2 -> CV 0.5
    assert utils.coefficient_of_variation(pd.Series([2, 4, 6])) == pytest.approx(0.5)
    assert np.isnan(utils.coefficient_of_variation(pd.Series([5])))
    assert np.isnan(utils.coefficient_of_variation(pd.Series([-1, 1])))  # zero mean


def test_segment_and_supplier_kpis(small_orders):
    kpis = utils.segment_kpis(small_orders, "supplier_id")
    assert kpis.loc["S1", "orders"] == 3
    assert kpis.loc["S1", "on_time_rate"] == pytest.approx(2 / 3)
    assert kpis.loc["S2", "on_time_rate"] == pytest.approx(1 / 3)
    assert kpis.loc["S1", "mean_lead_time"] == pytest.approx(32 / 3)
    assert kpis.loc["S2", "cv_lead_time"] == pytest.approx(
        kpis.loc["S2", "std_lead_time"] / kpis.loc["S2", "mean_lead_time"]
    )

    suppliers = pd.DataFrame(
        {
            "supplier_id": ["S1", "S2"],
            "country": ["Germany", "China"],
            "region": ["Europe", "Asia"],
            "category": ["MRO", "Packaging"],
            "transport_mode": ["road", "sea"],
            "reliability_tier": ["A", "C"],
        }
    )
    merged = utils.supplier_kpis(small_orders, suppliers)
    assert merged.loc["S2", "country"] == "China"
    assert merged.loc["S1", "reliability_tier"] == "A"
    assert merged.index.name == "supplier_id"


def test_statistical_test_wrappers(small_orders):
    chi = utils.chi_square_on_time(small_orders, "reliability_tier")
    assert set(chi) == {"chi2", "p_value", "dof"}
    assert chi["dof"] == 1
    assert 0.0 <= chi["p_value"] <= 1.0

    kw = utils.kruskal_delay(small_orders, "reliability_tier")
    assert set(kw) == {"h_stat", "p_value"}
    assert 0.0 <= kw["p_value"] <= 1.0
    assert kw["h_stat"] >= 0.0


def test_prepare_features(small_orders):
    X, y_late, y_delay = utils.prepare_features(small_orders)
    # 4 numeric features + 1 dummy for each of the 4 two-level categoricals
    assert X.shape == (6, 8)
    assert not X.isna().any().any()
    assert list(y_late) == [0, 0, 1, 0, 1, 1]
    assert list(y_delay) == [-2, 0, 4, 0, 5, 10]
    # Q4 flag: orders placed in October-December
    assert list(X["is_q4"]) == [0, 0, 1, 0, 1, 0]
    assert X["log_order_qty"].iloc[0] == pytest.approx(np.log1p(100))


def test_metric_functions():
    y_true = np.array([0, 0, 1, 1])
    perfect = utils.classification_metrics(y_true, np.array([0.1, 0.2, 0.8, 0.9]))
    for name in ["roc_auc", "accuracy", "precision", "recall", "f1"]:
        assert perfect[name] == pytest.approx(1.0)

    reg = utils.regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 5.0])
    assert reg["mae"] == pytest.approx(2 / 3)
    assert reg["rmse"] == pytest.approx(np.sqrt(4 / 3))
    perfect_reg = utils.regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert perfect_reg["r2"] == pytest.approx(1.0)

    # Majority class in train is 1, so accuracy is the share of 1s in test
    assert utils.majority_baseline_accuracy([1, 1, 0], [1, 0, 1, 1]) == pytest.approx(0.75)
