"""Tests for the helper functions in src/utils.py (small hand-made tables)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

import utils


def tiny_sessions() -> pd.DataFrame:
    """Four sessions with a known funnel: 4 -> 3 -> 2 -> 1 -> 1."""
    return pd.DataFrame(
        {
            "segment": ["a", "a", "b", "b"],
            "viewed_product": [1, 1, 1, 0],
            "added_to_cart": [1, 1, 0, 0],
            "reached_checkout": [1, 0, 0, 0],
            "purchased": [1, 0, 0, 0],
        }
    )


def test_step_transitions_counts_and_rates():
    trans = utils.step_transitions(tiny_sessions())
    assert list(trans["entered"]) == [4, 3, 2, 1]
    assert list(trans["continued"]) == [3, 2, 1, 1]
    assert list(trans["lost"]) == [1, 1, 1, 0]
    assert trans["step_conversion"].iloc[0] == pytest.approx(0.75)
    assert trans["drop_off"].iloc[3] == pytest.approx(0.0)
    # Step conversion and drop-off always add up to one
    assert (trans["step_conversion"] + trans["drop_off"]).tolist() == pytest.approx([1.0] * 4)


def test_segment_step_conversion_values():
    seg = utils.segment_step_conversion(tiny_sessions(), "segment")
    row = seg.loc["a"]
    assert row["sessions"] == 2
    assert row["session_to_view"] == pytest.approx(1.0)
    assert row["view_to_cart"] == pytest.approx(1.0)
    assert row["cart_to_checkout"] == pytest.approx(0.5)
    assert row["checkout_to_purchase"] == pytest.approx(1.0)
    assert row["purchase_rate"] == pytest.approx(0.5)


def test_two_proportion_and_chi_square_tests():
    same = utils.two_proportion_test(50, 100, 50, 100)
    assert same["difference"] == pytest.approx(0.0)
    assert same["p_value"] == pytest.approx(1.0)

    diff = utils.two_proportion_test(60, 100, 40, 100)
    assert diff["difference"] == pytest.approx(0.2)
    assert diff["ci_low"] < 0.2 < diff["ci_high"]
    assert diff["p_value"] < 0.01

    # Chi-square: an outcome fully determined by the segment gives Cramer's V of 1.
    # Three segments are used on purpose: scipy applies Yates' continuity
    # correction to 2x2 tables, which would lower V slightly (0.98 here).
    df = pd.DataFrame(
        {
            "seg": ["x"] * 40 + ["y"] * 30 + ["z"] * 30,
            "purchased": [1] * 40 + [0] * 60,
        }
    )
    res = utils.chi_square_test(df, "seg")
    assert res["p_value"] < 0.001
    assert res["cramers_v"] == pytest.approx(1.0)


def test_design_matrix_and_metrics():
    # Design matrix: reference levels dropped, reference row is all zeros
    df = pd.DataFrame(
        {
            "traffic_source": ["organic_search", "social"],
            "device": ["desktop", "mobile"],
            "user_type": ["new", "returning"],
        }
    )
    X = utils.build_design_matrix(df)
    assert X.shape == (2, 8)
    assert "traffic_source_organic_search" not in X.columns
    assert X.iloc[0].sum() == 0
    assert X.loc[1, ["traffic_source_social", "device_mobile", "user_type_returning"]].tolist() == [1, 1, 1]

    # Metrics: a perfectly separated example
    m = utils.classification_metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], threshold=0.5)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)

    # Baseline: constant prediction equal to the training purchase rate
    base = utils.baseline_metrics([1, 0, 0, 0], [0, 1])
    assert base["train_prevalence"] == pytest.approx(0.25)
    assert base["brier_score"] == pytest.approx(np.mean([(0.25 - 0) ** 2, (0.25 - 1) ** 2]))
