"""Tests for the statistical helpers in src/utils.py (small in-memory inputs)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

import utils


def test_ztest_and_chi_square_agree():
    x1, n1, x2, n2 = 100, 1000, 130, 1000
    z, p_z = utils.two_proportion_ztest(x1, n1, x2, n2)
    chi2, p_chi = utils.chi_square_test(x1, n1, x2, n2)
    assert z > 0  # variant (group 2) is higher
    assert chi2 == pytest.approx(z ** 2, rel=1e-6)
    assert p_chi == pytest.approx(p_z, rel=1e-6)
    # identical groups: no difference at all
    z0, p0 = utils.two_proportion_ztest(50, 500, 50, 500)
    assert z0 == pytest.approx(0.0)
    assert p0 == pytest.approx(1.0)


def test_confidence_intervals_contain_estimates():
    lo, hi = utils.wilson_ci(100, 1000)
    assert 0 <= lo < 0.10 < hi <= 1
    diff, d_lo, d_hi = utils.diff_confidence_interval(100, 1000, 130, 1000)
    assert diff == pytest.approx(0.03)
    assert d_lo < diff < d_hi
    uplift, u_lo, u_hi = utils.relative_uplift_ci(100, 1000, 130, 1000)
    assert uplift == pytest.approx(0.30)
    assert u_lo < uplift < u_hi
    # no conversions in a group: the relative uplift is undefined
    assert np.isnan(utils.relative_uplift_ci(0, 1000, 10, 1000)[0])


def test_power_and_sample_size_are_consistent():
    p1 = 0.10
    n_small_effect = utils.required_sample_size(p1, 0.12)
    n_large_effect = utils.required_sample_size(p1, 0.15)
    assert n_large_effect < n_small_effect  # bigger effects need fewer visitors
    # at the required size the power reaches the target
    assert utils.achieved_power(p1, 0.12, n_small_effect, n_small_effect) >= 0.80 - 1e-6
    # the minimum detectable rate is above baseline and gives about 80% power
    mde = utils.minimum_detectable_rate(p1, 2000)
    assert mde > p1
    assert utils.achieved_power(p1, mde, 2000, 2000) == pytest.approx(0.80, abs=0.01)
    # a perfectly balanced split has no sample ratio mismatch
    assert utils.sample_ratio_test(1000, 1000) == pytest.approx(1.0)


def test_feature_and_revenue_preparation():
    users = pd.DataFrame(
        {"user_id": [1, 2, 3], "group": ["control", "variant", "variant"],
         "converted": [0, 1, 0]}
    )
    orders = pd.DataFrame({"user_id": [2], "order_value": [49.5]})
    df = utils.add_revenue(utils.prepare_features(users), orders)
    assert df["is_variant"].tolist() == [0, 1, 1]
    assert df["revenue"].tolist() == [0.0, 49.5, 0.0]  # non-buyers count as 0
    assert len(df) == len(users)


def test_segment_breakdown_and_holm_adjustment():
    rows = []
    # one segment: control 10/100 converted, variant 20/100 converted
    for group, buyers in [("control", 10), ("variant", 20)]:
        converted = [1] * buyers + [0] * (100 - buyers)
        rows += [{"device": "mobile", "group": group, "converted": c} for c in converted]
    seg = utils.segment_breakdown(pd.DataFrame(rows), "device")
    assert len(seg) == 1
    row = seg.iloc[0]
    assert row["segment"] == "device: mobile"
    assert row["n_control"] == 100 and row["n_variant"] == 100
    assert row["diff"] == pytest.approx(0.10)
    assert row["ci_low"] < row["diff"] < row["ci_high"]

    raw = pd.Series([0.01, 0.04, 0.03])
    adjusted = utils.holm_adjust(raw)
    assert (adjusted >= raw.to_numpy()).all()
    assert (adjusted <= 1).all()
