"""Tests for the helper functions in src/utils.py (SYNTHETIC survey project)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

import utils


def test_calculate_nps_known_values():
    # 5 promoters, 3 passives, 2 detractors out of 10 -> (50% - 20%) = 30
    scores = [10, 9, 9, 10, 9, 7, 8, 7, 6, 0]
    assert utils.calculate_nps(scores) == pytest.approx(30.0)
    assert utils.calculate_nps([10, 10, 9]) == pytest.approx(100.0)
    assert utils.calculate_nps([0, 3, 6]) == pytest.approx(-100.0)
    assert utils.calculate_nps([7, 8, 7]) == pytest.approx(0.0)


def test_calculate_nps_edge_cases():
    assert np.isnan(utils.calculate_nps([]))
    with pytest.raises(ValueError):
        utils.calculate_nps([5, 11])
    with pytest.raises(ValueError):
        utils.calculate_nps([-1, 5])


def test_nps_category_boundaries():
    assert utils.nps_category(6) == "Detractor"
    assert utils.nps_category(7) == "Passive"
    assert utils.nps_category(8) == "Passive"
    assert utils.nps_category(9) == "Promoter"


def test_bootstrap_ci_contains_estimate_and_is_reproducible():
    scores = [10] * 50 + [8] * 30 + [3] * 20  # NPS = 30
    low, high = utils.bootstrap_nps_ci(scores, n_boot=500, seed=1)
    assert low <= utils.calculate_nps(scores) <= high
    assert -100 <= low < high <= 100
    assert (low, high) == utils.bootstrap_nps_ci(scores, n_boot=500, seed=1)


def test_bootstrap_difference_sign_and_interval():
    better = [10] * 60 + [8] * 20 + [4] * 20  # NPS = 40
    worse = [10] * 20 + [8] * 30 + [4] * 50   # NPS = -30
    diff, low, high = utils.bootstrap_nps_difference(better, worse, n_boot=500, seed=1)
    assert diff == pytest.approx(70.0)
    assert low <= diff <= high
    assert low > 0  # groups are far apart, so the CI excludes zero


def test_add_tenure_band_boundaries():
    df = pd.DataFrame({"tenure_months": [1, 12, 13, 24, 25, 48, 49, 72]})
    bands = utils.add_tenure_band(df)["tenure_band"].tolist()
    assert bands == ["1-12", "1-12", "13-24", "13-24", "25-48", "25-48", "49-72", "49-72"]


def test_nps_by_group_and_compare_groups():
    df = pd.DataFrame(
        {
            "grp": ["A"] * 60 + ["B"] * 60,
            "nps_score": [10] * 40 + [8] * 10 + [3] * 10 + [10] * 10 + [8] * 10 + [3] * 40,
        }
    )
    df["nps_category"] = df["nps_score"].map(utils.nps_category)

    table = utils.nps_by_group(df, "grp", n_boot=200, seed=1)
    assert list(table["group"]) == ["A", "B"]  # best group first
    assert table.loc[0, "nps"] == pytest.approx(50.0)
    assert table.loc[1, "nps"] == pytest.approx(-50.0)
    shares = table[["promoter_pct", "passive_pct", "detractor_pct"]].sum(axis=1)
    assert shares.tolist() == pytest.approx([100.0, 100.0])

    tests = utils.compare_groups(df, "grp")
    assert tests["chi2_p_value"] < 0.001
    assert tests["kruskal_p_value"] < 0.001


def test_extract_terms_and_keyword_frequency():
    terms = utils.extract_terms("Overall, great customer support.")
    assert {"great", "customer", "support", "great customer", "customer support"} <= terms
    assert "overall" not in terms  # stop word

    df = pd.DataFrame(
        {
            "nps_category": ["Detractor", "Detractor", "Detractor", "Promoter", "Detractor"],
            "comment": [
                "Frequent outages.",
                "Frequent outages and billing errors.",
                "Billing errors",
                "Easy to use",
                None,  # missing comments are ignored
            ],
        }
    )
    freq = utils.keyword_frequency(df, top_n=20)
    row = freq[(freq["category"] == "Detractor") & (freq["term"] == "frequent outages")]
    assert len(row) == 1
    assert row.iloc[0]["comments"] == 2
    assert row.iloc[0]["share_pct"] == pytest.approx(2 / 3 * 100)
    assert "easy" in set(freq.loc[freq["category"] == "Promoter", "term"])
