"""Tests for the helper functions in src/utils.py.

The tests use small hand-made tables, so they are fast and do not need data files.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import utils  # noqa: E402


def make_orders() -> pd.DataFrame:
    """Three orders for two customers with known RFM values."""
    return pd.DataFrame(
        {
            "order_id": ["O1", "O2", "O3"],
            "customer_id": ["C1", "C1", "C2"],
            "order_date": pd.to_datetime(["2025-12-01", "2025-12-20", "2025-12-30"]),
            "order_value": [10.0, 20.0, 5.0],
        }
    )


def test_snapshot_and_compute_rfm():
    orders = make_orders()
    snapshot = utils.get_snapshot_date(orders)
    assert snapshot == pd.Timestamp("2025-12-31")

    rfm = utils.compute_rfm(orders, snapshot)
    assert rfm.loc["C1", "recency"] == 11
    assert rfm.loc["C1", "frequency"] == 2
    assert rfm.loc["C1", "monetary"] == 30.0
    assert rfm.loc["C2", "recency"] == 1
    assert rfm.loc["C2", "frequency"] == 1
    assert rfm.loc["C2", "monetary"] == 5.0


def test_rfm_scores_cover_all_quintiles_with_ties():
    # All customers have the same frequency: ties must still give scores 1-5
    rfm = pd.DataFrame(
        {
            "recency": np.arange(1, 11),
            "frequency": [1] * 10,
            "monetary": np.arange(10, 110, 10, dtype=float),
        },
        index=[f"C{i}" for i in range(10)],
    )
    scored = utils.add_rfm_scores(rfm)
    for col in ["r_score", "f_score", "m_score"]:
        assert set(scored[col]) == {1, 2, 3, 4, 5}
    # Recency scale is reversed: the most recent customer scores highest
    assert scored.loc["C0", "r_score"] == 5
    assert scored.loc["C9", "r_score"] == 1
    # Higher spend gives a higher monetary score
    assert scored.loc["C9", "m_score"] == 5
    assert (scored["rfm_total"] == scored[["r_score", "f_score", "m_score"]].sum(axis=1)).all()


def test_rule_based_segment_rules():
    assert utils.rule_based_segment(5, 5) == "Champions"
    assert utils.rule_based_segment(3, 3) == "Loyal"
    assert utils.rule_based_segment(1, 4) == "At Risk"
    assert utils.rule_based_segment(1, 1) == "Lapsed"
    assert utils.rule_based_segment(5, 1) == "New / Promising"
    assert utils.rule_based_segment(3, 2) == "Occasional"


def test_prepare_features_is_standardised():
    rng = np.random.default_rng(0)
    rfm = pd.DataFrame(
        {
            "recency": rng.integers(1, 400, 200),
            "frequency": rng.integers(1, 15, 200),
            "monetary": rng.lognormal(5, 1, 200),
        }
    )
    x = utils.prepare_features(rfm)
    assert x.shape == (200, 3)
    assert np.allclose(x.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(x.std(axis=0), 1, atol=1e-8)


def test_choose_k_and_relabel_by_monetary():
    k_scores = pd.DataFrame(
        {"k": [2, 3, 4], "inertia": [30.0, 20.0, 15.0], "silhouette": [0.30, 0.45, 0.40]}
    )
    assert utils.choose_k(k_scores) == 3

    rfm = pd.DataFrame({"monetary": [10.0, 10.0, 100.0, 100.0]})
    relabeled = utils.relabel_by_monetary(np.array([0, 0, 1, 1]), rfm)
    # The cluster with the highest mean spend becomes cluster 0
    assert relabeled.tolist() == [1, 1, 0, 0]


def test_kmeans_and_profile_on_two_clear_groups():
    rng = np.random.default_rng(1)
    low = pd.DataFrame(
        {
            "recency": rng.integers(200, 300, 30),
            "frequency": rng.integers(1, 3, 30),
            "monetary": rng.uniform(10, 50, 30),
        }
    )
    high = pd.DataFrame(
        {
            "recency": rng.integers(1, 20, 30),
            "frequency": rng.integers(10, 15, 30),
            "monetary": rng.uniform(800, 1200, 30),
        }
    )
    rfm = pd.concat([low, high], ignore_index=True)
    x = utils.prepare_features(rfm)

    labels = utils.relabel_by_monetary(utils.fit_kmeans(x, 2), rfm)
    # The two groups are well separated, so each must get one cluster label
    assert len(set(labels[:30])) == 1
    assert len(set(labels[30:])) == 1
    assert labels[0] != labels[-1]
    assert labels[-1] == 0  # high spenders are cluster 0

    quality = utils.clustering_quality(x, labels)
    assert quality["silhouette"] > 0.5

    profile = utils.profile_segments(rfm, labels)
    assert profile["customers"].tolist() == [30, 30]
    assert abs(profile["customer_share"].sum() - 1) < 1e-9
    assert abs(profile["revenue_share"].sum() - 1) < 1e-9
    assert profile.loc[0, "avg_monetary"] > profile.loc[1, "avg_monetary"]
