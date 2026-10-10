"""Tests for the metric functions in src/utils.py.

A tiny hand-made dataset (SYNTHETIC) with known answers is used, so every
expected number can be checked by hand. One integration test runs the bridge
on the generated data.

Run from the project root:
    python -m pytest -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import data_generator as dg  # noqa: E402
import utils  # noqa: E402


@pytest.fixture
def customers():
    return pd.DataFrame({
        "customer_id": [1, 2],
        "signup_date": pd.to_datetime(["2023-01-10", "2023-02-01"]),
        "initial_plan": ["Starter", "Growth"],
    })


@pytest.fixture
def events():
    """Customer 1: new 100 (Jan), +50 (Feb), churn (Mar).
    Customer 2: new 200 (Feb), -50 (Mar)."""
    rows = [
        (1, 1, "2023-01-10", "new", 0.0, 100.0),
        (2, 2, "2023-02-01", "new", 0.0, 200.0),
        (3, 1, "2023-02-15", "expansion", 100.0, 150.0),
        (4, 1, "2023-03-05", "churn", 150.0, 0.0),
        (5, 2, "2023-03-10", "contraction", 200.0, 150.0),
    ]
    df = pd.DataFrame(rows, columns=["event_id", "customer_id", "event_date",
                                     "event_type", "mrr_before", "mrr_after"])
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["mrr_change"] = df["mrr_after"] - df["mrr_before"]
    return df


def test_load_data_reads_csv_and_reports_missing(tmp_path, customers, events):
    with pytest.raises(FileNotFoundError):
        utils.load_data(tmp_path)
    customers.to_csv(tmp_path / "customers.csv", index=False)
    events.to_csv(tmp_path / "subscription_events.csv", index=False)
    c, e = utils.load_data(tmp_path)
    assert len(c) == 2 and len(e) == 5
    assert pd.api.types.is_datetime64_any_dtype(c["signup_date"])
    assert pd.api.types.is_datetime64_any_dtype(e["event_date"])


def test_monthly_mrr_bridge_values(events):
    bridge = utils.monthly_mrr_bridge(events)
    assert len(bridge) == 3
    assert bridge["opening_mrr"].tolist() == [0.0, 100.0, 350.0]
    assert bridge["new_mrr"].tolist() == [100.0, 200.0, 0.0]
    assert bridge["expansion_mrr"].tolist() == [0.0, 50.0, 0.0]
    assert bridge["contraction_mrr"].tolist() == [0.0, 0.0, -50.0]
    assert bridge["churned_mrr"].tolist() == [0.0, 0.0, -150.0]
    assert bridge["closing_mrr"].tolist() == [100.0, 350.0, 150.0]
    assert bridge["active_start"].tolist() == [0, 1, 2]
    assert bridge["active_end"].tolist() == [1, 2, 1]


def test_bridge_reconciles_with_customer_matrix(events):
    bridge = utils.monthly_mrr_bridge(events)
    matrix = utils.customer_mrr_matrix(events, bridge.index)
    assert matrix.sum(axis=0).tolist() == [100.0, 350.0, 150.0]
    assert utils.bridge_reconciles(bridge, matrix)

    # A tampered bridge must fail the check.
    broken = bridge.copy()
    broken.loc[broken.index[-1], "closing_mrr"] += 10.0
    assert not utils.bridge_reconciles(broken, matrix)


def test_monthly_rates_and_overall_summary(events):
    bridge = utils.monthly_mrr_bridge(events)
    rates = utils.add_monthly_rates(bridge)

    # First month has no opening base, so rates are NaN.
    assert np.isnan(rates["nrr"].iloc[0])
    assert rates["nrr"].iloc[1] == pytest.approx(1.5)
    assert rates["grr"].iloc[1] == pytest.approx(1.0)
    assert rates["customer_churn_rate"].iloc[2] == pytest.approx(0.5)
    assert rates["revenue_churn_rate"].iloc[2] == pytest.approx(150 / 350)
    assert rates["grr"].iloc[2] == pytest.approx(150 / 350)

    summary = utils.summarize_retention(bridge)
    assert summary["months_used"] == 2
    assert summary["avg_monthly_nrr"] == pytest.approx(300 / 450)
    assert summary["avg_monthly_grr"] == pytest.approx(250 / 450)
    assert summary["avg_monthly_customer_churn_rate"] == pytest.approx(1 / 3)
    assert summary["avg_monthly_revenue_churn_rate"] == pytest.approx(150 / 450)


def test_segment_retention(events, customers):
    seg = utils.segment_retention(events, customers, "initial_plan")
    assert set(seg.index) == {"Starter", "Growth"}
    assert float(seg.loc["Starter", "customers"]) == 1
    assert float(seg.loc["Starter", "final_mrr"]) == 0.0
    assert float(seg.loc["Starter", "avg_monthly_nrr"]) == pytest.approx(0.6)
    assert float(seg.loc["Starter", "avg_monthly_grr"]) == pytest.approx(0.4)
    # Growth only has an opening base in March.
    assert float(seg.loc["Growth", "months_used"]) == 1
    assert float(seg.loc["Growth", "avg_monthly_grr"]) == pytest.approx(0.75)
    assert float(seg.loc["Growth", "final_mrr"]) == 150.0


def test_cohort_revenue_and_retention(events, customers):
    bridge = utils.monthly_mrr_bridge(events)
    matrix = utils.customer_mrr_matrix(events, bridge.index)
    cohort = utils.cohort_revenue(customers, matrix)

    assert cohort.loc[pd.Period("2023-01", "M")].tolist()[:3] == [100.0, 150.0, 0.0]
    feb = cohort.loc[pd.Period("2023-02", "M")]
    assert feb[0] == 200.0 and feb[1] == 150.0
    assert np.isnan(feb[2])  # offset not reached yet

    pct = utils.cohort_retention_pct(cohort)
    assert pct.loc[pd.Period("2023-01", "M"), 1] == pytest.approx(150.0)
    assert (pct[0] == 100.0).all()

    assert utils.pooled_cohort_retention(cohort, 1) == pytest.approx(100.0)
    assert utils.pooled_cohort_retention(cohort, 2) == pytest.approx(0.0)
    assert np.isnan(utils.pooled_cohort_retention(cohort, 99))


def test_bridge_reconciles_on_generated_data():
    customers, events = dg.generate()
    bridge = utils.monthly_mrr_bridge(events)
    matrix = utils.customer_mrr_matrix(events, bridge.index)
    assert utils.bridge_reconciles(bridge, matrix)
    assert len(bridge) == dg.N_MONTHS
    assert bridge["active_end"].iloc[-1] == matrix.iloc[:, -1].gt(0).sum()
    assert bridge["closing_mrr"].iloc[-1] == pytest.approx(events["mrr_change"].sum())
