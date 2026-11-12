"""Tests for the helper functions in src/utils.py.

Feature and metric tests use tiny hand-made tables whose expected values are
computed by hand. One test fits the BG/NBD model on freshly generated synthetic
data. No CSV files or figures are needed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import data_generator as gen  # noqa: E402
import utils  # noqa: E402


@pytest.fixture
def small_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Three customers; customer 3 is acquired after the cutoff used in the tests."""
    customers = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "acquisition_date": pd.to_datetime(["2024-01-01", "2024-01-10", "2024-03-10"]),
            "acquisition_channel": ["organic", "email", "social"],
            "country": ["US", "UK", "DE"],
        }
    )
    transactions = pd.DataFrame(
        {
            "transaction_id": [1, 2, 3, 4, 5, 6],
            "customer_id": [1, 2, 1, 1, 3, 1],
            "transaction_date": pd.to_datetime(
                ["2024-01-01", "2024-01-10", "2024-01-15", "2024-02-12", "2024-03-10", "2024-03-15"]
            ),
            "order_value": [10.0, 50.0, 20.0, 30.0, 5.0, 100.0],
        }
    )
    return customers, transactions


def test_build_rfm_hand_computed_values(small_tables):
    customers, transactions = small_tables
    rfm = utils.build_rfm(customers, transactions, "2024-03-01").set_index("customer_id")

    # customer 3 was acquired after the cutoff and must be excluded
    assert sorted(rfm.index) == [1, 2]

    # customer 1: orders on Jan 1, Jan 15, Feb 12 (the Mar 15 order is after the cutoff)
    assert rfm.loc[1, "n_orders"] == 3
    assert rfm.loc[1, "frequency"] == 2
    assert rfm.loc[1, "recency_weeks"] == pytest.approx(42 / 7)  # Jan 1 -> Feb 12
    assert rfm.loc[1, "T_weeks"] == pytest.approx(60 / 7)  # Jan 1 -> Mar 1 (2024 is a leap year)
    assert rfm.loc[1, "revenue"] == pytest.approx(60.0)
    assert rfm.loc[1, "avg_order_value"] == pytest.approx(20.0)

    # customer 2: a single order, so frequency and recency are zero
    assert rfm.loc[2, "frequency"] == 0
    assert rfm.loc[2, "recency_weeks"] == 0.0
    assert rfm.loc[2, "T_weeks"] == pytest.approx(51 / 7)  # Jan 10 -> Mar 1


def test_shrunk_aov_and_holdout_actuals(small_tables):
    customers, transactions = small_tables
    rfm = utils.build_rfm(customers, transactions, "2024-03-01").set_index("customer_id")

    # global mean = 110 / 4 orders = 27.5; weight on own mean = n / (n + 5)
    aov = utils.shrunk_avg_order_value(rfm)
    assert aov.loc[1] == pytest.approx((3 * 20.0 + 5 * 27.5) / 8)
    assert aov.loc[2] == pytest.approx((1 * 50.0 + 5 * 27.5) / 6)

    # holdout window (2024-03-01, 2024-03-31]: customer 1 and 3 ordered once each
    actual = utils.holdout_actuals(customers, transactions, "2024-03-01", "2024-03-31")
    actual = actual.set_index("customer_id")
    assert list(actual.index) == [1, 2, 3]  # customers without orders are kept with zeros
    assert actual["actual_orders"].tolist() == [1, 0, 1]
    assert actual["actual_revenue"].tolist() == pytest.approx([100.0, 0.0, 5.0])


def test_bgnbd_prediction_properties():
    # Hand-picked parameters (a > 1 as required by the expected-purchases formula)
    params = {"r": 0.5, "alpha": 5.0, "a": 1.5, "b": 2.5}
    x = np.array([0, 1, 3, 8])
    t_x = np.array([0.0, 4.0, 20.0, 40.0])
    T = np.array([30.0, 30.0, 45.0, 50.0])

    alive = utils.prob_alive(params, x, t_x, T)
    assert np.all((alive >= 0) & (alive <= 1))
    assert alive[0] == pytest.approx(1.0)  # no repeat purchases: no evidence of churn

    short = utils.expected_purchases(params, 10, x, t_x, T)
    long = utils.expected_purchases(params, 52, x, t_x, T)
    assert np.all(np.isfinite(short))
    assert np.all(short >= 0)
    assert np.all(long > short)  # a longer horizon gives more expected purchases

    # the log-likelihood is finite for every customer
    ll = utils.bgnbd_log_likelihood((0.5, 5.0, 1.5, 2.5), x, t_x, T)
    assert np.all(np.isfinite(ll))


def test_fit_bgnbd_on_generated_data():
    rng = np.random.default_rng(gen.SEED)
    customers = gen.generate_customers(rng)
    transactions = gen.generate_transactions(customers, rng)

    rfm = utils.build_rfm(customers, transactions, utils.OBSERVATION_END)
    assert len(rfm) == len(customers)

    params = utils.fit_bgnbd(rfm)
    assert params["n_customers"] == len(rfm)
    assert np.isfinite(params["log_likelihood"])
    assert params["r"] > 0 and params["alpha"] > 0 and params["b"] > 0
    assert params["a"] >= utils.A_FLOOR

    # predictions from the fitted model are valid for every customer
    x, t_x, T = rfm["frequency"], rfm["recency_weeks"], rfm["T_weeks"]
    alive = utils.prob_alive(params, x, t_x, T)
    pred = utils.expected_purchases(params, 52, x, t_x, T)
    assert np.all((alive >= 0) & (alive <= 1))
    assert np.all(np.isfinite(pred)) and np.all(pred >= 0)


def test_metrics_and_naive_baseline():
    y_true = np.array([0.0, 10.0, 20.0, 70.0])

    # perfect predictions: zero error, R^2 = 1, perfect ranking
    perfect = utils.regression_metrics(y_true, y_true)
    assert perfect["mae"] == pytest.approx(0.0)
    assert perfect["rmse"] == pytest.approx(0.0)
    assert perfect["r2"] == pytest.approx(1.0)
    assert perfect["total_error_pct"] == pytest.approx(0.0)
    assert utils.rank_correlation(y_true, y_true) == pytest.approx(1.0)

    # constant offset of +5 gives MAE = RMSE = 5 and a total error of +20%
    shifted = utils.regression_metrics(y_true, y_true + 5.0)
    assert shifted["mae"] == pytest.approx(5.0)
    assert shifted["rmse"] == pytest.approx(5.0)
    assert shifted["total_error_pct"] == pytest.approx(100.0 * 20.0 / 100.0)

    # top 25% of 4 customers = 1 customer; ranking by truth captures 70 / 100
    assert utils.top_share_captured(y_true, y_true, frac=0.25) == pytest.approx(0.70)
    # ranking in reverse order picks the customer with 0 revenue
    assert utils.top_share_captured(y_true, -y_true, frac=0.25) == pytest.approx(0.0)

    # naive baseline: repeat rate x / T extrapolated over the horizon; T = 0 gives 0
    naive = utils.naive_expected_purchases(10, np.array([2, 5, 0]), np.array([20.0, 50.0, 0.0]))
    assert naive.tolist() == pytest.approx([1.0, 1.0, 0.0])
