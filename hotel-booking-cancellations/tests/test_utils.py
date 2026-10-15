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
def small_df():
    """Joined dataset built from a small synthetic sample."""
    rng = np.random.default_rng(dg.SEED)
    guests = dg.generate_guests(rng, 200)
    bookings = dg.generate_bookings(rng, 600, guests)
    return utils.build_dataset(guests, bookings)


def test_build_dataset_adds_total_nights(small_df):
    expected = small_df["stay_nights_weekend"] + small_df["stay_nights_week"]
    assert (small_df["total_nights"] == expected).all()
    assert len(small_df) == 600
    assert not small_df["previous_stays"].isna().any()


def test_build_dataset_unknown_guest_raises():
    rng = np.random.default_rng(dg.SEED)
    guests = dg.generate_guests(rng, 20)
    bookings = dg.generate_bookings(rng, 30, guests)
    bookings.loc[0, "guest_id"] = "G99999"
    with pytest.raises(ValueError):
        utils.build_dataset(guests, bookings)


def test_split_xy_columns_and_missing(small_df):
    X, y = utils.split_xy(small_df)
    assert list(X.columns) == utils.NUMERIC_FEATURES + utils.CATEGORICAL_FEATURES
    assert utils.TARGET not in X.columns
    assert "booking_id" not in X.columns
    assert len(X) == len(y)
    with pytest.raises(ValueError):
        utils.split_xy(small_df.drop(columns=["adr"]))


def test_classification_metrics_known_values():
    y = [1, 0, 1, 0]
    proba = [0.9, 0.8, 0.3, 0.1]
    m = utils.classification_metrics(y, proba)
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    # A perfect ranking gives ROC AUC of 1.
    perfect = utils.classification_metrics([1, 1, 0, 0], [0.9, 0.6, 0.4, 0.1])
    assert perfect["roc_auc"] == pytest.approx(1.0)
    assert perfect["accuracy"] == pytest.approx(1.0)


def test_classification_metrics_validation():
    with pytest.raises(ValueError):
        utils.classification_metrics([1, 1, 1], [0.2, 0.5, 0.9])
    with pytest.raises(ValueError):
        utils.classification_metrics([0, 1], [0.5])


def test_top_fraction_lift():
    y = np.array([1, 1] + [0] * 8)
    proba = np.linspace(1, 0, 10)
    res = utils.top_fraction_lift(y, proba, 0.1)
    assert res["n_bookings"] == 1
    assert res["top_cancel_rate"] == pytest.approx(1.0)
    assert res["base_cancel_rate"] == pytest.approx(0.2)
    assert res["lift"] == pytest.approx(5.0)
    with pytest.raises(ValueError):
        utils.top_fraction_lift(y, proba, 0)


def test_error_rate_by_group():
    df = pd.DataFrame({"grp": ["a", "a", "b", "b"]})
    out = utils.error_rate_by_group(df, [1, 0, 1, 0], [0.9, 0.8, 0.3, 0.1], "grp")
    assert out.loc["a", "error_rate"] == pytest.approx(0.5)
    assert out.loc["a", "false_positive_rate_overall"] == pytest.approx(0.5)
    assert out.loc["a", "false_negative_rate_overall"] == pytest.approx(0.0)
    assert out.loc["b", "false_negative_rate_overall"] == pytest.approx(0.5)
    assert out["bookings"].sum() == 4


def test_pipelines_fit_and_predict(small_df):
    X, y = utils.split_xy(small_df)
    for make in (utils.make_logistic_pipeline, utils.make_gb_pipeline):
        pipe = make()
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)[:, 1]
        assert proba.shape == (len(X),)
        assert ((proba >= 0) & (proba <= 1)).all()
