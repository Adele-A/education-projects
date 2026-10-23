"""Tests for the helper functions in src/utils.py.

Uses small in-memory data (a subset of the synthetic flights) and temporary
folders. No figures or pre-generated files are required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import data_generator as gen  # noqa: E402
import utils  # noqa: E402


@pytest.fixture(scope="module")
def model_frame():
    """Model frame built from freshly generated synthetic tables."""
    flights = gen.build_flights(np.random.default_rng(gen.SEED))
    return utils.build_model_frame(flights, gen.build_airports())


def test_build_model_frame_adds_features(model_frame):
    assert len(model_frame) == gen.N_FLIGHTS
    assert model_frame["congestion_factor"].notna().all()
    assert set(model_frame["is_weekend"].unique()) <= {0, 1}
    weekend_rows = model_frame["day_of_week"].isin(["Saturday", "Sunday"])
    assert (model_frame.loc[weekend_rows, "is_weekend"] == 1).all()
    assert (model_frame.loc[~weekend_rows, "is_weekend"] == 0).all()


def test_get_X_y_has_no_leakage(model_frame):
    X, y = utils.get_X_y(model_frame)
    assert list(X.columns) == utils.FEATURES
    assert "dep_delay_min" not in X.columns
    assert "arr_delay_min" not in X.columns
    assert len(X) == len(y)
    assert set(y.unique()) <= {0, 1}


def test_compute_metrics_perfect_and_worst():
    y = np.array([0, 1, 1, 0, 1])
    perfect = utils.compute_metrics(y, y, y.astype(float))
    for value in perfect.values():
        assert value == pytest.approx(1.0)
    wrong = utils.compute_metrics(y, 1 - y, 1.0 - y)
    assert wrong["accuracy"] == pytest.approx(0.0)
    assert wrong["roc_auc"] == pytest.approx(0.0)


def test_majority_baseline_accuracy():
    y_train = pd.Series([0, 0, 0, 1])
    y_test = pd.Series([0, 0, 1, 1])
    assert utils.majority_baseline_accuracy(y_train, y_test) == pytest.approx(0.5)


def test_chi_square_detects_association_and_independence():
    # Group "c" is always delayed, groups "a" and "b" never: perfect link
    dependent = pd.DataFrame({
        "grp": ["a"] * 10 + ["b"] * 10 + ["c"] * 20,
        utils.TARGET: [0] * 20 + [1] * 20,
    })
    res = utils.chi_square_delay_test(dependent, "grp")
    assert res["dof"] == 2
    assert res["p_value"] < 0.001
    assert res["cramers_v"] == pytest.approx(1.0)

    # Same delay rate in every group: no association
    independent = pd.DataFrame({
        "grp": ["a"] * 20 + ["b"] * 20 + ["c"] * 20,
        utils.TARGET: ([0] * 10 + [1] * 10) * 3,
    })
    res = utils.chi_square_delay_test(independent, "grp")
    assert res["cramers_v"] == pytest.approx(0.0, abs=1e-9)
    assert res["p_value"] > 0.9


def test_models_fit_and_predict_probabilities(model_frame):
    X, y = utils.get_X_y(model_frame.head(1000))
    models = utils.build_models()
    assert set(models) == {"logistic_regression", "random_forest"}
    for model in models.values():
        model.fit(X, y)
        proba = model.predict_proba(X.head(20))
        assert proba.shape == (20, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)


def test_load_tables_keeps_na_carrier_code(tmp_path):
    pd.DataFrame({"airport_code": ["ATL"], "city": ["Atlanta"],
                  "region": ["South"], "latitude": [33.64],
                  "longitude": [-84.43], "congestion_factor": [1.3]}
                 ).to_csv(tmp_path / "airports.csv", index=False)
    pd.DataFrame({"carrier_code": ["NA"], "carrier_name": ["Northern Air"],
                  "carrier_type": ["legacy"]}
                 ).to_csv(tmp_path / "carriers.csv", index=False)
    flights = gen.build_flights(np.random.default_rng(gen.SEED)).head(5)
    flights.to_csv(tmp_path / "flights.csv", index=False)

    airports, carriers, loaded = utils.load_tables(tmp_path)
    assert carriers.loc[0, "carrier_code"] == "NA"
    assert len(airports) == 1
    assert len(loaded) == 5
    assert loaded["carrier_code"].notna().all()
