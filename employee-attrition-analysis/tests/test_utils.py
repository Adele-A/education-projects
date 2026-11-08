"""Tests for the reusable helpers in src/utils.py.

Small synthetic tables are generated in memory (no CSV files, no figures),
so the tests run in a few seconds.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import data_generator as dg  # noqa: E402
import utils  # noqa: E402


def make_joined_data(n: int = 800, seed: int = 5) -> pd.DataFrame:
    """Generate employees, drop some survey rows and join them."""
    rng = np.random.default_rng(seed)
    employees = dg.generate_employees(rng, n)
    surveys_all = dg.generate_surveys(rng, employees)
    employees = dg.add_attrition(rng, employees, surveys_all)
    responded = rng.random(n) < 0.9
    surveys = surveys_all[responded].reset_index(drop=True)
    return utils.merge_data(employees, surveys)


def test_merge_data_keeps_all_employees_with_missing_surveys():
    employees = pd.DataFrame({"employee_id": [1, 2, 3], "attrition": [0, 1, 0]})
    surveys = pd.DataFrame(
        {"employee_id": [1, 3], "job_satisfaction": [2, 4],
         "work_life_balance": [3, 1], "manager_support": [2, 2]}
    )
    merged = utils.merge_data(employees, surveys)
    assert len(merged) == 3
    assert merged.loc[merged["employee_id"] == 2, "job_satisfaction"].isna().all()


def test_merge_data_rejects_duplicate_survey_ids():
    employees = pd.DataFrame({"employee_id": [1, 2], "attrition": [0, 1]})
    surveys = pd.DataFrame(
        {"employee_id": [1, 1], "job_satisfaction": [2, 3],
         "work_life_balance": [3, 3], "manager_support": [2, 2]}
    )
    with pytest.raises(pd.errors.MergeError):
        utils.merge_data(employees, surveys)


def test_split_features_target_shapes_and_columns():
    df = make_joined_data(n=200)
    X, y = utils.split_features_target(df)
    assert list(X.columns) == utils.FEATURES
    assert len(X) == len(y) == len(df)
    assert utils.TARGET not in X.columns
    assert set(y.unique()) <= {0, 1}


def test_compute_metrics_known_values():
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.6, 0.4, 0.9]
    m = utils.compute_metrics(y_true, y_prob, threshold=0.5)
    # Predictions at 0.5: [0, 1, 0, 1] -> one of each confusion cell
    assert (m["true_negatives"], m["false_positives"]) == (1, 1)
    assert (m["false_negatives"], m["true_positives"]) == (1, 1)
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    # 3 of the 4 positive/negative pairs are ranked correctly
    assert m["roc_auc"] == pytest.approx(0.75)


def test_best_f1_threshold_on_separable_data():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    threshold = utils.best_f1_threshold(y_true, y_prob)
    assert 0.2 < threshold <= 0.8
    assert utils.compute_metrics(y_true, y_prob, threshold)["f1"] == pytest.approx(1.0)


def test_model_pipeline_and_odds_ratio_table():
    df = make_joined_data()
    X, y = utils.split_features_target(df)
    model = utils.build_model()
    model.fit(X, y)

    prob = model.predict_proba(X)[:, 1]
    assert prob.shape == (len(X),)
    assert ((prob >= 0) & (prob <= 1)).all()

    table = utils.odds_ratio_table(model, X, y)
    # 10 numeric features + 5 department + 2 gender + 1 overtime dummies
    assert len(table) == 18
    assert "const" not in set(table["feature"])
    assert (table["odds_ratio"] > 0).all()
    assert (table["ci_low"] <= table["odds_ratio"]).all()
    assert (table["odds_ratio"] <= table["ci_high"]).all()
