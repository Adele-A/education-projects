"""Tests for the helper functions in src/utils.py (SYNTHETIC gym data)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import data_generator as dg  # noqa: E402
import utils  # noqa: E402


@pytest.fixture(scope="module")
def members():
    """Members table generated in memory with the project seed."""
    rng = np.random.default_rng(dg.SEED)
    members, signup, end_date = dg.generate_members(rng)
    members, _ = dg.generate_upgrades(rng, members, signup, end_date)
    return members


def test_upgrade_eligible_rules():
    df = pd.DataFrame({
        "initial_plan": ["Basic", "Premium", "Basic", "Standard", "Basic"],
        "signup_date": pd.to_datetime(
            ["2026-01-01", "2026-01-01", "2026-03-01", "2026-03-01", "2026-10-15"]),
        "churn_date": pd.to_datetime([None, None, "2026-03-31", "2026-05-30", None]),
    })
    mask = utils.upgrade_eligible(df).tolist()
    # Eligible; Premium; churned after 30 days; exactly 90 days; only 16 days active
    assert mask == [True, False, False, True, False]


def test_assign_risk_tier_thresholds():
    tiers = utils.assign_risk_tier([0.0, 0.249, 0.25, 0.499, 0.5, 0.99]).tolist()
    assert tiers == ["Low", "Low", "Medium", "Medium", "High", "High"]


def test_classification_metrics_and_deciles():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
    proba = np.array([0.1, 0.2, 0.15, 0.3, 0.7, 0.8, 0.9, 0.95, 0.05, 0.6])
    m = utils.classification_metrics(y, proba)
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["prevalence"] == pytest.approx(0.5)
    assert 0 <= m["brier_score"] < 0.1

    table = utils.risk_decile_table(y, proba, n_bins=5)
    assert len(table) == 5
    assert (table["members"] == 2).all()
    # Observed churn rises with the predicted risk bin
    assert table["observed_rate"].is_monotonic_increasing


def test_churn_model_pipeline_runs(members):
    X, y = utils.prepare_churn_features(members)
    # No leakage columns in the features
    for col in ["tenure_months", "churn_date", "current_plan", "monthly_fee"]:
        assert col not in X.columns
    assert len(X) == len(y) == len(members)

    model = utils.build_churn_model(random_state=42)
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.min() >= 0 and proba.max() <= 1
    # In-sample ranking should be clearly better than random
    assert utils.classification_metrics(y, proba)["roc_auc"] > 0.6

    coefs = utils.model_coefficients(model)
    assert {"feature", "coefficient", "odds_ratio"} <= set(coefs.columns)
    assert np.allclose(coefs["odds_ratio"], np.exp(coefs["coefficient"]))
    # More visits per week should lower churn risk
    visits = coefs.loc[coefs["feature"] == "avg_visits_per_week", "coefficient"].iloc[0]
    assert visits < 0
