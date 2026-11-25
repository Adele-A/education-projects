"""Reusable helpers for the gym membership analysis (SYNTHETIC data).

Contains feature preparation, model construction, metric functions and
small helpers that are used by src/analysis.py and can be unit tested.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SNAPSHOT_DATE = pd.Timestamp("2026-10-31")
MIN_UPGRADE_TENURE_DAYS = 90  # members need at least 90 days to be upgrade-eligible

# Features available for the churn model. tenure_months, churn_date and
# current_plan / monthly_fee are excluded on purpose: they are derived from the
# outcome (or from the upgrade) and would leak information.
NUMERIC_FEATURES = [
    "avg_visits_per_week",
    "days_since_last_visit",
    "distance_to_gym_km",
    "age",
    "has_personal_trainer",
    "uses_group_classes",
]
CATEGORICAL_FEATURES = ["initial_plan", "contract_type", "payment_method"]
TARGET = "is_churned"

# Default probability cut-offs for the risk tiers
MEDIUM_RISK_THRESHOLD = 0.25
HIGH_RISK_THRESHOLD = 0.50


def load_data(data_dir):
    """Load members and upgrades tables with parsed date columns."""
    data_dir = Path(data_dir)
    members = pd.read_csv(data_dir / "members.csv", parse_dates=["signup_date", "churn_date"])
    upgrades = pd.read_csv(data_dir / "upgrades.csv", parse_dates=["upgrade_date"])
    return members, upgrades


def membership_end(members):
    """End of membership: churn date if churned, otherwise the snapshot date."""
    return members["churn_date"].fillna(SNAPSHOT_DATE)


def upgrade_eligible(members):
    """Boolean mask: not Premium at sign-up and at least 90 days of membership."""
    tenure_days = (membership_end(members) - members["signup_date"]).dt.days
    return (members["initial_plan"] != "Premium") & (tenure_days >= MIN_UPGRADE_TENURE_DAYS)


def add_upgrade_flag(members, upgrades):
    """Return a copy of members with an `upgraded` column (1 if an upgrade event exists)."""
    out = members.copy()
    out["upgraded"] = out["member_id"].isin(upgrades["member_id"]).astype(int)
    return out


def prepare_churn_features(members):
    """Return the feature matrix X and the target y for the churn model."""
    X = members[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = members[TARGET].astype(int)
    return X, y


def build_churn_model(random_state=42):
    """Interpretable churn model: scaled numeric + one-hot categorical -> logistic regression."""
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first"), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
    ])


def model_coefficients(model):
    """Return a DataFrame of coefficients and odds ratios sorted by absolute size."""
    names = model.named_steps["prep"].get_feature_names_out()
    coef = model.named_steps["clf"].coef_[0]
    table = pd.DataFrame({"feature": names, "coefficient": coef})
    table["odds_ratio"] = np.exp(table["coefficient"])
    order = table["coefficient"].abs().sort_values(ascending=False).index
    return table.loc[order].reset_index(drop=True)


def classification_metrics(y_true, proba, threshold=0.5):
    """Compute standard binary classification metrics from predicted probabilities."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "prevalence": float(y_true.mean()),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
        "brier_score": float(brier_score_loss(y_true, proba)),
    }


def assign_risk_tier(proba, medium=MEDIUM_RISK_THRESHOLD, high=HIGH_RISK_THRESHOLD):
    """Map churn probabilities to Low / Medium / High tiers."""
    proba = np.asarray(proba)
    return np.select([proba >= high, proba >= medium], ["High", "Medium"], default="Low")


def risk_decile_table(y_true, proba, n_bins=10):
    """Observed churn rate vs mean predicted probability per risk bin (1 = lowest risk)."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba)})
    # rank(method="first") breaks ties so equal-sized bins can always be formed
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False) + 1
    table = (
        df.groupby("bin")
        .agg(members=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
        .reset_index()
    )
    return table
