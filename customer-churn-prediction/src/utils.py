"""Reusable helpers for the Customer Churn Prediction project.

Contains feature preparation and metric functions used by src/analysis.py
and by the tests. All data in this project is SYNTHETIC.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "churned"

# Columns that identify a row or leak nothing useful for modelling
DROP_COLUMNS = ["customer_id", "signup_date"]

CATEGORICAL_FEATURES = ["country", "contract_type", "plan", "payment_method"]


def add_ticket_features(customers: pd.DataFrame, tickets: pd.DataFrame) -> pd.DataFrame:
    """Aggregate support tickets per customer and join them to the customers table.

    New columns:
      * n_tickets            - number of tickets (0 if the customer has none)
      * n_billing_tickets    - number of tickets in the Billing category (0 if none)
      * avg_resolution_hours - mean resolution time (NaN if no tickets)
      * avg_satisfaction     - mean satisfaction score, ignoring missing answers
                               (NaN if no scores are available)
    """
    grouped = tickets.groupby("customer_id")
    agg = pd.DataFrame({
        "n_tickets": grouped.size(),
        "n_billing_tickets": grouped["category"].apply(lambda s: int((s == "Billing").sum())),
        "avg_resolution_hours": grouped["resolution_hours"].mean(),
        "avg_satisfaction": grouped["satisfaction_score"].apply(
            lambda s: pd.to_numeric(s, errors="coerce").mean()),
    }).reset_index()

    merged = customers.merge(agg, on="customer_id", how="left")
    merged["n_tickets"] = merged["n_tickets"].fillna(0).astype(int)
    merged["n_billing_tickets"] = merged["n_billing_tickets"].fillna(0).astype(int)
    return merged


def build_features(customers: pd.DataFrame, tickets: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create the model matrix X and the target y.

    Adds ticket aggregates and one engineered feature, `weekly_minutes`
    (logins per week * average session length), as an engagement measure.
    """
    df = add_ticket_features(customers, tickets)
    df["weekly_minutes"] = df["avg_weekly_logins"] * df["avg_session_minutes"]

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET] + DROP_COLUMNS)
    return X, y


def split_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_columns, categorical_columns) for the feature matrix."""
    categorical = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    numeric = [c for c in X.columns if c not in categorical]
    return numeric, categorical


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Median-impute and scale numeric columns; one-hot encode categorical ones."""
    numeric, categorical = split_feature_types(X)
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict[str, float]:
    """Compute classification metrics from true labels and predicted probabilities.

    ROC AUC and average precision use the probabilities; accuracy, precision,
    recall and F1 use the given decision threshold.
    """
    y_pred = (pd.Series(y_proba).to_numpy() >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }
