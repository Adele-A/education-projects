"""Reusable helpers for the cancellation analysis (SYNTHETIC data).

Contains feature preparation, model pipeline builders and metric functions so
they can be unit tested separately from the reporting script.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "is_canceled"

# Predictors available at booking time. Identifiers and the raw arrival date
# are deliberately left out.
NUMERIC_FEATURES = [
    "lead_time_days", "arrival_month", "stay_nights_weekend", "stay_nights_week",
    "total_nights", "adults", "children", "adr", "special_requests",
    "parking_spaces", "booking_changes", "previous_stays", "previous_cancellations",
]
CATEGORICAL_FEATURES = [
    "hotel_type", "meal_plan", "market_segment", "deposit_type", "room_type",
    "country", "age_group",
]

LEAD_BINS = [-1, 7, 30, 90, 180, 450]
LEAD_LABELS = ["0-7", "8-30", "31-90", "91-180", "181-450"]


def build_dataset(guests: pd.DataFrame, bookings: pd.DataFrame) -> pd.DataFrame:
    """Join guest history onto bookings and add derived features."""
    df = bookings.merge(guests, on="guest_id", how="left", validate="many_to_one")
    if df["previous_stays"].isna().any():
        raise ValueError("Some bookings reference a guest_id missing from guests")
    df["total_nights"] = df["stay_nights_weekend"] + df["stay_nights_week"]
    return df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the feature matrix and the target."""
    missing = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
               if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy(), df[TARGET].astype(int)


def make_logistic_pipeline() -> Pipeline:
    """Baseline: scaled numerics + one-hot categoricals + logistic regression."""
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("pre", pre),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def make_gb_pipeline() -> Pipeline:
    """Gradient boosting classifier on one-hot encoded categoricals."""
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("pre", pre),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=200, max_depth=3,
            l2_regularization=1.0, random_state=42)),
    ])


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """Threshold-based and ranking metrics for predicted probabilities."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    if len(y_true) != len(proba):
        raise ValueError("y_true and proba must have the same length")
    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true must contain both classes")
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
        "brier_score": float(brier_score_loss(y_true, proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def top_fraction_lift(y_true, proba, fraction: float = 0.1) -> dict:
    """Cancellation rate among the highest-risk bookings versus the base rate."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    k = max(1, int(round(len(y_true) * fraction)))
    top = np.argsort(-proba, kind="stable")[:k]
    base = float(y_true.mean())
    top_rate = float(y_true[top].mean())
    return {
        "fraction": float(fraction),
        "n_bookings": int(k),
        "top_cancel_rate": top_rate,
        "base_cancel_rate": base,
        "lift": float(top_rate / base) if base > 0 else float("nan"),
    }


def error_rate_by_group(df: pd.DataFrame, y_true, proba, by: str,
                        threshold: float = 0.5) -> pd.DataFrame:
    """Share of misclassified bookings, false positives and false negatives per group."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba, dtype=float) >= threshold).astype(int)
    tmp = pd.DataFrame({
        "group": df[by].to_numpy(),
        "error": (pred != y_true).astype(int),
        "fp": ((pred == 1) & (y_true == 0)).astype(int),
        "fn": ((pred == 0) & (y_true == 1)).astype(int),
    })
    out = tmp.groupby("group", observed=True).agg(
        bookings=("error", "count"),
        error_rate=("error", "mean"),
        false_positive_rate_overall=("fp", "mean"),
        false_negative_rate_overall=("fn", "mean"),
    )
    out.index.name = by
    return out.round(3)
