"""Reusable helpers for the fraud detection analysis (SYNTHETIC data).

Contains feature preparation and metric functions so they can be unit tested
separately from the analysis script.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

# Fixed category lists keep the one-hot columns identical across runs and splits
MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "fuel", "retail",
    "travel", "electronics", "entertainment", "health",
]
CHANNELS = ["pos", "online", "atm"]
NIGHT_HOURS = [0, 1, 2, 3, 4, 5, 22, 23]


def load_data(data_dir=DATA_DIR):
    """Load both tables; fail with a clear message if data is missing."""
    data_dir = Path(data_dir)
    cust_path = data_dir / "customers.csv"
    tx_path = data_dir / "transactions.csv"
    if not cust_path.exists() or not tx_path.exists():
        raise SystemExit("Data files not found. Run: python src/data_generator.py")
    customers = pd.read_csv(cust_path)
    transactions = pd.read_csv(tx_path, parse_dates=["timestamp"])
    return customers, transactions


def build_features(transactions, customers):
    """Join customer attributes and build a numeric feature matrix.

    Returns (X, y). The label is never used as a feature.
    """
    df = transactions.merge(customers, on="customer_id", how="left", validate="m:1")

    hour = df["timestamp"].dt.hour
    feats = pd.DataFrame(index=df.index)
    feats["log_amount"] = np.log1p(df["amount"])
    feats["amount_to_limit"] = df["amount"] / df["credit_limit"]
    feats["log_distance"] = np.log1p(df["distance_from_home_km"])
    feats["hour"] = hour
    feats["is_night"] = hour.isin(NIGHT_HOURS).astype(int)
    feats["is_foreign"] = df["is_foreign"]
    feats["new_device"] = df["new_device"]
    feats["age"] = df["age"]
    feats["account_age_days"] = df["account_age_days"]

    # One-hot encode with fixed categories
    channel = pd.Categorical(df["channel"], categories=CHANNELS)
    category = pd.Categorical(df["merchant_category"], categories=MERCHANT_CATEGORIES)
    feats = pd.concat([
        feats,
        pd.get_dummies(channel, prefix="channel").astype(int).set_axis(df.index),
        pd.get_dummies(category, prefix="cat").astype(int).set_axis(df.index),
    ], axis=1)

    return feats, df["is_fraud"].astype(int)


def top_k_metrics(y_true, scores, k):
    """Precision, recall and F1 when the k highest-scored rows are flagged."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    k = int(min(max(k, 1), len(scores)))
    flagged = np.argsort(-scores, kind="stable")[:k]
    tp = int(y_true[flagged].sum())
    total_pos = int(y_true.sum())
    precision = tp / k
    recall = tp / total_pos if total_pos else 0.0
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return {
        "flagged": k,
        "true_positives": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_scores(y_true, scores, alert_budget=0.02):
    """Ranking metrics (PR-AUC, ROC-AUC) plus precision/recall at an alert budget.

    alert_budget is the share of transactions the review team can check.
    """
    k = math.ceil(alert_budget * len(scores))
    result = {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "alert_budget": alert_budget,
    }
    result.update(top_k_metrics(y_true, scores, k))
    return result
