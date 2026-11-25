"""Reusable helpers for the fraud detection analysis (SYNTHETIC data).

Contains feature preparation and metric functions so they can be unit tested
separately from the analysis script.
"""

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
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


def load_data(data_dir: str | Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both tables; fail with a clear message if data is missing.

    Args:
        data_dir: Folder that contains customers.csv and transactions.csv.

    Returns:
        A tuple (customers, transactions). The timestamp column of
        transactions is parsed as datetime.

    Raises:
        SystemExit: If either CSV file does not exist.
    """
    data_dir = Path(data_dir)
    cust_path = data_dir / "customers.csv"
    tx_path = data_dir / "transactions.csv"
    if not cust_path.exists() or not tx_path.exists():
        raise SystemExit("Data files not found. Run: python src/data_generator.py")
    customers = pd.read_csv(cust_path)
    transactions = pd.read_csv(tx_path, parse_dates=["timestamp"])
    return customers, transactions


def build_features(
    transactions: pd.DataFrame, customers: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """Join customer attributes and build a numeric feature matrix.

    Args:
        transactions: Transactions table with a datetime timestamp column.
        customers: Customers table keyed by customer_id.

    Returns:
        (X, y): feature DataFrame and the integer fraud label Series.
        The label is never used as a feature.
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


def top_k_metrics(y_true: ArrayLike, scores: ArrayLike, k: int) -> dict[str, Any]:
    """Precision, recall and F1 when the k highest-scored rows are flagged.

    Args:
        y_true: Binary labels (1 = fraud).
        scores: Anomaly or fraud scores; higher means more suspicious.
        k: Number of rows to flag. Clipped to the range [1, len(scores)].

    Returns:
        Dictionary with keys flagged, true_positives, precision, recall, f1.
    """
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


def evaluate_scores(
    y_true: ArrayLike, scores: ArrayLike, alert_budget: float = 0.02
) -> dict[str, Any]:
    """Ranking metrics (PR-AUC, ROC-AUC) plus precision/recall at an alert budget.

    Args:
        y_true: Binary labels (1 = fraud).
        scores: Anomaly or fraud scores; higher means more suspicious.
        alert_budget: Share of transactions the review team can check.

    Returns:
        Dictionary with pr_auc, roc_auc, alert_budget and the keys returned
        by top_k_metrics for the flagged share of transactions.
    """
    k = math.ceil(alert_budget * len(scores))
    result = {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "alert_budget": alert_budget,
    }
    result.update(top_k_metrics(y_true, scores, k))
    return result
