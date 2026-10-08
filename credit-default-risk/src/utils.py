"""Reusable helpers for the credit default analysis (SYNTHETIC data).

Contains feature preparation, model pipeline construction and metric functions
so they can be unit-tested independently of src/analysis.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

# Identifiers and the application date are deliberately NOT used as features
NUMERIC_FEATURES = [
    "age", "annual_income", "employment_years", "credit_score", "existing_debt",
    "num_credit_lines", "loan_amount", "term_months", "interest_rate",
    "debt_to_income", "num_delinquencies_2y",
]
CATEGORICAL_FEATURES = ["home_ownership", "region", "loan_purpose"]
TARGET = "default"


def load_model_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load both tables and join them into one row per loan."""
    customers = pd.read_csv(data_dir / "customers.csv")
    loans = pd.read_csv(data_dir / "loans.csv")
    return loans.merge(customers, on="customer_id", how="left", validate="m:1")


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the feature matrix X and the binary target y."""
    missing = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
               if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].astype(int)
    return X, y


def build_pipeline(estimator) -> Pipeline:
    """Wrap an estimator with scaling for numeric and one-hot for categorical columns."""
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("prep", preprocessor), ("model", estimator)])


def expected_cost(y_true, y_pred, cost_fn: float, cost_fp: float) -> float:
    """Total cost of errors: missed defaults (FN) and wrongly rejected good loans (FP)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fn * cost_fn + fp * cost_fp)


def cost_curve(y_true, y_score, cost_fn: float, cost_fp: float,
               thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """Total cost for each decision threshold (predict default if score >= threshold)."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    costs = [expected_cost(y_true, (y_score >= t).astype(int), cost_fn, cost_fp)
             for t in thresholds]
    return pd.DataFrame({"threshold": thresholds, "cost": costs})


def best_threshold(y_true, y_score, cost_fn: float, cost_fp: float) -> float:
    """Threshold with the lowest total cost (first one in case of ties)."""
    curve = cost_curve(y_true, y_score, cost_fn, cost_fp)
    return float(curve.loc[curve["cost"].idxmin(), "threshold"])


def compute_metrics(y_true, y_score, threshold: float = 0.5,
                    cost_fn: float = 5.0, cost_fp: float = 1.0) -> dict:
    """Ranking metrics (ROC AUC, PR AUC) and threshold-dependent metrics."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "total_cost": float(fn * cost_fn + fp * cost_fp),
    }
