"""Reusable helpers for the picking time analysis (SYNTHETIC data).

Contains feature preparation, metric functions and clustering helpers used by
src/analysis.py. Kept free of printing and file output so it is easy to test.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

# Paths are computed relative to the project root, not the working directory
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

TARGET = "pick_time_min"

# Numeric order columns used as regression features
NUMERIC_FEATURES = [
    "n_lines",
    "n_units",
    "total_weight_kg",
    "n_zones",
    "distance_m",
    "experience_months",
]
# Categorical columns that are one-hot encoded for regression
CATEGORICAL_FEATURES = ["shift", "priority", "contract_type"]

# Order-only columns used for clustering (no target, no picker information)
CLUSTER_FEATURES = ["n_lines", "n_units", "total_weight_kg", "n_zones", "distance_m"]
# Right-skewed columns that are log-transformed before scaling
CLUSTER_LOG_COLS = ["n_lines", "n_units", "total_weight_kg"]


def load_merged(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load orders and pickers and return them joined on picker_id."""
    pickers = pd.read_csv(data_dir / "pickers.csv")
    orders = pd.read_csv(data_dir / "orders.csv", parse_dates=["order_date"])
    return orders.merge(pickers, on="picker_id", how="left", validate="many_to_one")


def build_regression_data(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the feature matrix X (numeric + one-hot categorical) and target y."""
    X = pd.get_dummies(
        merged[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
        columns=CATEGORICAL_FEATURES,
        drop_first=True,
        dtype=float,
    )
    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].astype(float)
    return X, merged[TARGET].astype(float)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return MAE, RMSE and R2 as plain Python floats."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def prepare_cluster_features(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Log-transform skewed columns and standardize the clustering features.

    Returns the scaled matrix and the fitted scaler.
    """
    feats = df[CLUSTER_FEATURES].astype(float).copy()
    for col in CLUSTER_LOG_COLS:
        feats[col] = np.log1p(feats[col])
    scaler = StandardScaler()
    return scaler.fit_transform(feats), scaler


def evaluate_k_range(
    X: np.ndarray, k_values: range, random_state: int = 42
) -> pd.DataFrame:
    """Fit k-means for each k; return inertia and silhouette score per k."""
    rows = []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        rows.append(
            {
                "k": k,
                "inertia": float(km.inertia_),
                "silhouette": float(silhouette_score(X, labels)),
            }
        )
    return pd.DataFrame(rows)


def choose_k(scores: pd.DataFrame) -> int:
    """Pick the k with the highest silhouette score."""
    return int(scores.loc[scores["silhouette"].idxmax(), "k"])


def profile_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Summarize each cluster: size, share, mean order features and mean picking time."""
    tmp = df[CLUSTER_FEATURES + [TARGET]].copy()
    tmp["cluster"] = labels
    summary = tmp.groupby("cluster").mean()
    summary.insert(0, "share_pct", tmp["cluster"].value_counts(normalize=True) * 100)
    summary.insert(0, "orders", tmp["cluster"].value_counts())
    return summary.sort_values(TARGET).round(2)
