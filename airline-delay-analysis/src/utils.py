"""Reusable helpers for the SYNTHETIC airline delay analysis.

Contains data loading, feature preparation, model construction, metric
functions and a chi-square helper. Kept free of printing and plotting so the
functions can be unit-tested.

The data is synthetic and does not describe real flights or airlines.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

RANDOM_STATE = 42
TARGET = "is_delayed"

# Only information known before departure is used as a feature.
# dep_delay_min and arr_delay_min are deliberately excluded (target leakage).
NUMERIC_FEATURES = ["scheduled_dep_hour", "distance_miles", "weather_severity",
                    "congestion_factor", "month", "is_weekend"]
CATEGORICAL_FEATURES = ["carrier_code", "origin"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_tables(data_dir: str | Path = DATA_DIR
                ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load airports, carriers and flights from CSV files.

    Returns the three tables in this order. The carrier code "NA" is kept
    as a string, not read as a missing value.
    """
    data_dir = Path(data_dir)
    # keep_default_na=False: carrier code "NA" must not be read as missing
    airports = pd.read_csv(data_dir / "airports.csv", keep_default_na=False)
    carriers = pd.read_csv(data_dir / "carriers.csv", keep_default_na=False)
    flights = pd.read_csv(data_dir / "flights.csv", keep_default_na=False,
                          na_values=[""])
    return airports, carriers, flights


def build_model_frame(flights: pd.DataFrame,
                      airports: pd.DataFrame) -> pd.DataFrame:
    """Add pre-departure features: origin congestion and a weekend flag."""
    df = flights.merge(
        airports[["airport_code", "congestion_factor"]],
        left_on="origin", right_on="airport_code", how="left",
    ).drop(columns="airport_code")
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
    return df


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a model frame into the feature matrix and the target."""
    return df[FEATURES].copy(), df[TARGET].astype(int)


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric features and one-hot encode categorical ones."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ])


def build_models() -> dict[str, Pipeline]:
    """Return the baseline and the tree-based model as sklearn pipelines."""
    logreg = Pipeline([
        ("prep", build_preprocessor()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced",
                                     random_state=RANDOM_STATE)),
    ])
    forest = Pipeline([
        ("prep", build_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=1)),
    ])
    return {"logistic_regression": logreg, "random_forest": forest}


def compute_metrics(y_true: ArrayLike, y_pred: ArrayLike,
                    y_proba: ArrayLike) -> dict[str, float]:
    """Classification metrics for the positive class (delayed = 1)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def majority_baseline_accuracy(y_train: pd.Series,
                               y_test: pd.Series) -> float:
    """Accuracy of always predicting the most frequent training class."""
    majority = int(y_train.mean() >= 0.5)
    return float((np.asarray(y_test) == majority).mean())


def chi_square_delay_test(df: pd.DataFrame,
                          column: str) -> dict[str, str | float | int]:
    """Chi-square test of independence between a column and the delay flag.

    Returns chi2, p-value, degrees of freedom and Cramer's V (effect size).
    """
    table = pd.crosstab(df[column], df[TARGET])
    chi2, p_value, dof, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * k))) if k > 0 and n > 0 else 0.0
    return {"column": column, "chi2": float(chi2), "p_value": float(p_value),
            "dof": int(dof), "cramers_v": cramers_v}
