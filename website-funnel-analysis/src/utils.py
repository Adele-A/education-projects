"""Reusable helpers for the Website Funnel Analysis project (SYNTHETIC data).

Contains funnel calculations, statistical test wrappers, feature preparation
for the purchase model and classification metric functions. Kept free of
printing and file I/O so the functions can be unit tested.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from statsmodels.stats.proportion import proportions_ztest

# Ordered funnel steps; the first step (session start) is every row
FUNNEL_STEPS = ["viewed_product", "added_to_cart", "reached_checkout", "purchased"]

# Categorical model features. The first level of each list is the reference
# level (dropped from the dummy columns). Fixed lists keep the design matrix
# identical for train and test data.
CATEGORY_LEVELS = {
    "traffic_source": ["organic_search", "paid_search", "direct", "social",
                       "email", "referral"],
    "device": ["desktop", "mobile", "tablet"],
    "user_type": ["new", "returning"],
}


# --- Funnel calculations -----------------------------------------------------

def step_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """Step-by-step funnel: sessions entering, sessions continuing, drop-off.

    One row per transition (session -> viewed_product -> ... -> purchased).
    """
    steps = ["session"] + FUNNEL_STEPS
    counts = [len(df)] + [int(df[s].sum()) for s in FUNNEL_STEPS]
    rows = []
    for i in range(1, len(steps)):
        entered, continued = counts[i - 1], counts[i]
        rate = continued / entered if entered > 0 else np.nan
        rows.append(
            {
                "transition": f"{steps[i - 1]} -> {steps[i]}",
                "entered": entered,
                "continued": continued,
                "lost": entered - continued,
                "step_conversion": rate,
                "drop_off": 1 - rate if entered > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def segment_step_conversion(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Conditional step conversion rates per segment.

    Each column is the share of sessions that reached the previous step and
    also reached this one (the first column is relative to all sessions).
    """
    g = df.groupby(by)
    out = pd.DataFrame(
        {
            "sessions": g.size(),
            "session_to_view": g["viewed_product"].sum() / g.size(),
            "view_to_cart": g["added_to_cart"].sum() / g["viewed_product"].sum(),
            "cart_to_checkout": g["reached_checkout"].sum() / g["added_to_cart"].sum(),
            "checkout_to_purchase": g["purchased"].sum() / g["reached_checkout"].sum(),
            "purchase_rate": g["purchased"].mean(),
        }
    )
    return out.sort_values("purchase_rate", ascending=False)


# --- Statistical tests -------------------------------------------------------

def chi_square_test(df: pd.DataFrame, segment: str, outcome: str = "purchased") -> dict:
    """Chi-square test of independence between a segment column and a 0/1 outcome.

    Also returns Cramer's V as an effect size.
    """
    table = pd.crosstab(df[segment], df[outcome])
    chi2, p_value, dof, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * k))) if k > 0 and n > 0 else float("nan")
    return {
        "segment": segment,
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p_value),
        "cramers_v": cramers_v,
    }


def two_proportion_test(x1: int, n1: int, x2: int, n2: int) -> dict:
    """Two-sided two-proportion z-test with a normal-approximation 95% CI.

    Compares group 1 against group 2 (difference = p1 - p2).
    """
    p1, p2 = x1 / n1, x2 / n2
    z_stat, p_value = proportions_ztest([x1, x2], [n1, n2])
    diff = p1 - p2
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return {
        "rate_1": float(p1),
        "rate_2": float(p2),
        "difference": float(diff),
        "ci_low": float(diff - 1.96 * se),
        "ci_high": float(diff + 1.96 * se),
        "z_stat": float(z_stat),
        "p_value": float(p_value),
    }


def compare_groups(df: pd.DataFrame, column: str, group_1: str, group_2: str,
                   outcome: str = "purchased") -> dict:
    """Run the two-proportion test on the outcome rate of two segment values."""
    a = df.loc[df[column] == group_1, outcome]
    b = df.loc[df[column] == group_2, outcome]
    result = two_proportion_test(int(a.sum()), len(a), int(b.sum()), len(b))
    result.update({"column": column, "group_1": group_1, "group_2": group_2})
    return result


# --- Feature preparation -----------------------------------------------------

def build_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode source, device and user type (reference levels dropped).

    Column names look like 'traffic_source_social'. Values outside the known
    levels simply map to the reference level (all dummies zero).
    """
    X = pd.DataFrame(index=df.index)
    for col, levels in CATEGORY_LEVELS.items():
        for level in levels[1:]:
            X[f"{col}_{level}"] = (df[col] == level).astype(int)
    return X


# --- Metrics -----------------------------------------------------------------

def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Probability-based and threshold-based metrics for a binary classifier."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def baseline_metrics(y_train, y_test) -> dict:
    """Metrics of a constant model that always predicts the training purchase rate."""
    prevalence = float(np.mean(y_train))
    y_prob = np.full(len(y_test), prevalence)
    return {
        "train_prevalence": prevalence,
        "brier_score": float(brier_score_loss(y_test, y_prob)),
        "log_loss": float(log_loss(y_test, y_prob, labels=[0, 1])),
        "roc_auc": 0.5,
    }
