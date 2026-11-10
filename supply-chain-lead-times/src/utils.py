"""Reusable helpers for the supply chain lead time analysis.

Contains KPI calculations, statistical test wrappers, feature preparation
and metric functions. Kept free of file I/O and plotting so that the
functions can be unit tested. All data is SYNTHETIC.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

# Categorical order/supplier attributes used as model features
CATEGORICAL_FEATURES = ["transport_mode", "reliability_tier", "region", "category"]


def coefficient_of_variation(values: pd.Series) -> float:
    """Return std / mean of a series (sample std, ddof=1).

    Returns NaN if the series has fewer than two values or a zero mean.
    """
    values = pd.Series(values).dropna()
    if len(values) < 2 or values.mean() == 0:
        return float("nan")
    return float(values.std(ddof=1) / values.mean())


def segment_kpis(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Compute on-time and lead time KPIs for each value of a column.

    Args:
        df: Orders merged with supplier attributes (needs po_id, on_time,
            delay_days and actual_lead_time_days columns).
        by: Column name to group by.

    Returns:
        DataFrame indexed by the group value with columns orders,
        on_time_rate, mean_delay, mean_lead_time, std_lead_time and
        cv_lead_time.
    """
    grouped = df.groupby(by).agg(
        orders=("po_id", "count"),
        on_time_rate=("on_time", "mean"),
        mean_delay=("delay_days", "mean"),
        mean_lead_time=("actual_lead_time_days", "mean"),
        std_lead_time=("actual_lead_time_days", "std"),
    )
    grouped["cv_lead_time"] = grouped["std_lead_time"] / grouped["mean_lead_time"]
    return grouped


def supplier_kpis(orders: pd.DataFrame, suppliers: pd.DataFrame) -> pd.DataFrame:
    """Build one KPI row per supplier and attach supplier attributes.

    Args:
        orders: Purchase orders table.
        suppliers: Supplier master table.

    Returns:
        DataFrame indexed by supplier_id with order count, on-time rate,
        delay and lead time statistics, plus country, region, category,
        transport_mode and reliability_tier.
    """
    kpis = segment_kpis(orders, "supplier_id")
    attrs = suppliers.set_index("supplier_id")[
        ["country", "region", "category", "transport_mode", "reliability_tier"]
    ]
    return kpis.join(attrs, how="left")


def chi_square_on_time(df: pd.DataFrame, by: str) -> dict:
    """Chi-square test of independence between a segment and the on_time flag.

    Args:
        df: Table with an on_time column (0/1) and the grouping column.
        by: Column to test against on_time.

    Returns:
        Dict with chi2 statistic, p_value and degrees of freedom (dof).
    """
    table = pd.crosstab(df[by], df["on_time"])
    chi2, p_value, dof, _ = stats.chi2_contingency(table)
    return {"chi2": float(chi2), "p_value": float(p_value), "dof": int(dof)}


def kruskal_delay(df: pd.DataFrame, by: str) -> dict:
    """Kruskal-Wallis test of delay_days across the groups of a column.

    Returns:
        Dict with the H statistic and p_value.
    """
    samples = [g["delay_days"].to_numpy() for _, g in df.groupby(by)]
    h_stat, p_value = stats.kruskal(*samples)
    return {"h_stat": float(h_stat), "p_value": float(p_value)}


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Create the model matrix and targets from orders merged with suppliers.

    Features: promised lead time, expedited flag, log of order quantity,
    Q4 flag (order placed October-December) and one-hot encoded transport
    mode, reliability tier, region and category (first level dropped).

    Args:
        df: Orders merged with supplier attributes.

    Returns:
        Tuple (X, y_late, y_delay) where y_late is 1 for late orders
        (on_time == 0) and y_delay is delay_days.
    """
    order_date = pd.to_datetime(df["order_date"])
    numeric = pd.DataFrame(
        {
            "promised_lead_time_days": df["promised_lead_time_days"].astype(float),
            "is_expedited": df["is_expedited"].astype(int),
            "log_order_qty": np.log1p(df["order_qty"].astype(float)),
            "is_q4": (order_date.dt.month >= 10).astype(int),
        },
        index=df.index,
    )
    dummies = pd.get_dummies(df[CATEGORICAL_FEATURES], drop_first=True, dtype=int)
    X = pd.concat([numeric, dummies], axis=1)
    y_late = (1 - df["on_time"]).astype(int)
    y_delay = df["delay_days"].astype(float)
    return X, y_late, y_delay


def regression_metrics(y_true, y_pred) -> dict:
    """Return MAE, RMSE and R^2 for a regression prediction."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """Return ROC AUC, accuracy, precision, recall and F1 for probabilities.

    Args:
        y_true: True 0/1 labels.
        proba: Predicted probability of the positive class.
        threshold: Probability cut-off for the hard label.
    """
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def majority_baseline_accuracy(y_train, y_test) -> float:
    """Accuracy on y_test of always predicting the most common train class."""
    majority = int(pd.Series(y_train).mode().iloc[0])
    return float((np.asarray(y_test) == majority).mean())
