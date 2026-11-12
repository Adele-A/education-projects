"""Reusable helpers for the CLV analysis (SYNTHETIC data).

Contains:
  * data loading;
  * RFM-style feature preparation (frequency, recency, age T, average order value);
  * a BG/NBD model (log-likelihood, fitting with scipy, expected purchases,
    probability of being alive);
  * evaluation metrics.

Time inside the BG/NBD model is measured in WEEKS. The model follows
Fader, Hardie and Lee (2005), implemented here directly with numpy/scipy.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, hyp2f1
from scipy.stats import spearmanr

# End of the observation window (same as in the data generator)
OBSERVATION_END = pd.Timestamp("2025-12-31")

# The BG/NBD expected-purchases formula needs a > 1, so the parameter is kept above this floor
A_FLOOR = 1.05

# Shrinkage strength for the average order value (pseudo-orders at the global mean)
AOV_SHRINKAGE_ORDERS = 5.0


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both CSV files and parse the date columns."""
    customers_path = Path(data_dir) / "customers.csv"
    transactions_path = Path(data_dir) / "transactions.csv"
    for path in (customers_path, transactions_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run python src/data_generator.py first.")
    customers = pd.read_csv(customers_path, parse_dates=["acquisition_date"])
    transactions = pd.read_csv(transactions_path, parse_dates=["transaction_date"])
    return customers, transactions


# --------------------------------------------------------------------------
# Feature preparation
# --------------------------------------------------------------------------
def build_rfm(customers: pd.DataFrame, transactions: pd.DataFrame, cutoff) -> pd.DataFrame:
    """Build the RFM summary using only orders up to and including `cutoff`.

    Columns:
      frequency       number of REPEAT orders (orders minus one)
      recency_weeks   weeks between the first and the last order
      T_weeks         weeks between the first order and the cutoff
      avg_order_value mean order value in the period
      n_orders, revenue  totals in the period (historical CLV = revenue)
    Customers acquired after the cutoff are excluded.
    """
    cutoff = pd.Timestamp(cutoff)
    cust = customers[customers["acquisition_date"] <= cutoff]
    tx = transactions[
        (transactions["transaction_date"] <= cutoff)
        & (transactions["customer_id"].isin(cust["customer_id"]))
    ]
    grouped = tx.groupby("customer_id").agg(
        n_orders=("order_value", "size"),
        revenue=("order_value", "sum"),
        first_order=("transaction_date", "min"),
        last_order=("transaction_date", "max"),
    )
    rfm = cust.merge(grouped, on="customer_id", how="inner")
    rfm["frequency"] = rfm["n_orders"] - 1
    rfm["recency_weeks"] = (rfm["last_order"] - rfm["first_order"]).dt.days / 7.0
    rfm["T_weeks"] = (cutoff - rfm["first_order"]).dt.days / 7.0
    rfm["avg_order_value"] = rfm["revenue"] / rfm["n_orders"]
    return rfm.drop(columns=["first_order", "last_order"])


def shrunk_avg_order_value(rfm: pd.DataFrame, k: float = AOV_SHRINKAGE_ORDERS) -> pd.Series:
    """Average order value pulled toward the global mean.

    Customers with few orders have noisy averages; the weight on the customer's own
    mean is n / (n + k). This is a simple heuristic, not a fitted model.
    """
    global_mean = rfm["revenue"].sum() / rfm["n_orders"].sum()
    n = rfm["n_orders"]
    return (n * rfm["avg_order_value"] + k * global_mean) / (n + k)


def holdout_actuals(
    customers: pd.DataFrame, transactions: pd.DataFrame, cutoff, end
) -> pd.DataFrame:
    """Actual orders and revenue per customer in the period (cutoff, end]."""
    cutoff, end = pd.Timestamp(cutoff), pd.Timestamp(end)
    window = transactions[
        (transactions["transaction_date"] > cutoff) & (transactions["transaction_date"] <= end)
    ]
    agg = window.groupby("customer_id").agg(
        actual_orders=("order_value", "size"), actual_revenue=("order_value", "sum")
    )
    ids = customers["customer_id"]
    return agg.reindex(ids, fill_value=0).rename_axis("customer_id").reset_index()


# --------------------------------------------------------------------------
# BG/NBD model
# --------------------------------------------------------------------------
def bgnbd_log_likelihood(params, x, t_x, T) -> np.ndarray:
    """Per-customer BG/NBD log-likelihood. params = (r, alpha, a, b)."""
    r, alpha, a, b = params
    x = np.asarray(x, dtype=float)
    t_x = np.asarray(t_x, dtype=float)
    T = np.asarray(T, dtype=float)

    x_safe = np.maximum(x, 1.0)  # avoids invalid Beta arguments in the x = 0 branch
    a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    a2 = betaln(a, b + x) - betaln(a, b) - (r + x) * np.log(alpha + T)
    a3 = betaln(a + 1, b + x_safe - 1) - betaln(a, b) - (r + x) * np.log(alpha + t_x)
    a3 = np.where(x > 0, a3, -np.inf)
    return a1 + np.logaddexp(a2, a3)


def _unpack(theta) -> tuple[float, float, float, float]:
    """Map unconstrained parameters to (r, alpha, a, b)."""
    r, alpha, b = np.exp(theta[0]), np.exp(theta[1]), np.exp(theta[3])
    a = A_FLOOR + np.exp(theta[2])
    return r, alpha, a, b


def fit_bgnbd(rfm: pd.DataFrame) -> dict:
    """Fit BG/NBD by maximum likelihood with scipy (L-BFGS-B on transformed parameters)."""
    x = rfm["frequency"].to_numpy(float)
    t_x = rfm["recency_weeks"].to_numpy(float)
    T = rfm["T_weeks"].to_numpy(float)

    def objective(theta):
        value = -np.sum(bgnbd_log_likelihood(_unpack(theta), x, t_x, T))
        return value if np.isfinite(value) else 1e12

    result = minimize(
        objective,
        x0=np.array([0.0, 2.0, 0.0, 1.0]),
        method="L-BFGS-B",
        bounds=[(-6, 6)] * 4,
    )
    r, alpha, a, b = _unpack(result.x)
    return {
        "r": float(r),
        "alpha": float(alpha),
        "a": float(a),
        "b": float(b),
        "log_likelihood": float(-result.fun),
        "converged": bool(result.success),
        "n_customers": int(len(rfm)),
    }


def _param_tuple(params: dict) -> tuple[float, float, float, float]:
    return params["r"], params["alpha"], params["a"], params["b"]


def prob_alive(params: dict, x, t_x, T) -> np.ndarray:
    """Probability that a customer is still active at the end of the calibration period."""
    r, alpha, a, b = _param_tuple(params)
    x = np.asarray(x, dtype=float)
    t_x = np.asarray(t_x, dtype=float)
    T = np.asarray(T, dtype=float)
    x_safe = np.maximum(x, 1.0)
    ratio = a / (b + x_safe - 1) * ((alpha + T) / (alpha + t_x)) ** (r + x)
    return 1.0 / (1.0 + np.where(x > 0, ratio, 0.0))


def expected_purchases(params: dict, t_weeks: float, x, t_x, T) -> np.ndarray:
    """Expected number of purchases in the next `t_weeks` weeks, given each customer's history."""
    r, alpha, a, b = _param_tuple(params)
    x = np.asarray(x, dtype=float)
    t_x = np.asarray(t_x, dtype=float)
    T = np.asarray(T, dtype=float)

    z = t_weeks / (alpha + T + t_weeks)
    hyp = hyp2f1(r + x, b + x, a + b + x - 1, z)
    numerator = (a + b + x - 1) / (a - 1) * (
        1.0 - ((alpha + T) / (alpha + T + t_weeks)) ** (r + x) * hyp
    )
    x_safe = np.maximum(x, 1.0)
    extra = a / (b + x_safe - 1) * ((alpha + T) / (alpha + t_x)) ** (r + x)
    denominator = 1.0 + np.where(x > 0, extra, 0.0)
    return numerator / denominator


def naive_expected_purchases(t_weeks: float, x, T) -> np.ndarray:
    """Baseline: observed repeat-purchase rate extrapolated forward (ignores churn)."""
    x = np.asarray(x, dtype=float)
    T = np.asarray(T, dtype=float)
    return np.divide(x, T, out=np.zeros_like(x), where=T > 0) * t_weeks


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE, R^2 and aggregate totals for customer-level predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    sst = np.sum((y_true - y_true.mean()) ** 2)
    total_true = float(y_true.sum())
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1.0 - np.sum(err**2) / sst) if sst > 0 else float("nan"),
        "total_actual": total_true,
        "total_predicted": float(y_pred.sum()),
        "total_error_pct": float(100.0 * (y_pred.sum() - total_true) / total_true)
        if total_true > 0
        else float("nan"),
    }


def rank_correlation(y_true, y_pred) -> float:
    """Spearman rank correlation between actual and predicted values."""
    value = spearmanr(y_true, y_pred).correlation
    return float(value)


def top_share_captured(y_true, y_score, frac: float = 0.10) -> float:
    """Share of total actual value held by the top `frac` of customers ranked by `y_score`."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n_top = int(np.ceil(frac * len(y_true)))
    order = np.argsort(-y_score, kind="stable")[:n_top]
    total = y_true.sum()
    return float(y_true[order].sum() / total) if total > 0 else float("nan")


def segment_summary(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Average historical and predicted CLV per segment."""
    return (
        df.groupby(by)
        .agg(
            customers=("customer_id", "count"),
            avg_historical_revenue=("revenue", "mean"),
            avg_p_alive=("p_alive", "mean"),
            avg_predicted_12m_revenue=("predicted_12m_revenue", "mean"),
        )
        .round(2)
    )
