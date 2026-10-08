"""Reusable logic for the subscription revenue analysis.

Contains data loading, the monthly MRR bridge, retention metrics, segment
comparison and cohort revenue helpers. Kept free of printing and plotting so
that the functions can be unit tested.

All data is SYNTHETIC, produced by src/data_generator.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

EVENT_ORDER = ["new", "expansion", "contraction", "churn"]
MRR_COLUMNS = {
    "new": "new_mrr",
    "expansion": "expansion_mrr",
    "contraction": "contraction_mrr",
    "churn": "churned_mrr",
}


def load_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load customers and subscription events, parsing date columns."""
    customers_path = Path(data_dir) / "customers.csv"
    events_path = Path(data_dir) / "subscription_events.csv"
    for path in (customers_path, events_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run 'python src/data_generator.py' first."
            )
    customers = pd.read_csv(customers_path, parse_dates=["signup_date"])
    events = pd.read_csv(events_path, parse_dates=["event_date"])
    return customers, events


def month_range(events: pd.DataFrame) -> pd.PeriodIndex:
    """Return every calendar month between the first and last event."""
    months = events["event_date"].dt.to_period("M")
    return pd.period_range(months.min(), months.max(), freq="M")


def monthly_mrr_bridge(events: pd.DataFrame, months: pd.PeriodIndex | None = None) -> pd.DataFrame:
    """Build the monthly MRR bridge: opening + movements = closing.

    Churned and contraction MRR are negative numbers (as in `mrr_change`).
    Customer counts (active at start, new, churned, active at end) are included.
    Pass `months` to force a common month index (useful for segments).
    """
    if months is None:
        months = month_range(events)
    ev = events.assign(month=events["event_date"].dt.to_period("M"))

    amounts = ev.pivot_table(
        index="month", columns="event_type", values="mrr_change",
        aggfunc="sum", fill_value=0.0,
    ).reindex(columns=EVENT_ORDER, fill_value=0.0).reindex(months, fill_value=0.0)
    counts = ev.pivot_table(
        index="month", columns="event_type", values="event_id",
        aggfunc="count", fill_value=0,
    ).reindex(columns=EVENT_ORDER, fill_value=0).reindex(months, fill_value=0)

    bridge = amounts.rename(columns=MRR_COLUMNS)
    net = bridge[list(MRR_COLUMNS.values())].sum(axis=1)
    bridge["closing_mrr"] = net.cumsum()
    bridge.insert(0, "opening_mrr", bridge["closing_mrr"].shift(1, fill_value=0.0))

    bridge["new_customers"] = counts["new"]
    bridge["churned_customers"] = counts["churn"]
    bridge["active_end"] = (counts["new"] - counts["churn"]).cumsum()
    bridge["active_start"] = bridge["active_end"].shift(1, fill_value=0)
    return bridge


def add_monthly_rates(bridge: pd.DataFrame) -> pd.DataFrame:
    """Add monthly churn and retention rates (NaN when the opening base is 0)."""
    out = bridge.copy()
    opening = out["opening_mrr"].replace(0, np.nan)
    active = out["active_start"].replace(0, np.nan)
    out["customer_churn_rate"] = out["churned_customers"] / active
    out["revenue_churn_rate"] = -out["churned_mrr"] / opening
    out["grr"] = (out["opening_mrr"] + out["contraction_mrr"] + out["churned_mrr"]) / opening
    out["nrr"] = (
        out["opening_mrr"] + out["expansion_mrr"] + out["contraction_mrr"] + out["churned_mrr"]
    ) / opening
    return out


def summarize_retention(bridge: pd.DataFrame) -> dict:
    """Opening-MRR-weighted monthly retention metrics over the whole period.

    Months with zero opening MRR (the first month) are excluded. NRR and GRR
    are monthly figures: movements divided by the opening MRR.
    """
    valid = bridge[bridge["opening_mrr"] > 0]
    nan = float("nan")
    if valid.empty:
        return {"avg_monthly_nrr": nan, "avg_monthly_grr": nan,
                "avg_monthly_customer_churn_rate": nan,
                "avg_monthly_revenue_churn_rate": nan, "months_used": 0}
    opening = valid["opening_mrr"].sum()
    exp = valid["expansion_mrr"].sum()
    con = valid["contraction_mrr"].sum()
    churn = valid["churned_mrr"].sum()
    active = valid["active_start"].sum()
    return {
        "avg_monthly_nrr": float((opening + exp + con + churn) / opening),
        "avg_monthly_grr": float((opening + con + churn) / opening),
        "avg_monthly_customer_churn_rate": float(valid["churned_customers"].sum() / active),
        "avg_monthly_revenue_churn_rate": float(-churn / opening),
        "months_used": int(len(valid)),
    }


def customer_mrr_matrix(events: pd.DataFrame, months: pd.PeriodIndex | None = None) -> pd.DataFrame:
    """Customer x month matrix with each customer's MRR at the end of the month."""
    if months is None:
        months = month_range(events)
    ev = events.assign(month=events["event_date"].dt.to_period("M"))
    changes = ev.pivot_table(
        index="customer_id", columns="month", values="mrr_change",
        aggfunc="sum", fill_value=0.0,
    ).reindex(columns=months, fill_value=0.0)
    return changes.cumsum(axis=1)


def bridge_reconciles(bridge: pd.DataFrame, matrix: pd.DataFrame, tol: float = 0.01) -> bool:
    """Check the bridge against an independent calculation.

    1. opening + movements must equal closing in every month,
    2. closing MRR must equal the column sums of the customer MRR matrix.
    """
    movements = bridge[list(MRR_COLUMNS.values())].sum(axis=1)
    inner = (bridge["opening_mrr"] + movements - bridge["closing_mrr"]).abs().max() <= tol
    outer = (bridge["closing_mrr"].to_numpy() - matrix.sum(axis=0).to_numpy())
    return bool(inner and np.abs(outer).max() <= tol)


def segment_retention(events: pd.DataFrame, customers: pd.DataFrame, column: str) -> pd.DataFrame:
    """Compare retention metrics across the values of a customer attribute."""
    months = month_range(events)
    merged = events.merge(customers[["customer_id", column]], on="customer_id", how="left")
    rows = {}
    for value, part in merged.groupby(column):
        bridge = monthly_mrr_bridge(part, months)
        summary = summarize_retention(bridge)
        summary["customers"] = int((part["event_type"] == "new").sum())
        summary["final_mrr"] = float(bridge["closing_mrr"].iloc[-1])
        rows[value] = summary
    return pd.DataFrame(rows).T


def cohort_revenue(customers: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    """Cohort MRR by months since signup (rows: signup month, columns: 0, 1, 2, ...).

    Values are the cohort's month-end MRR. Offsets not yet observed are NaN.
    """
    months = matrix.columns
    cohort = customers.set_index("customer_id")["signup_date"].dt.to_period("M")
    cohort = cohort.reindex(matrix.index)
    by_calendar = matrix.groupby(cohort).sum()

    out = pd.DataFrame(np.nan, index=by_calendar.index, columns=range(len(months)))
    for cohort_month in by_calendar.index:
        pos = months.get_loc(cohort_month)
        values = by_calendar.loc[cohort_month].to_numpy()[pos:]
        out.loc[cohort_month, : len(values) - 1] = values
    return out


def cohort_retention_pct(cohort_mrr: pd.DataFrame) -> pd.DataFrame:
    """Cohort MRR as a percentage of the cohort's MRR in its signup month."""
    base = cohort_mrr[0].replace(0, np.nan)
    return cohort_mrr.div(base, axis=0) * 100


def pooled_cohort_retention(cohort_mrr: pd.DataFrame, offset: int) -> float:
    """Pooled MRR retention (%) at a given month offset.

    Uses only cohorts that have reached the offset: sum of their MRR at the
    offset divided by sum of their MRR in the signup month.
    """
    if offset not in cohort_mrr.columns:
        return float("nan")
    reached = cohort_mrr[offset].notna()
    base = cohort_mrr.loc[reached, 0].sum()
    if not reached.any() or base == 0:
        return float("nan")
    return float(cohort_mrr.loc[reached, offset].sum() / base * 100)
