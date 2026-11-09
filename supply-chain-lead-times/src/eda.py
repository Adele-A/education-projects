"""Exploratory data analysis for the synthetic supply chain lead time data.

Loads suppliers.csv and purchase_orders.csv from data/, prints data quality
checks, descriptive statistics and several aggregations, and saves charts
to figures/. All data is SYNTHETIC.

Run from the project root:
    python src/eda.py
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, no display needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
FIG_DIR = PROJECT_DIR / "figures"

DATE_COLS = ["order_date", "promised_date", "delivery_date"]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both CSV tables and parse the date columns.

    Returns:
        Tuple (suppliers, orders).
    """
    suppliers = pd.read_csv(DATA_DIR / "suppliers.csv")
    orders = pd.read_csv(DATA_DIR / "purchase_orders.csv", parse_dates=DATE_COLS)
    return suppliers, orders


def describe_table(name: str, df: pd.DataFrame) -> None:
    """Print shape, dtypes and missing values of a table."""
    print(f"\n=== {name} ===")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nData types:")
    print(df.dtypes.to_string())
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    print(f"\nDuplicated rows: {df.duplicated().sum()}")


def integrity_checks(suppliers: pd.DataFrame, orders: pd.DataFrame) -> None:
    """Print simple key and date logic checks."""
    print("\n=== Integrity checks ===")
    print(f"Unique supplier_id: {suppliers['supplier_id'].is_unique}")
    print(f"Unique po_id: {orders['po_id'].is_unique}")
    orphans = (~orders["supplier_id"].isin(suppliers["supplier_id"])).sum()
    print(f"Orders with unknown supplier_id: {orphans}")
    calc_delay = (orders["delivery_date"] - orders["promised_date"]).dt.days
    print(f"delay_days consistent with dates: {(calc_delay == orders['delay_days']).all()}")
    print(f"on_time consistent with delay_days: "
          f"{((orders['delay_days'] <= 0).astype(int) == orders['on_time']).all()}")
    print(f"Suppliers without any orders: "
          f"{(~suppliers['supplier_id'].isin(orders['supplier_id'])).sum()}")


def print_statistics(orders: pd.DataFrame) -> None:
    """Print descriptive statistics of key numeric order columns."""
    cols = ["promised_lead_time_days", "actual_lead_time_days", "delay_days",
            "order_qty", "unit_price", "order_value"]
    print("\n=== Descriptive statistics (purchase orders) ===")
    print(orders[cols].describe().round(2).to_string())
    print(f"\nOverall on-time rate: {orders['on_time'].mean():.3f}")
    print(f"Expedited share: {orders['is_expedited'].mean():.3f}")


def print_aggregations(df: pd.DataFrame) -> None:
    """Print on-time rate and lead time aggregations by segment."""
    def summarize(by: str) -> pd.DataFrame:
        g = df.groupby(by).agg(
            orders=("po_id", "count"),
            on_time_rate=("on_time", "mean"),
            mean_delay=("delay_days", "mean"),
            mean_lead_time=("actual_lead_time_days", "mean"),
            std_lead_time=("actual_lead_time_days", "std"),
        )
        return g.round(3)

    for by in ["reliability_tier", "transport_mode", "region", "category"]:
        print(f"\n=== Aggregation by {by} ===")
        print(summarize(by).to_string())

    df = df.assign(quarter=df["order_date"].dt.quarter)
    print("\n=== Aggregation by order quarter (all years) ===")
    print(summarize("quarter").to_string())

    print("\n=== Aggregation by expedited flag ===")
    print(summarize("is_expedited").to_string())

    sup = df.groupby("supplier_id").agg(
        orders=("po_id", "count"),
        on_time_rate=("on_time", "mean"),
    )
    print("\n=== Five lowest on-time rates (suppliers with >= 30 orders) ===")
    print(sup[sup["orders"] >= 30].nsmallest(5, "on_time_rate").round(3).to_string())


def save_charts(df: pd.DataFrame) -> None:
    """Save EDA charts as PNG files in figures/."""
    os.makedirs(FIG_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Distribution of delay days
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["delay_days"], bins=60, ax=ax, color="steelblue")
    ax.axvline(0, color="red", linestyle="--", label="Promised date")
    ax.set_title("Distribution of delivery delay (synthetic data)")
    ax.set_xlabel("Delay (days, negative = early)")
    ax.set_ylabel("Number of orders")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "delay_distribution.png", dpi=120)
    plt.close(fig)

    # 2. On-time rate by reliability tier and transport mode
    fig, ax = plt.subplots(figsize=(8, 5))
    rates = (df.groupby(["transport_mode", "reliability_tier"])["on_time"]
             .mean().reset_index())
    sns.barplot(data=rates, x="transport_mode", y="on_time",
                hue="reliability_tier", hue_order=["A", "B", "C"], ax=ax)
    ax.set_title("On-time rate by transport mode and reliability tier")
    ax.set_xlabel("Transport mode")
    ax.set_ylabel("On-time rate")
    ax.set_ylim(0, 1)
    ax.legend(title="Tier")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "on_time_by_mode_tier.png", dpi=120)
    plt.close(fig)

    # 3. Monthly on-time rate trend
    monthly = (df.set_index("order_date")["on_time"].resample("MS").mean()
               .reset_index())
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=monthly, x="order_date", y="on_time", marker="o", ax=ax)
    ax.set_title("Monthly on-time rate by order date")
    ax.set_xlabel("Order month")
    ax.set_ylabel("On-time rate")
    ax.set_ylim(0, 1)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "monthly_on_time_rate.png", dpi=120)
    plt.close(fig)

    # 4. Lead time variability by region
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="region", y="actual_lead_time_days", ax=ax)
    ax.set_title("Actual lead time by region")
    ax.set_xlabel("Region")
    ax.set_ylabel("Actual lead time (days)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lead_time_by_region.png", dpi=120)
    plt.close(fig)

    print(f"\nSaved 4 charts to {FIG_DIR}")


def main() -> None:
    """Run the full EDA workflow."""
    suppliers, orders = load_data()
    describe_table("suppliers", suppliers)
    describe_table("purchase_orders", orders)
    integrity_checks(suppliers, orders)
    print_statistics(orders)

    merged = orders.merge(suppliers, on="supplier_id", how="left")
    print_aggregations(merged)
    save_charts(merged)


if __name__ == "__main__":
    main()
