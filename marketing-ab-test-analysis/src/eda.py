"""Exploratory data analysis for the SYNTHETIC landing page A/B test.

Loads data/users.csv and data/orders.csv, prints data quality checks,
descriptive statistics and a few aggregations, and saves charts to figures/.

Run from the project root:
    python src/eda.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, no display needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_DIR / "figures"

# load_data and section are shared with analysis.py through utils
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_data, section  # noqa: E402


def overview(name: str, df: pd.DataFrame) -> None:
    """Print shape, dtypes, missing values and duplicate count."""
    section(f"Overview: {name}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nDtypes:")
    print(df.dtypes.to_string())
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")


def descriptive_stats(users: pd.DataFrame, orders: pd.DataFrame) -> None:
    section("Descriptive statistics")
    print("Users (numeric columns):")
    num_cols = ["is_returning", "page_views", "time_on_page_sec", "converted"]
    print(users[num_cols].describe().round(3).to_string())
    print("\nOrders (numeric columns):")
    print(orders[["items", "order_value"]].describe().round(3).to_string())


def aggregations(users: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Print key aggregations; return users merged with order value."""
    section("Aggregation 1: group balance (sample ratio)")
    balance = users["group"].value_counts().rename("visitors").to_frame()
    balance["share"] = (balance["visitors"] / len(users)).round(4)
    print(balance.to_string())

    section("Aggregation 2: conversion rate by group")
    by_group = users.groupby("group")["converted"].agg(
        visitors="count", buyers="sum", conversion_rate="mean"
    )
    by_group["conversion_rate"] = by_group["conversion_rate"].round(4)
    print(by_group.to_string())

    # Revenue per visitor: visitors without an order count as 0
    merged = users.merge(
        orders[["user_id", "order_value", "items"]], on="user_id", how="left"
    )
    merged["revenue"] = merged["order_value"].fillna(0.0)

    section("Aggregation 3: revenue metrics by group")
    revenue = merged.groupby("group").agg(
        total_revenue=("revenue", "sum"),
        revenue_per_visitor=("revenue", "mean"),
        avg_order_value=("order_value", "mean"),
    )
    print(revenue.round(2).to_string())

    section("Aggregation 4: covariate balance across groups (row shares)")
    for col in ["device", "traffic_source", "country"]:
        shares = pd.crosstab(users[col], users["group"], normalize="columns")
        print(f"\n{col} (share within group):")
        print(shares.round(3).to_string())
    print("\nMean of numeric covariates by group:")
    print(
        users.groupby("group")[["is_returning", "page_views", "time_on_page_sec"]]
        .mean()
        .round(3)
        .to_string()
    )

    section("Aggregation 5: conversion rate by segment and group")
    for col in ["device", "traffic_source"]:
        seg = users.pivot_table(
            index=col, columns="group", values="converted", aggfunc="mean"
        )
        print(f"\n{col}:")
        print(seg.round(4).to_string())

    return merged


def integrity_checks(users: pd.DataFrame, orders: pd.DataFrame) -> None:
    section("Integrity checks")
    print(f"user_id unique: {users['user_id'].is_unique}")
    print(f"order_id unique: {orders['order_id'].is_unique}")
    buyers = set(users.loc[users["converted"] == 1, "user_id"])
    order_users = set(orders["user_id"])
    print(f"Converted users match users with orders: {buyers == order_users}")


def make_charts(users: pd.DataFrame, merged: pd.DataFrame) -> None:
    """Save a few PNG charts to figures/."""
    FIG_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Conversion rate by group
    rates = users.groupby("group")["converted"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(data=rates, x="group", y="converted", ax=ax)
    ax.set_title("Conversion rate by group (synthetic data)")
    ax.set_xlabel("Group")
    ax.set_ylabel("Conversion rate")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "conversion_by_group.png", dpi=120)
    plt.close(fig)

    # 2. Conversion rate by traffic source and group
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(
        data=users, x="traffic_source", y="converted", hue="group",
        errorbar=None, ax=ax,
    )
    ax.set_title("Conversion rate by traffic source and group (synthetic data)")
    ax.set_xlabel("Traffic source")
    ax.set_ylabel("Conversion rate")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "conversion_by_source.png", dpi=120)
    plt.close(fig)

    # 3. Order value distribution by group (buyers only)
    buyers = merged.dropna(subset=["order_value"])
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(
        data=buyers, x="order_value", hue="group", bins=30,
        element="step", stat="density", common_norm=False, ax=ax,
    )
    ax.set_title("Order value distribution by group (synthetic data)")
    ax.set_xlabel("Order value (USD-like)")
    ax.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "order_value_distribution.png", dpi=120)
    plt.close(fig)

    # 4. Daily visitors by group
    daily = (
        users.groupby(["visit_date", "group"]).size().reset_index(name="visitors")
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=daily, x="visit_date", y="visitors", hue="group",
                 marker="o", ax=ax)
    ax.set_title("Daily visitors by group (synthetic data)")
    ax.set_xlabel("Visit date")
    ax.set_ylabel("Visitors")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "daily_visitors.png", dpi=120)
    plt.close(fig)

    print(f"\nSaved 4 charts to {FIG_DIR}")


def main() -> None:
    users, orders = load_data()
    overview("users", users)
    overview("orders", orders)
    integrity_checks(users, orders)
    descriptive_stats(users, orders)
    merged = aggregations(users, orders)
    make_charts(users, merged)


if __name__ == "__main__":
    main()
