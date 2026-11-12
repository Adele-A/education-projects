"""Exploratory data analysis for the SYNTHETIC CLV dataset.

Loads data/customers.csv and data/transactions.csv, prints an overview
(shape, dtypes, missing values, descriptive statistics), prints several
aggregations and saves a few charts to figures/.

Run from the project root:
    python src/eda.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: figures are only saved to files
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

# Project paths (resolved relative to the project root)
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "figures"

# End of the observation window (same as in the data generator)
OBSERVATION_END = pd.Timestamp("2025-12-31")

sns.set_theme(style="whitegrid")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both CSV files and parse date columns."""
    customers_path = DATA_DIR / "customers.csv"
    transactions_path = DATA_DIR / "transactions.csv"
    for path in (customers_path, transactions_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run python src/data_generator.py first.")

    customers = pd.read_csv(customers_path, parse_dates=["acquisition_date"])
    transactions = pd.read_csv(transactions_path, parse_dates=["transaction_date"])
    return customers, transactions


def print_overview(name: str, df: pd.DataFrame) -> None:
    """Print shape, dtypes, missing values and descriptive statistics of a table."""
    print(f"\n=== Table: {name} ===")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nData types:")
    print(df.dtypes.to_string())
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    print("\nDescriptive statistics:")
    print(df.describe(include="all").T.to_string())


def build_customer_summary(customers: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """One row per customer: number of orders, revenue, average order value, observed days."""
    per_customer = transactions.groupby("customer_id").agg(
        n_orders=("transaction_id", "count"),
        revenue=("order_value", "sum"),
    )
    summary = customers.merge(per_customer, on="customer_id", how="left")
    summary[["n_orders", "revenue"]] = summary[["n_orders", "revenue"]].fillna(0)
    summary["avg_order_value"] = summary["revenue"] / summary["n_orders"].replace(0, np.nan)
    summary["observed_days"] = (OBSERVATION_END - summary["acquisition_date"]).dt.days
    summary["cohort_quarter"] = summary["acquisition_date"].dt.to_period("Q").astype(str)
    return summary


def segment_table(summary: pd.DataFrame, by: str) -> pd.DataFrame:
    """Aggregate customer-level metrics by a segment column."""
    return (
        summary.groupby(by)
        .agg(
            customers=("customer_id", "count"),
            avg_orders=("n_orders", "mean"),
            avg_revenue=("revenue", "mean"),
            avg_order_value=("avg_order_value", "mean"),
            avg_observed_days=("observed_days", "mean"),
        )
        .round(2)
    )


def print_aggregations(summary: pd.DataFrame, transactions: pd.DataFrame) -> None:
    """Print the key aggregations."""
    print("\n=== Aggregation 1: distribution of orders per customer ===")
    print(summary["n_orders"].describe().round(2).to_string())
    one_order_share = (summary["n_orders"] == 1).mean()
    print(f"\nShare of customers with exactly one order: {one_order_share:.1%}")

    print("\n=== Aggregation 2: revenue concentration ===")
    sorted_rev = summary["revenue"].sort_values(ascending=False)
    top_n = int(np.ceil(0.10 * len(sorted_rev)))
    top_share = sorted_rev.head(top_n).sum() / sorted_rev.sum()
    print(f"Top 10% of customers ({top_n} customers) generate {top_share:.1%} of total revenue")
    print(f"Total revenue: {sorted_rev.sum():,.2f}")

    print("\n=== Aggregation 3: segments by acquisition channel ===")
    print(segment_table(summary, "acquisition_channel").to_string())

    print("\n=== Aggregation 4: segments by country ===")
    print(segment_table(summary, "country").to_string())

    print("\n=== Aggregation 5: acquisition cohorts (by quarter) ===")
    print("Note: later cohorts have fewer observed days, so totals are not directly comparable.")
    print(segment_table(summary, "cohort_quarter").to_string())

    print("\n=== Inter-purchase times (days between consecutive orders) ===")
    ordered = transactions.sort_values(["customer_id", "transaction_date"])
    gaps = ordered.groupby("customer_id")["transaction_date"].diff().dt.days.dropna()
    if len(gaps) > 0:
        print(gaps.describe().round(2).to_string())
    else:
        print("No repeat purchases found.")


def plot_orders_per_customer(summary: pd.DataFrame, cap: int = 10) -> None:
    """Bar chart of orders per customer; values above `cap` are grouped."""
    capped = summary["n_orders"].clip(upper=cap).astype(int)
    counts = capped.value_counts().reindex(range(1, cap + 1), fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=counts.index.astype(str), y=counts.values, color="steelblue", ax=ax)
    labels = [str(i) for i in range(1, cap)] + [f"{cap}+"]
    ax.set_xticks(range(cap))
    ax.set_xticklabels(labels)
    ax.set_title("Orders per customer (synthetic data)")
    ax.set_xlabel("Number of orders")
    ax.set_ylabel("Number of customers")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "orders_per_customer.png", dpi=120)
    plt.close(fig)


def plot_order_value_distribution(transactions: pd.DataFrame) -> None:
    """Histogram of order values."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(transactions["order_value"], bins=40, color="darkorange", ax=ax)
    ax.set_title("Distribution of order values (synthetic data)")
    ax.set_xlabel("Order value")
    ax.set_ylabel("Number of orders")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "order_value_distribution.png", dpi=120)
    plt.close(fig)


def plot_revenue_by_channel(summary: pd.DataFrame) -> None:
    """Average revenue per customer by acquisition channel (with 95% CI)."""
    order = (
        summary.groupby("acquisition_channel")["revenue"].mean().sort_values(ascending=False).index
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=summary, x="acquisition_channel", y="revenue", order=order,
        color="seagreen", errorbar=("ci", 95), ax=ax,
    )
    ax.set_title("Average revenue per customer by acquisition channel (synthetic data)")
    ax.set_xlabel("Acquisition channel")
    ax.set_ylabel("Average revenue per customer")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "revenue_by_channel.png", dpi=120)
    plt.close(fig)


def plot_monthly_revenue(transactions: pd.DataFrame) -> None:
    """Line chart of total revenue per calendar month."""
    monthly = transactions.set_index("transaction_date")["order_value"].resample("MS").sum()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(monthly.index, monthly.values, marker="o", color="purple")
    ax.set_title("Monthly revenue (synthetic data)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total revenue")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "monthly_revenue.png", dpi=120)
    plt.close(fig)


def main() -> None:
    customers, transactions = load_data()

    print_overview("customers", customers)
    print_overview("transactions", transactions)

    summary = build_customer_summary(customers, transactions)
    print_aggregations(summary, transactions)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_orders_per_customer(summary)
    plot_order_value_distribution(transactions)
    plot_revenue_by_channel(summary)
    plot_monthly_revenue(transactions)
    print(f"\nSaved 4 charts to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
