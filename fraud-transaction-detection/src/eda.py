"""Exploratory data analysis for the SYNTHETIC fraud transaction data.

Loads data/customers.csv and data/transactions.csv (created by
src/data_generator.py), prints an overview and a few aggregations, and
saves charts to figures/. Run from the project root:

    python src/eda.py

Output:
    figures/class_balance_and_amount.png
    figures/fraud_by_hour.png
    figures/fraud_rate_by_segment.png
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


def load_data():
    """Load both tables; fail with a clear message if data is missing."""
    cust_path = DATA_DIR / "customers.csv"
    tx_path = DATA_DIR / "transactions.csv"
    if not cust_path.exists() or not tx_path.exists():
        raise SystemExit("Data files not found. Run: python src/data_generator.py")
    customers = pd.read_csv(cust_path)
    transactions = pd.read_csv(tx_path, parse_dates=["timestamp"])
    return customers, transactions


def overview(name, df):
    """Print shape, dtypes, missing values and descriptive statistics."""
    print("=" * 70)
    print("TABLE: %s" % name)
    print("=" * 70)
    print("Shape: %d rows, %d columns" % df.shape)
    print("\nData types:")
    print(df.dtypes.to_string())
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    print("\nDescriptive statistics (numeric columns):")
    print(df.describe().round(2).T.to_string())
    print()


def print_aggregations(tx):
    """Print meaningful aggregations of the transactions table."""
    tx = tx.copy()
    tx["hour"] = tx["timestamp"].dt.hour
    tx["is_night"] = tx["hour"].isin([0, 1, 2, 3, 4, 5, 22, 23]).astype(int)

    print("=" * 70)
    print("AGGREGATIONS")
    print("=" * 70)

    # 1. Class balance
    balance = tx["is_fraud"].value_counts().sort_index()
    print("\n1) Class balance (is_fraud):")
    print(pd.DataFrame({
        "count": balance,
        "share": (balance / len(tx)).round(4),
    }).to_string())

    # 2. Amount statistics by label
    print("\n2) Amount by fraud label:")
    print(tx.groupby("is_fraud")["amount"]
          .agg(["count", "mean", "median", "max"]).round(2).to_string())

    # 3. Fraud rate by channel
    print("\n3) Fraud rate by channel:")
    print(tx.groupby("channel")["is_fraud"]
          .agg(transactions="count", frauds="sum", fraud_rate="mean")
          .round(4).sort_values("fraud_rate", ascending=False).to_string())

    # 4. Fraud rate by merchant category
    print("\n4) Fraud rate by merchant category:")
    print(tx.groupby("merchant_category")["is_fraud"]
          .agg(transactions="count", frauds="sum", fraud_rate="mean")
          .round(4).sort_values("fraud_rate", ascending=False).to_string())

    # 5. Night vs day, foreign vs domestic
    print("\n5) Fraud rate by night-time flag and foreign flag:")
    print(tx.groupby(["is_night", "is_foreign"])["is_fraud"]
          .agg(transactions="count", frauds="sum", fraud_rate="mean")
          .round(4).to_string())
    print()


def plot_class_balance_and_amount(tx):
    """Chart 1: class balance and amount distribution by label."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    sns.countplot(data=tx, x="is_fraud", ax=axes[0], color="steelblue")
    axes[0].set_title("Class balance (synthetic data)")
    axes[0].set_xlabel("is_fraud (0 = legitimate, 1 = fraud)")
    axes[0].set_ylabel("Number of transactions")

    # Log scale because amounts are right-skewed
    sns.boxplot(data=tx, x="is_fraud", y="amount", ax=axes[1], color="lightsteelblue")
    axes[1].set_yscale("log")
    axes[1].set_title("Transaction amount by label (log scale)")
    axes[1].set_xlabel("is_fraud (0 = legitimate, 1 = fraud)")
    axes[1].set_ylabel("Amount")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_balance_and_amount.png", dpi=120)
    plt.close(fig)


def plot_fraud_by_hour(tx):
    """Chart 2: hourly distribution of each class (normalised per class)."""
    hours = tx.assign(hour=tx["timestamp"].dt.hour)
    share = (hours.groupby(["is_fraud", "hour"]).size()
             .groupby(level=0).transform(lambda s: s / s.sum())
             .reset_index(name="share"))
    share["label"] = share["is_fraud"].map({0: "legitimate", 1: "fraud"})

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.lineplot(data=share, x="hour", y="share", hue="label", marker="o", ax=ax)
    ax.set_title("Share of transactions by hour of day (within each class)")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Share of class transactions")
    ax.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fraud_by_hour.png", dpi=120)
    plt.close(fig)


def plot_fraud_rate_by_segment(tx):
    """Chart 3: fraud rate by channel and by merchant category."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    by_channel = tx.groupby("channel")["is_fraud"].mean().sort_values(ascending=False)
    sns.barplot(x=by_channel.index, y=by_channel.values, ax=axes[0], color="indianred")
    axes[0].set_title("Fraud rate by channel")
    axes[0].set_xlabel("Channel")
    axes[0].set_ylabel("Fraud rate")

    by_cat = tx.groupby("merchant_category")["is_fraud"].mean().sort_values(ascending=False)
    sns.barplot(x=by_cat.values, y=by_cat.index, ax=axes[1], color="indianred")
    axes[1].set_title("Fraud rate by merchant category")
    axes[1].set_xlabel("Fraud rate")
    axes[1].set_ylabel("Merchant category")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fraud_rate_by_segment.png", dpi=120)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    customers, transactions = load_data()

    overview("customers", customers)
    overview("transactions", transactions)
    print_aggregations(transactions)

    plot_class_balance_and_amount(transactions)
    plot_fraud_by_hour(transactions)
    plot_fraud_rate_by_segment(transactions)
    print("Charts saved to:", FIG_DIR)


if __name__ == "__main__":
    main()
