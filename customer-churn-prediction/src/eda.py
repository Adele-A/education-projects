"""Exploratory data analysis for the Customer Churn Prediction project.

Loads data/customers.csv and data/support_tickets.csv (SYNTHETIC data created by
src/data_generator.py), prints data quality checks, descriptive statistics and
several aggregations, and saves charts to figures/.

Run from the project root:  python src/eda.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, no display needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "figures"

sns.set_theme(style="whitegrid")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both tables, failing with a clear message if data is missing."""
    cust_path = DATA_DIR / "customers.csv"
    tick_path = DATA_DIR / "support_tickets.csv"
    if not cust_path.exists() or not tick_path.exists():
        raise FileNotFoundError(
            "Data files not found. Run `python src/data_generator.py` first."
        )
    customers = pd.read_csv(cust_path, parse_dates=["signup_date"])
    tickets = pd.read_csv(tick_path, parse_dates=["created_date"])
    return customers, tickets


def section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def overview(name: str, df: pd.DataFrame) -> None:
    """Print shape, dtypes, missing values and duplicate count for a table."""
    section(f"{name}: overview")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nData types:")
    print(df.dtypes.to_string())
    print("\nMissing values per column:")
    missing = df.isna().sum()
    print(missing.to_string())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")


def descriptive_stats(customers: pd.DataFrame, tickets: pd.DataFrame) -> None:
    """Print descriptive statistics for numeric columns."""
    section("Descriptive statistics: customers (numeric)")
    num_cols = ["tenure_months", "age", "monthly_price", "avg_weekly_logins",
                "avg_session_minutes", "late_payments_6m"]
    print(customers[num_cols].describe().round(2).to_string())

    section("Descriptive statistics: support tickets (numeric)")
    print(tickets[["resolution_hours", "satisfaction_score"]].describe().round(2).to_string())

    section("Target: churn rate")
    print(f"Overall churn rate: {customers['churned'].mean():.3f}")
    print(customers["churned"].value_counts().rename("customers").to_string())


def churn_by(customers: pd.DataFrame, col: str) -> pd.DataFrame:
    """Customer count and churn rate by a categorical column."""
    return (customers.groupby(col)["churned"]
            .agg(customers="size", churn_rate="mean")
            .round(3)
            .sort_values("churn_rate", ascending=False))


def aggregations(customers: pd.DataFrame, tickets: pd.DataFrame) -> pd.DataFrame:
    """Print key aggregations; returns customers with ticket features added."""
    section("Aggregation 1: churn rate by contract type")
    print(churn_by(customers, "contract_type").to_string())

    section("Aggregation 2: churn rate by plan")
    print(churn_by(customers, "plan").to_string())

    section("Aggregation 3: churn rate by tenure band")
    bands = pd.cut(customers["tenure_months"], bins=[0, 6, 12, 24, 36, 60],
                   labels=["1-6", "7-12", "13-24", "25-36", "37-60"])
    tenure_tbl = (customers.groupby(bands, observed=True)["churned"]
                  .agg(customers="size", churn_rate="mean").round(3))
    print(tenure_tbl.to_string())

    section("Aggregation 4: churn rate by late payments (last 6 months)")
    late = customers["late_payments_6m"].clip(upper=3).map(
        lambda v: "3+" if v >= 3 else str(v))
    late_tbl = (customers.groupby(late)["churned"]
                .agg(customers="size", churn_rate="mean").round(3))
    print(late_tbl.to_string())

    section("Aggregation 5: ticket load vs. churn")
    ticket_counts = tickets.groupby("customer_id").size().rename("n_tickets")
    merged = customers.merge(ticket_counts, on="customer_id", how="left")
    merged["n_tickets"] = merged["n_tickets"].fillna(0).astype(int)
    merged["ticket_group"] = merged["n_tickets"].clip(upper=3).map(
        lambda v: "3+" if v >= 3 else str(v))
    ticket_tbl = (merged.groupby("ticket_group")["churned"]
                  .agg(customers="size", churn_rate="mean").round(3))
    print(ticket_tbl.to_string())
    return merged


def make_charts(customers: pd.DataFrame, merged: pd.DataFrame) -> None:
    """Save charts to figures/."""
    FIGURES_DIR.mkdir(exist_ok=True)

    # Chart 1: churn rate by contract type
    rates = customers.groupby("contract_type")["churned"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=rates.index, y=rates.values, ax=ax, color="steelblue")
    ax.set_title("Churn rate by contract type (synthetic data)")
    ax.set_xlabel("Contract type")
    ax.set_ylabel("Churn rate")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "churn_by_contract.png", dpi=120)
    plt.close(fig)

    # Chart 2: tenure distribution by churn status
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=customers, x="tenure_months", hue="churned", bins=30,
                 element="step", stat="density", common_norm=False, ax=ax)
    ax.set_title("Tenure distribution by churn status (synthetic data)")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "tenure_by_churn.png", dpi=120)
    plt.close(fig)

    # Chart 3: weekly logins by churn status
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=customers, x="churned", y="avg_weekly_logins", ax=ax)
    ax.set_title("Average weekly logins by churn status (synthetic data)")
    ax.set_xlabel("Churned (0 = no, 1 = yes)")
    ax.set_ylabel("Average weekly logins")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "logins_by_churn.png", dpi=120)
    plt.close(fig)

    # Chart 4: correlation heatmap of numeric features, including ticket count
    cols = ["tenure_months", "age", "monthly_price", "avg_weekly_logins",
            "avg_session_minutes", "late_payments_6m", "auto_renew",
            "discount_flag", "n_tickets", "churned"]
    corr = merged[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation matrix (synthetic data)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=120)
    plt.close(fig)

    print(f"\nSaved 4 charts to {FIGURES_DIR}")


def main() -> None:
    """Run the full EDA."""
    customers, tickets = load_data()
    overview("customers", customers)
    overview("support_tickets", tickets)
    descriptive_stats(customers, tickets)
    merged = aggregations(customers, tickets)
    make_charts(customers, merged)


if __name__ == "__main__":
    main()
