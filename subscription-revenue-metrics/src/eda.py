"""Exploratory data analysis for the synthetic subscription dataset.

Reads (relative to the project root):
    data/customers.csv
    data/subscription_events.csv

Prints shape, dtypes, missing values, descriptive statistics, data quality
checks and several aggregations, and saves charts to figures/.

All data is SYNTHETIC, produced by src/data_generator.py.

Run from the project root:
    python src/eda.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, no display needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "figures"

EVENT_ORDER = ["new", "expansion", "contraction", "churn"]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both CSV files, parsing date columns."""
    customers_path = DATA_DIR / "customers.csv"
    events_path = DATA_DIR / "subscription_events.csv"
    for path in (customers_path, events_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run 'python src/data_generator.py' first."
            )
    customers = pd.read_csv(customers_path, parse_dates=["signup_date"])
    events = pd.read_csv(events_path, parse_dates=["event_date"])
    return customers, events


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def describe_table(name: str, df: pd.DataFrame) -> None:
    """Print shape, dtypes, missing values and descriptive statistics."""
    section(f"TABLE: {name}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nDtypes:")
    print(df.dtypes.to_string())
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    print("\nDescriptive statistics (numeric and date columns):")
    print(df.describe(include=["number", "datetime"]).T.to_string())


def quality_checks(customers: pd.DataFrame, events: pd.DataFrame) -> None:
    """Run basic integrity checks and print the outcome of each."""
    section("DATA QUALITY CHECKS")
    print(f"Duplicate customer_id rows: {customers['customer_id'].duplicated().sum()}")
    print(f"Duplicate event_id rows: {events['event_id'].duplicated().sum()}")

    orphans = ~events["customer_id"].isin(customers["customer_id"])
    print(f"Events with unknown customer_id: {orphans.sum()}")

    diff = (events["mrr_after"] - events["mrr_before"] - events["mrr_change"]).abs()
    print(f"Rows where mrr_change != mrr_after - mrr_before: {(diff > 0.01).sum()}")

    new_per_customer = events[events["event_type"] == "new"].groupby("customer_id").size()
    print(f"Customers without exactly one 'new' event: "
          f"{(new_per_customer != 1).sum() + (len(customers) - len(new_per_customer))}")

    # Sign of mrr_change must match the event type.
    positive = events["event_type"].isin(["new", "expansion"])
    bad_sign = (positive & (events["mrr_change"] <= 0)) | (~positive & (events["mrr_change"] >= 0))
    print(f"Events whose sign of mrr_change contradicts event_type: {bad_sign.sum()}")

    # New event date must equal the customer signup date.
    first = events[events["event_type"] == "new"].merge(
        customers[["customer_id", "signup_date"]], on="customer_id"
    )
    print(f"'new' events not on signup_date: {(first['event_date'] != first['signup_date']).sum()}")


def aggregations(customers: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Print the key aggregations; return the monthly MRR table for plotting."""
    section("AGGREGATION 1: events by type")
    by_type = (
        events.groupby("event_type")["mrr_change"]
        .agg(events="count", total_mrr_change="sum", avg_mrr_change="mean")
        .reindex(EVENT_ORDER)
        .round(2)
    )
    print(by_type.to_string())

    section("AGGREGATION 2: customer mix by plan, channel and company size")
    for col in ["initial_plan", "acquisition_channel", "company_size"]:
        share = customers[col].value_counts(normalize=True).mul(100).round(1)
        print(f"\n{col} (% of customers):")
        print(share.to_string())

    section("AGGREGATION 3: monthly new customers and MRR movements")
    monthly = events.assign(month=events["event_date"].dt.to_period("M"))
    movements = monthly.pivot_table(
        index="month", columns="event_type", values="mrr_change",
        aggfunc="sum", fill_value=0.0,
    ).reindex(columns=EVENT_ORDER, fill_value=0.0)
    all_months = pd.period_range(monthly["month"].min(), monthly["month"].max(), freq="M")
    movements = movements.reindex(all_months, fill_value=0.0)
    movements["net_change"] = movements[EVENT_ORDER].sum(axis=1)
    movements["mrr_end_of_month"] = movements["net_change"].cumsum()
    new_counts = (
        monthly[monthly["event_type"] == "new"].groupby("month").size()
        .reindex(all_months, fill_value=0)
    )
    movements.insert(0, "new_customers", new_counts)
    print(movements.round(2).to_string())

    section("AGGREGATION 4: churned share and average starting MRR by channel")
    churned_ids = set(events.loc[events["event_type"] == "churn", "customer_id"])
    start_mrr = (
        events[events["event_type"] == "new"].set_index("customer_id")["mrr_after"]
    )
    cust = customers.assign(
        churned=customers["customer_id"].isin(churned_ids),
        start_mrr=customers["customer_id"].map(start_mrr),
    )
    by_channel = cust.groupby("acquisition_channel").agg(
        customers=("customer_id", "count"),
        churned_share_pct=("churned", lambda s: s.mean() * 100),
        avg_start_mrr=("start_mrr", "mean"),
    ).round(2)
    print(by_channel.to_string())

    section("AGGREGATION 5: starting MRR by initial plan and company size")
    by_plan_size = cust.pivot_table(
        index="initial_plan", columns="company_size", values="start_mrr",
        aggfunc="mean",
    ).round(2)
    print(by_plan_size.to_string())

    # Keep the customer-level table for plotting.
    movements.attrs["customer_view"] = cust
    return movements


def make_charts(events: pd.DataFrame, movements: pd.DataFrame) -> None:
    """Save the EDA charts to figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    months = movements.index.to_timestamp()

    # Chart 1: total MRR at the end of each month.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(months, movements["mrr_end_of_month"], marker="o", color="tab:blue")
    ax.set_title("Total MRR at the end of each month (synthetic data)")
    ax.set_xlabel("Month")
    ax.set_ylabel("MRR, USD")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mrr_over_time.png", dpi=120)
    plt.close(fig)

    # Chart 2: new customers per month.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(months, movements["new_customers"], width=20, color="tab:green")
    ax.set_title("New customers per month (synthetic data)")
    ax.set_xlabel("Month")
    ax.set_ylabel("New customers")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "new_customers_per_month.png", dpi=120)
    plt.close(fig)

    # Chart 3: distribution of absolute MRR change by event type.
    plot_df = events.assign(abs_mrr_change=events["mrr_change"].abs())
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=plot_df, x="event_type", y="abs_mrr_change",
                order=EVENT_ORDER, ax=ax)
    ax.set_yscale("log")
    ax.set_title("Size of MRR change by event type (log scale, synthetic data)")
    ax.set_xlabel("Event type")
    ax.set_ylabel("Absolute MRR change, USD")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mrr_change_by_event_type.png", dpi=120)
    plt.close(fig)

    # Chart 4: share of churned customers by acquisition channel.
    cust = movements.attrs["customer_view"]
    churn_share = (
        cust.groupby("acquisition_channel")["churned"].mean().mul(100).sort_values()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=churn_share.index, y=churn_share.values, ax=ax, color="tab:red")
    ax.set_title("Share of customers who churned, by acquisition channel (synthetic data)")
    ax.set_xlabel("Acquisition channel")
    ax.set_ylabel("Churned customers, %")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "churn_share_by_channel.png", dpi=120)
    plt.close(fig)

    print(f"\nSaved 4 charts to {FIGURES_DIR}")


def main() -> None:
    customers, events = load_data()
    describe_table("customers.csv", customers)
    describe_table("subscription_events.csv", events)
    quality_checks(customers, events)
    movements = aggregations(customers, events)
    make_charts(events, movements)


if __name__ == "__main__":
    main()
