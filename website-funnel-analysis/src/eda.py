"""Exploratory data analysis for the Website Funnel Analysis project.

Loads the SYNTHETIC tables from data/, prints data quality checks,
descriptive statistics and several aggregations, and saves charts to figures/.

Run from the project root:  python src/eda.py
(Run python src/data_generator.py first to create the CSV files.)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: no windows, files only
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
FIG_DIR = PROJECT_DIR / "figures"

# Ordered funnel steps (the first step, session start, is every row)
FUNNEL_STEPS = ["viewed_product", "added_to_cart", "reached_checkout", "purchased"]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the sessions and traffic source tables."""
    sessions_path = DATA_DIR / "sessions.csv"
    sources_path = DATA_DIR / "traffic_sources.csv"
    if not sessions_path.exists() or not sources_path.exists():
        raise FileNotFoundError(
            "CSV files not found in data/. Run: python src/data_generator.py"
        )
    sessions = pd.read_csv(sessions_path, parse_dates=["session_date"])
    sources = pd.read_csv(sources_path)
    return sessions, sources


def print_overview(name: str, df: pd.DataFrame) -> None:
    """Print shape, dtypes and missing values for one table."""
    print(f"\n=== {name}: overview ===")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nData types:")
    print(df.dtypes.to_string())
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")


def print_descriptives(sessions: pd.DataFrame) -> None:
    """Print descriptive statistics for numeric and categorical columns."""
    print("\n=== sessions: numeric descriptive statistics ===")
    num_cols = ["pages_viewed", "session_duration_sec", "order_value"]
    print(sessions[num_cols].describe().round(2).to_string())

    purchases = sessions.loc[sessions["purchased"] == 1, "order_value"]
    print("\nOrder value among purchasing sessions only:")
    print(purchases.describe().round(2).to_string())

    print("\n=== sessions: categorical distributions (share of sessions) ===")
    for col in ["traffic_source", "device", "country", "user_type"]:
        print(f"\n{col}:")
        print(sessions[col].value_counts(normalize=True).round(3).to_string())


def funnel_table(sessions: pd.DataFrame) -> pd.DataFrame:
    """Overall funnel: counts, step conversion and conversion from session start."""
    counts = [len(sessions)] + [int(sessions[s].sum()) for s in FUNNEL_STEPS]
    steps = ["session"] + FUNNEL_STEPS
    table = pd.DataFrame({"step": steps, "sessions": counts})
    table["step_conversion"] = table["sessions"] / table["sessions"].shift(1)
    table["overall_conversion"] = table["sessions"] / table["sessions"].iloc[0]
    table["drop_off"] = 1 - table["step_conversion"]
    return table


def segment_funnel(sessions: pd.DataFrame, by: str) -> pd.DataFrame:
    """Funnel rates, revenue per session and average order value by segment."""
    grouped = sessions.groupby(by)
    out = grouped.agg(
        sessions=("session_id", "count"),
        view_rate=("viewed_product", "mean"),
        cart_rate=("added_to_cart", "mean"),
        checkout_rate=("reached_checkout", "mean"),
        purchase_rate=("purchased", "mean"),
        revenue=("order_value", "sum"),
    )
    orders = grouped["purchased"].sum()
    out["revenue_per_session"] = out["revenue"] / out["sessions"]
    out["avg_order_value"] = out["revenue"] / orders.where(orders > 0)
    return out.sort_values("purchase_rate", ascending=False)


def source_cost_table(sessions: pd.DataFrame, sources: pd.DataFrame) -> pd.DataFrame:
    """Join cost data and compute cost per order and revenue minus cost per source."""
    seg = segment_funnel(sessions, "traffic_source").reset_index()
    merged = seg.merge(sources, on="traffic_source", how="left")
    merged["total_cost"] = merged["sessions"] * merged["cost_per_session"]
    orders = merged["sessions"] * merged["purchase_rate"]
    merged["cost_per_order"] = merged["total_cost"] / orders.where(orders > 0)
    merged["net_revenue"] = merged["revenue"] - merged["total_cost"]
    return merged


def plot_funnel(table: pd.DataFrame) -> None:
    """Bar chart of overall funnel counts with drop-off annotations."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=table, x="step", y="sessions", color="steelblue", ax=ax)
    for i, row in table.iterrows():
        label = f"{int(row['sessions'])}"
        if i > 0:
            label += f"\n({row['step_conversion']:.1%} of previous)"
        ax.text(i, row["sessions"], label, ha="center", va="bottom", fontsize=9)
    ax.set_title("Overall funnel (synthetic data)")
    ax.set_xlabel("Funnel step")
    ax.set_ylabel("Number of sessions")
    ax.set_ylim(0, table["sessions"].max() * 1.15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "funnel_overall.png", dpi=120)
    plt.close(fig)


def plot_source_conversion(source_table: pd.DataFrame) -> None:
    """Purchase rate by traffic source."""
    data = source_table.sort_values("purchase_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=data, x="traffic_source", y="purchase_rate",
                color="seagreen", ax=ax)
    ax.set_title("Purchase rate by traffic source (synthetic data)")
    ax.set_xlabel("Traffic source")
    ax.set_ylabel("Purchase rate (purchases / sessions)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "purchase_rate_by_source.png", dpi=120)
    plt.close(fig)


def plot_device_funnel(sessions: pd.DataFrame) -> None:
    """Step rates by device, shown as grouped bars."""
    seg = segment_funnel(sessions, "device").reset_index()
    long = seg.melt(
        id_vars="device",
        value_vars=["view_rate", "cart_rate", "checkout_rate", "purchase_rate"],
        var_name="step",
        value_name="rate",
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=long, x="step", y="rate", hue="device", ax=ax)
    ax.set_title("Share of sessions reaching each step, by device (synthetic data)")
    ax.set_xlabel("Funnel step")
    ax.set_ylabel("Share of all sessions")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "funnel_by_device.png", dpi=120)
    plt.close(fig)


def plot_order_value(sessions: pd.DataFrame) -> None:
    """Distribution of order value for purchasing sessions."""
    orders = sessions.loc[sessions["purchased"] == 1, "order_value"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(orders, bins=40, color="darkorange", ax=ax)
    ax.set_title("Order value distribution, purchasing sessions (synthetic data)")
    ax.set_xlabel("Order value (USD)")
    ax.set_ylabel("Number of orders")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "order_value_distribution.png", dpi=120)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    sns.set_theme(style="whitegrid")

    sessions, sources = load_data()

    print_overview("sessions", sessions)
    print_overview("traffic_sources", sources)

    # Basic integrity checks relevant to the funnel
    nested = (
        (sessions["added_to_cart"] <= sessions["viewed_product"]).all()
        and (sessions["reached_checkout"] <= sessions["added_to_cart"]).all()
        and (sessions["purchased"] <= sessions["reached_checkout"]).all()
    )
    print(f"\nFunnel steps nested correctly: {nested}")
    print(f"Session dates: {sessions['session_date'].min().date()} "
          f"to {sessions['session_date'].max().date()}")

    print_descriptives(sessions)

    # Aggregation 1: overall funnel
    funnel = funnel_table(sessions)
    print("\n=== Aggregation 1: overall funnel ===")
    print(funnel.round(3).to_string(index=False))

    # Aggregation 2: traffic source performance incl. cost
    source_table = source_cost_table(sessions, sources)
    print("\n=== Aggregation 2: traffic source performance and cost ===")
    cols = ["traffic_source", "channel_type", "sessions", "view_rate", "cart_rate",
            "checkout_rate", "purchase_rate", "revenue_per_session",
            "avg_order_value", "total_cost", "cost_per_order", "net_revenue"]
    print(source_table[cols].round(3).to_string(index=False))

    # Aggregation 3: device and user type
    for col in ["device", "user_type"]:
        print(f"\n=== Aggregation 3: funnel by {col} ===")
        print(segment_funnel(sessions, col).round(3).to_string())

    # Aggregation 4: monthly trend
    monthly = segment_funnel(
        sessions.assign(month=sessions["session_date"].dt.to_period("M").astype(str)),
        "month",
    ).sort_index()
    print("\n=== Aggregation 4: funnel by month ===")
    print(monthly.round(3).to_string())

    # Charts
    plot_funnel(funnel)
    plot_source_conversion(source_table)
    plot_device_funnel(sessions)
    plot_order_value(sessions)
    print(f"\nSaved charts to {FIG_DIR}")


if __name__ == "__main__":
    main()
