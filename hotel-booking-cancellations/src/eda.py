"""Exploratory data analysis for the SYNTHETIC hotel bookings dataset.

Loads data/guests.csv and data/bookings.csv, prints data quality checks,
descriptive statistics and several cancellation-rate aggregations, and saves
charts to figures/.

Run from the project root (after python src/data_generator.py):
    python src/eda.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: figures are only saved to disk
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
FIG_DIR = PROJECT_DIR / "figures"

# Lead time buckets (days between booking and arrival) used in aggregations.
LEAD_BINS = [-1, 7, 30, 90, 180, 450]
LEAD_LABELS = ["0-7", "8-30", "31-90", "91-180", "181-450"]

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read both CSV files, with a clear message if they are missing."""
    guests_path = DATA_DIR / "guests.csv"
    bookings_path = DATA_DIR / "bookings.csv"
    for path in (guests_path, bookings_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run 'python src/data_generator.py' first."
            )
    guests = pd.read_csv(guests_path)
    bookings = pd.read_csv(bookings_path, parse_dates=["arrival_date"])
    return guests, bookings


def section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def rate_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Booking count and cancellation rate for each value of `by`."""
    out = (
        df.groupby(by, observed=True)["is_canceled"]
        .agg(bookings="count", cancel_rate="mean")
        .round(3)
    )
    return out


def data_quality(name: str, df: pd.DataFrame) -> None:
    """Print shape, dtypes, missing values and duplicate count."""
    section(f"DATA QUALITY: {name}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nDtypes:")
    print(df.dtypes.to_string())
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")


def plot_lead_time(df: pd.DataFrame) -> None:
    """Lead time distribution by outcome and cancellation rate by lead bucket."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    sns.histplot(
        data=df.assign(outcome=df["is_canceled"].map({0: "Kept", 1: "Canceled"})),
        x="lead_time_days", hue="outcome", bins=40, stat="density",
        common_norm=False, element="step", ax=axes[0],
    )
    axes[0].set_title("Lead time distribution by outcome")
    axes[0].set_xlabel("Lead time (days)")
    axes[0].set_ylabel("Density")

    rates = rate_table(df, "lead_bucket")["cancel_rate"]
    sns.barplot(x=rates.index.astype(str), y=rates.values, color="steelblue", ax=axes[1])
    axes[1].set_title("Cancellation rate by lead time bucket")
    axes[1].set_xlabel("Lead time (days)")
    axes[1].set_ylabel("Cancellation rate")
    axes[1].set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "lead_time_vs_cancellation.png", dpi=120)
    plt.close(fig)


def plot_categorical_rates(df: pd.DataFrame) -> None:
    """Cancellation rate by deposit type and by market segment."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col, title in (
        (axes[0], "deposit_type", "Cancellation rate by deposit type"),
        (axes[1], "market_segment", "Cancellation rate by market segment"),
    ):
        rates = rate_table(df, col)["cancel_rate"].sort_values(ascending=False)
        sns.barplot(x=rates.index, y=rates.values, color="indianred", ax=ax)
        ax.set_title(title)
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel("Cancellation rate")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cancellation_by_deposit_and_segment.png", dpi=120)
    plt.close(fig)


def plot_monthly(df: pd.DataFrame) -> None:
    """Bookings volume and cancellation rate by arrival month."""
    monthly = df.groupby("arrival_month").agg(
        bookings=("is_canceled", "count"), cancel_rate=("is_canceled", "mean")
    )
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.bar(monthly.index, monthly["bookings"], color="lightgray", label="Bookings")
    ax1.set_xlabel("Arrival month")
    ax1.set_ylabel("Number of bookings")
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(MONTH_NAMES)

    ax2 = ax1.twinx()
    ax2.plot(monthly.index, monthly["cancel_rate"], color="darkred", marker="o",
             label="Cancellation rate")
    ax2.set_ylabel("Cancellation rate")
    ax2.set_ylim(0, 1)

    ax1.set_title("Bookings and cancellation rate by arrival month")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "monthly_bookings_and_cancellations.png", dpi=120)
    plt.close(fig)


def main() -> None:
    guests, bookings = load_data()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # --- Data quality ---------------------------------------------------
    data_quality("guests", guests)
    data_quality("bookings", bookings)

    # Referential integrity: every booking must point to a known guest.
    orphans = (~bookings["guest_id"].isin(guests["guest_id"])).sum()
    print(f"\nBookings with unknown guest_id: {orphans}")

    # --- Descriptive statistics -------------------------------------------
    section("DESCRIPTIVE STATISTICS: numeric columns of bookings")
    numeric_cols = [
        "lead_time_days", "stay_nights_weekend", "stay_nights_week", "adults",
        "children", "adr", "special_requests", "parking_spaces", "booking_changes",
    ]
    print(bookings[numeric_cols].describe().T.round(2).to_string())

    section("DESCRIPTIVE STATISTICS: guests")
    print(guests[["previous_stays", "previous_cancellations"]].describe().T.round(2).to_string())

    section("TARGET: is_canceled")
    print(bookings["is_canceled"].value_counts().sort_index().to_string())
    print(f"\nOverall cancellation rate: {bookings['is_canceled'].mean():.3f}")

    # --- Aggregations ---------------------------------------------------------
    # Join guest history onto bookings for the history-based aggregation.
    df = bookings.merge(guests, on="guest_id", how="left")
    df["lead_bucket"] = pd.cut(df["lead_time_days"], bins=LEAD_BINS, labels=LEAD_LABELS)
    df["total_nights"] = df["stay_nights_weekend"] + df["stay_nights_week"]
    df["guest_history"] = pd.cut(
        df["previous_cancellations"], bins=[-1, 0, 1, 8],
        labels=["0 previous", "1 previous", "2+ previous"],
    )

    section("AGGREGATION 1: cancellation rate by deposit type")
    print(rate_table(df, "deposit_type").to_string())

    section("AGGREGATION 2: cancellation rate by market segment")
    print(rate_table(df, "market_segment").sort_values("cancel_rate", ascending=False).to_string())

    section("AGGREGATION 3: cancellation rate by lead time bucket (days)")
    print(rate_table(df, "lead_bucket").to_string())

    section("AGGREGATION 4: cancellation rate by hotel type and special requests")
    print(
        df.pivot_table(index="special_requests", columns="hotel_type",
                       values="is_canceled", aggfunc="mean").round(3).to_string()
    )

    section("AGGREGATION 5: cancellation rate by number of earlier cancellations")
    print(rate_table(df, "guest_history").to_string())

    # --- Charts -----------------------------------------------------------------
    plot_lead_time(df)
    plot_categorical_rates(df)
    plot_monthly(df)

    section("FIGURES SAVED")
    for path in sorted(FIG_DIR.glob("*.png")):
        print(path.relative_to(PROJECT_DIR))


if __name__ == "__main__":
    main()
