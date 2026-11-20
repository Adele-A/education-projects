"""Exploratory data analysis of the SYNTHETIC warehouse order picking data.

Loads data/pickers.csv and data/orders.csv (created by src/data_generator.py),
prints a data quality overview, descriptive statistics and a few aggregations,
and saves charts to figures/.

Run from the project root:  python src/eda.py
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, no display needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Paths are computed relative to the project root, not the working directory
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "figures"

NUMERIC_COLS = [
    "n_lines",
    "n_units",
    "total_weight_kg",
    "n_zones",
    "distance_m",
    "pick_time_min",
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the pickers and orders tables from data/."""
    pickers = pd.read_csv(DATA_DIR / "pickers.csv")
    orders = pd.read_csv(DATA_DIR / "orders.csv", parse_dates=["order_date"])
    return pickers, orders


def print_overview(name: str, df: pd.DataFrame) -> None:
    """Print shape, dtypes and missing values of a table."""
    print(f"\n=== {name} ===")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nData types:")
    print(df.dtypes.to_string())
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")


def print_statistics(orders: pd.DataFrame) -> None:
    """Print descriptive statistics for the numeric order columns."""
    print("\n=== Descriptive statistics (orders) ===")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(orders[NUMERIC_COLS].describe().round(2).to_string())
    print("\nSkewness of numeric columns:")
    print(orders[NUMERIC_COLS].skew().round(2).to_string())


def print_aggregations(orders: pd.DataFrame, pickers: pd.DataFrame) -> pd.DataFrame:
    """Print several aggregations of picking time; return the merged table."""
    merged = orders.merge(pickers, on="picker_id", how="left")
    print(f"\nRows without a matching picker: {merged['contract_type'].isna().sum()}")

    # 1. By shift
    print("\n=== Picking time by shift ===")
    print(
        orders.groupby("shift")["pick_time_min"]
        .agg(["count", "mean", "median"])
        .round(2)
        .to_string()
    )

    # 2. By priority
    print("\n=== Picking time by priority ===")
    print(
        orders.groupby("priority")["pick_time_min"]
        .agg(["count", "mean", "median"])
        .round(2)
        .to_string()
    )

    # 3. By contract type
    print("\n=== Picking time by contract type ===")
    print(
        merged.groupby("contract_type")["pick_time_min"]
        .agg(["count", "mean", "median"])
        .round(2)
        .to_string()
    )

    # 4. By number of zones
    print("\n=== Picking time and size by number of zones ===")
    print(
        orders.groupby("n_zones")
        .agg(
            orders=("order_id", "count"),
            mean_lines=("n_lines", "mean"),
            mean_time=("pick_time_min", "mean"),
        )
        .round(2)
        .to_string()
    )

    # 5. By experience band
    merged["experience_band"] = pd.cut(
        merged["experience_months"],
        bins=[0, 12, 36, 72, 120],
        labels=["1-12", "13-36", "37-72", "73-120"],
    )
    print("\n=== Picking time by picker experience (months) ===")
    print(
        merged.groupby("experience_band", observed=True)["pick_time_min"]
        .agg(["count", "mean", "median"])
        .round(2)
        .to_string()
    )

    # Correlations with the target
    print("\n=== Correlation of numeric columns with pick_time_min ===")
    print(
        orders[NUMERIC_COLS]
        .corr()["pick_time_min"]
        .drop("pick_time_min")
        .round(3)
        .to_string()
    )
    return merged


def save_charts(orders: pd.DataFrame, merged: pd.DataFrame) -> None:
    """Save charts to figures/."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Chart 1: distribution of picking time
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(orders["pick_time_min"], bins=50, kde=True, ax=ax)
    ax.set_title("Distribution of picking time (synthetic data)")
    ax.set_xlabel("Picking time (min)")
    ax.set_ylabel("Number of orders")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pick_time_distribution.png", dpi=120)
    plt.close(fig)

    # Chart 2: picking time vs number of lines, colored by number of zones
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=orders,
        x="n_lines",
        y="pick_time_min",
        hue="n_zones",
        palette="viridis",
        alpha=0.5,
        s=18,
        ax=ax,
    )
    ax.set_title("Picking time vs number of lines (synthetic data)")
    ax.set_xlabel("Number of order lines")
    ax.set_ylabel("Picking time (min)")
    ax.legend(title="Zones", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pick_time_vs_lines.png", dpi=120)
    plt.close(fig)

    # Chart 3: picking time by shift and contract type
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=merged,
        x="shift",
        y="pick_time_min",
        hue="contract_type",
        order=["morning", "afternoon", "night"],
        ax=ax,
    )
    ax.set_title("Picking time by shift and contract type (synthetic data)")
    ax.set_xlabel("Shift")
    ax.set_ylabel("Picking time (min)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pick_time_by_shift_contract.png", dpi=120)
    plt.close(fig)

    # Chart 4: correlation heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        orders[NUMERIC_COLS].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Correlation matrix of numeric order columns")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=120)
    plt.close(fig)

    print(f"\nSaved 4 charts to {FIGURES_DIR}")


def main() -> None:
    """Run the full EDA."""
    pickers, orders = load_data()
    print_overview("pickers", pickers)
    print_overview("orders", orders)
    print_statistics(orders)
    merged = print_aggregations(orders, pickers)
    save_charts(orders, merged)


if __name__ == "__main__":
    main()
