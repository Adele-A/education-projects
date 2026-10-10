"""Exploratory data analysis for the synthetic weekly retail sales data.

Loads data/stores.csv and data/weekly_sales.csv, prints data quality checks,
descriptive statistics and a few aggregations, and saves charts to figures/.

All data is synthetic. Run from the project root: python src/eda.py
(run python src/data_generator.py first to create the CSV files).
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: figures are only saved to files
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "figures"

sns.set_theme(style="whitegrid")
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read both CSV files and parse the week_start column as dates."""
    stores_path = DATA_DIR / "stores.csv"
    sales_path = DATA_DIR / "weekly_sales.csv"
    for path in (stores_path, sales_path):
        if not path.exists():
            sys.exit(f"Missing {path}. Run 'python src/data_generator.py' first.")
    stores = pd.read_csv(stores_path)
    sales = pd.read_csv(sales_path, parse_dates=["week_start"])
    return stores, sales


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n=== {title} ===")


def data_quality(stores: pd.DataFrame, sales: pd.DataFrame) -> None:
    """Print shape, dtypes, missing values, duplicates and key integrity checks."""
    for name, df in (("stores", stores), ("weekly_sales", sales)):
        section(f"{name}: shape and dtypes")
        print(f"Shape: {df.shape}")
        print(df.dtypes)
        print("\nMissing values per column:")
        print(df.isna().sum())

    section("Integrity checks")
    key = ["week_start", "store_id", "category"]
    print(f"Duplicate rows on (week, store, category): {sales.duplicated(key).sum()}")
    print(f"Store IDs in sales but not in stores: "
          f"{sorted(set(sales['store_id']) - set(stores['store_id']))}")
    print(f"Rows with units_sold < 0: {(sales['units_sold'] < 0).sum()}")
    print(f"Rows with units_sold == 0: {(sales['units_sold'] == 0).sum()}")
    weeks = sales["week_start"].drop_duplicates().sort_values()
    gaps = weeks.diff().dropna().value_counts()
    print(f"Distinct gaps between consecutive weeks: {gaps.to_dict()}")
    print(f"Period: {weeks.min().date()} to {weeks.max().date()} ({len(weeks)} weeks)")


def descriptive_stats(sales: pd.DataFrame) -> None:
    """Print descriptive statistics of the numeric columns and flag shares."""
    section("Descriptive statistics")
    print(sales[["units_sold", "unit_price", "revenue"]].describe().round(2))
    print(f"\nShare of promotion rows: {sales['promo_flag'].mean():.3f}")
    print(f"Share of holiday-week rows: {sales['holiday_week'].mean():.3f}")


def aggregations(stores: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    """Print the main aggregations and return sales with helper columns."""
    df = sales.merge(stores, on="store_id", how="left")
    df["year"] = df["week_start"].dt.year
    df["month"] = df["week_start"].dt.month
    # Units relative to the average of the same store and category, so that
    # categories and stores of different sizes can be compared on one scale.
    df["units_rel"] = df["units_sold"] / df.groupby(["store_id", "category"])["units_sold"].transform("mean")

    section("1. Totals by category")
    by_cat = df.groupby("category").agg(
        units=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        avg_price=("unit_price", "mean"),
    ).sort_values("revenue", ascending=False)
    print(by_cat.round(2))

    section("2. Revenue by store (with region and size)")
    by_store = df.groupby(["store_id", "region", "size_sqm"])["revenue"].sum().reset_index()
    by_store["revenue_per_sqm"] = by_store["revenue"] / by_store["size_sqm"]
    print(by_store.sort_values("revenue", ascending=False).round(2).to_string(index=False))

    section("3. Revenue by year and category")
    print(df.pivot_table(index="year", columns="category", values="revenue",
                         aggfunc="sum").round(0))

    section("4. Promotion effect (raw comparison, mean units per row)")
    promo = df.pivot_table(index="category", columns="promo_flag",
                           values="units_sold", aggfunc="mean")
    promo.columns = ["no_promo", "promo"]
    promo["raw_uplift_pct"] = (promo["promo"] / promo["no_promo"] - 1) * 100
    print(promo.round(2))
    print("Note: this ignores seasonality, so it is only a rough indication.")

    section("5. Holiday-week effect (units relative to store/category mean)")
    holiday = df.pivot_table(index="category", columns="holiday_week",
                             values="units_rel", aggfunc="mean")
    holiday.columns = ["regular_week", "holiday_week"]
    print(holiday.round(3))
    return df


def save_figure(fig, filename: str) -> None:
    """Apply a tight layout, save the figure to figures/ and close it."""
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=120)
    plt.close(fig)


def plot_weekly_units(df: pd.DataFrame) -> None:
    """Line chart of total weekly units per category (one panel each)."""
    weekly = df.groupby(["week_start", "category"])["units_sold"].sum().reset_index()
    categories = sorted(weekly["category"].unique())
    fig, axes = plt.subplots(len(categories), 1, figsize=(10, 8), sharex=True)
    for ax, cat in zip(axes, categories):
        part = weekly[weekly["category"] == cat]
        ax.plot(part["week_start"], part["units_sold"], linewidth=1.2)
        ax.set_ylabel("Units")
        ax.set_title(cat, loc="left", fontsize=10)
    axes[-1].set_xlabel("Week start")
    fig.suptitle("Total weekly units sold by category (all stores, synthetic data)")
    save_figure(fig, "weekly_units_by_category.png")


def plot_seasonality_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of the average relative units by month and category."""
    pivot = df.pivot_table(index="category", columns="month",
                           values="units_rel", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={"label": "Relative units"}, ax=ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("Category")
    ax.set_title("Seasonal profile: units relative to store/category mean")
    save_figure(fig, "seasonality_heatmap.png")


def plot_promo_holiday(df: pd.DataFrame) -> None:
    """Bar charts of relative units in promotion and holiday weeks."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    sns.barplot(data=df, x="category", y="units_rel", hue="promo_flag",
                errorbar=None, ax=axes[0])
    axes[0].set_title("Promotion vs regular weeks")
    sns.barplot(data=df, x="category", y="units_rel", hue="holiday_week",
                errorbar=None, ax=axes[1])
    axes[1].set_title("Holiday vs regular weeks")
    for ax in axes:
        ax.set_xlabel("Category")
        ax.set_ylabel("Mean relative units")
    fig.suptitle("Relative units (store/category mean = 1) by week type")
    save_figure(fig, "promo_holiday_effect.png")


def plot_store_revenue(df: pd.DataFrame) -> None:
    """Bar chart of total revenue per store, colored by region."""
    by_store = (df.groupby(["store_id", "region"])["revenue"].sum()
                .reset_index().sort_values("store_id"))
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=by_store, x="store_id", y="revenue", hue="region", dodge=False, ax=ax)
    ax.set_xlabel("Store")
    ax.set_ylabel("Total revenue")
    ax.set_title("Total revenue by store (synthetic data)")
    save_figure(fig, "revenue_by_store.png")


def main() -> None:
    """Run the full EDA: checks, statistics, aggregations and charts."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    stores, sales = load_data()

    data_quality(stores, sales)
    descriptive_stats(sales)
    df = aggregations(stores, sales)

    plot_weekly_units(df)
    plot_seasonality_heatmap(df)
    plot_promo_holiday(df)
    plot_store_revenue(df)

    section("Charts saved")
    for path in sorted(FIGURES_DIR.glob("*.png")):
        print(path.relative_to(PROJECT_DIR))


if __name__ == "__main__":
    main()
