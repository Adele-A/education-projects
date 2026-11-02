"""Exploratory data analysis for the price elasticity project.

Loads the synthetic tables from <project root>/data/, prints an overview
(shape, dtypes, missing values, descriptive statistics, a few aggregations)
and saves a small set of charts to <project root>/figures/.

Run from the project root:
    python src/eda.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, works without a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "figures"

sns.set_theme(style="whitegrid")
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the two CSV files produced by data_generator.py."""
    products_path = DATA_DIR / "products.csv"
    sales_path = DATA_DIR / "weekly_sales.csv"
    if not products_path.exists() or not sales_path.exists():
        sys.exit("Data files not found. Run `python src/data_generator.py` first.")
    products = pd.read_csv(products_path)
    sales = pd.read_csv(sales_path, parse_dates=["week_start"])
    return products, sales


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def overview(name: str, df: pd.DataFrame) -> None:
    """Print shape, dtypes, missing values and descriptive statistics."""
    section(f"Table: {name}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nData types:")
    print(df.dtypes.to_string())
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    print(f"\nDuplicated rows: {int(df.duplicated().sum())}")
    print("\nDescriptive statistics (numeric columns):")
    print(df.describe().T.round(2).to_string())


def build_analysis_frame(products: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    """Join sales with product attributes and add helper columns."""
    df = sales.merge(products, on="product_id", how="left", validate="many_to_one")
    if df["category"].isna().any():
        print("Warning: some sales rows have no matching product.")

    # Price relative to the product's regular price (1.0 = regular price).
    df["price_ratio"] = df["price"] / df["base_price"]
    df["log_price_ratio"] = np.log(df["price_ratio"])

    # Log units are undefined for zero-sales weeks; keep them as NaN.
    df["log_units"] = np.log(df["units_sold"].where(df["units_sold"] > 0))
    # Remove each product's own demand level so products are comparable.
    df["log_units_centered"] = df["log_units"] - df.groupby("product_id")["log_units"].transform("mean")
    return df


def print_aggregations(df: pd.DataFrame) -> None:
    """Print a handful of aggregations relevant for elasticity work."""
    section("Aggregation 1: sales summary by category")
    by_cat = df.groupby("category").agg(
        products=("product_id", "nunique"),
        avg_price=("price", "mean"),
        avg_units=("units_sold", "mean"),
        total_revenue=("revenue", "sum"),
    )
    by_cat["revenue_share_pct"] = 100 * by_cat["total_revenue"] / by_cat["total_revenue"].sum()
    print(by_cat.round(2).sort_values("total_revenue", ascending=False).to_string())

    section("Aggregation 2: promo vs. non-promo weeks by category")
    promo = (
        df.groupby(["category", "is_promo"])
        .agg(rows=("product_id", "size"), avg_price_ratio=("price_ratio", "mean"), avg_units=("units_sold", "mean"))
        .round(3)
        .unstack("is_promo")
    )
    print(promo.to_string())
    print(f"\nShare of promo rows overall: {100 * df['is_promo'].mean():.1f}%")

    section("Aggregation 3: holiday vs. regular weeks (average units per category)")
    holiday = df.pivot_table(index="category", columns="is_holiday_week", values="units_sold", aggfunc="mean")
    holiday["holiday_vs_regular_ratio"] = holiday[1] / holiday[0]
    print(holiday.round(2).to_string())

    section("Aggregation 4: brand tier summary")
    tier = df.groupby("brand_tier").agg(
        products=("product_id", "nunique"),
        avg_base_price=("base_price", "mean"),
        avg_units=("units_sold", "mean"),
    )
    print(tier.round(2).to_string())

    section("Aggregation 5: correlation of log price ratio and centered log units")
    # A raw, uncontrolled association - not an elasticity estimate.
    corr = (
        df.dropna(subset=["log_units_centered"])
        .groupby("category")
        .apply(lambda g: g["log_price_ratio"].corr(g["log_units_centered"]), include_groups=False)
        .rename("correlation")
    )
    print(corr.round(3).to_string())
    print(f"\nZero-sales rows (excluded from log columns): {int((df['units_sold'] == 0).sum())}")


def plot_log_log_by_category(df: pd.DataFrame) -> None:
    """Scatter of log price ratio vs. centered log units, one panel per category."""
    data = df.dropna(subset=["log_units_centered"])
    g = sns.lmplot(
        data=data,
        x="log_price_ratio",
        y="log_units_centered",
        col="category",
        col_wrap=3,
        height=3.2,
        scatter_kws={"alpha": 0.25, "s": 10},
        line_kws={"color": "crimson"},
    )
    g.set_axis_labels("log(price / base price)", "log(units) - product mean")
    g.set_titles("{col_name}")
    g.figure.suptitle("Log-log relationship between price and units", y=1.03)
    g.savefig(FIGURES_DIR / "log_log_by_category.png", dpi=120, bbox_inches="tight")
    plt.close(g.figure)


def plot_price_ratio_by_promo(df: pd.DataFrame) -> None:
    """Boxplot showing that promotions shift prices below the regular level."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="category", y="price_ratio", hue="is_promo", ax=ax)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    ax.set_title("Price relative to base price: promo vs. non-promo weeks")
    ax.set_xlabel("Category")
    ax.set_ylabel("Price / base price")
    ax.legend(title="is_promo")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "price_ratio_by_promo.png", dpi=120)
    plt.close(fig)


def plot_weekly_seasonality(df: pd.DataFrame) -> None:
    """Average weekly units per category, indexed to each category's own mean."""
    weekly = df.groupby(["week_of_year", "category"])["units_sold"].mean().unstack("category")
    weekly_index = weekly / weekly.mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    weekly_index.plot(ax=ax, linewidth=1.5)
    holiday_weeks = sorted(df.loc[df["is_holiday_week"] == 1, "week_of_year"].unique())
    for i, week in enumerate(holiday_weeks):
        ax.axvspan(week - 0.5, week + 0.5, color="orange", alpha=0.2, label="Holiday week" if i == 0 else None)
    ax.set_title("Seasonality: average weekly units (index, category mean = 1)")
    ax.set_xlabel("Week of year")
    ax.set_ylabel("Units index")
    ax.legend(title="Category", fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "weekly_seasonality.png", dpi=120)
    plt.close(fig)


def main() -> None:
    products, sales = load_data()

    overview("products", products)
    overview("weekly_sales", sales)

    df = build_analysis_frame(products, sales)
    print_aggregations(df)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_log_log_by_category(df)
    plot_price_ratio_by_promo(df)
    plot_weekly_seasonality(df)

    section("Charts saved")
    for path in sorted(FIGURES_DIR.glob("*.png")):
        print(path.relative_to(PROJECT_DIR))


if __name__ == "__main__":
    main()
