"""Exploratory data analysis of the SYNTHETIC pharmacy sales data.

Reads data/categories.csv and data/weekly_sales.csv, prints data quality
checks, descriptive statistics and several aggregations, and saves charts
to figures/.

Run from the project root (after python src/data_generator.py):
    python src/eda.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, no display needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
FIG_DIR = PROJECT_DIR / "figures"

sns.set_theme(style="whitegrid")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both CSV tables; parse the date column."""
    categories = pd.read_csv(DATA_DIR / "categories.csv")
    sales = pd.read_csv(DATA_DIR / "weekly_sales.csv", parse_dates=["week_start"])
    return categories, sales


def print_overview(name: str, df: pd.DataFrame) -> None:
    """Print shape, dtypes and missing values of a table."""
    print(f"\n=== {name} ===")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nDtypes:")
    print(df.dtypes.to_string())
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")


def data_quality_checks(sales: pd.DataFrame) -> None:
    """Basic integrity checks on the weekly sales table."""
    print("\n=== Data quality checks ===")
    dup_keys = sales.duplicated(["week_start", "category"]).sum()
    print(f"Duplicate (week_start, category) keys: {dup_keys}")
    print(f"Date range: {sales['week_start'].min().date()} to {sales['week_start'].max().date()}")
    print(f"Weeks per category:\n{sales.groupby('category')['week_start'].nunique().to_string()}")
    print(f"Negative units: {(sales['units_sold'] < 0).sum()}")
    print(f"Non-positive prices: {(sales['unit_price'] <= 0).sum()}")
    print(f"Non-Monday week starts: {(sales['week_start'].dt.dayofweek != 0).sum()}")


def descriptive_stats(sales: pd.DataFrame) -> None:
    """Descriptive statistics of the numeric columns."""
    print("\n=== Descriptive statistics ===")
    cols = ["units_sold", "unit_price", "promo_flag", "revenue", "avg_temp_c"]
    print(sales[cols].describe().round(2).T.to_string())


def aggregations(sales: pd.DataFrame, categories: pd.DataFrame) -> dict:
    """Print the key aggregations and return the tables used for charts."""
    sales = sales.copy()
    sales["year"] = sales["week_start"].dt.year
    sales["month"] = sales["week_start"].dt.month

    # 1. Totals and averages per category
    by_cat = (
        sales.groupby("category")
        .agg(
            avg_weekly_units=("units_sold", "mean"),
            total_units=("units_sold", "sum"),
            total_revenue=("revenue", "sum"),
            promo_share=("promo_flag", "mean"),
        )
        .sort_values("total_revenue", ascending=False)
    )
    print("\n=== 1. Totals per category ===")
    print(by_cat.round(2).to_string())

    # 2. Yearly units per category (trend)
    yearly = sales.pivot_table(index="year", columns="category", values="units_sold", aggfunc="sum")
    print("\n=== 2. Yearly units per category ===")
    print(yearly.to_string())

    # 3. Seasonal profile: average units per month relative to the category mean
    monthly = sales.groupby(["category", "month"])["units_sold"].mean().unstack("month")
    seasonal_index = monthly.div(monthly.mean(axis=1), axis=0)
    print("\n=== 3. Seasonal index by month (1.00 = category average) ===")
    print(seasonal_index.round(2).to_string())

    # 4. Seasonality strength and peak month per category
    strength = pd.DataFrame(
        {
            "peak_month": seasonal_index.idxmax(axis=1),
            "trough_month": seasonal_index.idxmin(axis=1),
            "peak_to_trough": seasonal_index.max(axis=1) / seasonal_index.min(axis=1),
        }
    ).sort_values("peak_to_trough", ascending=False)
    strength = strength.join(categories.set_index("category")["seasonality_pattern"])
    print("\n=== 4. Seasonality strength (peak / trough of monthly index) ===")
    print(strength.round(2).to_string())

    # 5. Promotion effect: average units in promo vs non-promo weeks
    promo = sales.pivot_table(index="category", columns="promo_flag", values="units_sold", aggfunc="mean")
    promo.columns = ["no_promo", "promo"]
    promo["ratio"] = promo["promo"] / promo["no_promo"]
    print("\n=== 5. Average weekly units, promo vs non-promo weeks ===")
    print(promo.round(2).to_string())
    print("(Raw ratio; it is mixed with seasonality because promos are not timed evenly.)")

    # Correlation of units with temperature per category
    corr = sales.groupby("category").apply(
        lambda g: g["units_sold"].corr(g["avg_temp_c"]), include_groups=False
    )
    print("\nCorrelation of weekly units with temperature:")
    print(corr.round(2).to_string())

    return {"seasonal_index": seasonal_index, "yearly": yearly}


def plot_weekly_series(sales: pd.DataFrame) -> None:
    """Weekly units per category, one panel each."""
    g = sns.FacetGrid(sales, col="category", col_wrap=4, height=2.4, aspect=1.5, sharey=False)
    g.map_dataframe(sns.lineplot, x="week_start", y="units_sold", linewidth=0.8)
    g.set_titles("{col_name}")
    g.set_axis_labels("Week", "Units sold")
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=45)
    g.figure.suptitle("Weekly units sold by category (synthetic data)", y=1.02)
    g.savefig(FIG_DIR / "weekly_units_by_category.png", dpi=120, bbox_inches="tight")
    plt.close(g.figure)


def plot_seasonal_heatmap(seasonal_index: pd.DataFrame) -> None:
    """Heatmap of the monthly seasonal index per category."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.heatmap(seasonal_index, annot=True, fmt=".2f", cmap="coolwarm", center=1.0,
                cbar_kws={"label": "Index (1.00 = category average)"}, ax=ax)
    ax.set_title("Seasonal index by month and category (synthetic data)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Category")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "seasonal_index_heatmap.png", dpi=120)
    plt.close(fig)


def plot_promo_and_temperature(sales: pd.DataFrame) -> None:
    """Promo vs non-promo units and units vs temperature for selected categories."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Units normalised by category mean so categories are comparable
    norm = sales.copy()
    norm["units_rel"] = norm["units_sold"] / norm.groupby("category")["units_sold"].transform("mean")
    sns.boxplot(data=norm, x="promo_flag", y="units_rel", ax=axes[0])
    axes[0].set_title("Relative units: promo vs non-promo weeks")
    axes[0].set_xlabel("Promotion flag")
    axes[0].set_ylabel("Units / category mean")

    subset = norm[norm["category"].isin(["cold_and_flu", "sun_care", "allergy"])]
    sns.scatterplot(data=subset, x="avg_temp_c", y="units_rel", hue="category", s=14, ax=axes[1])
    axes[1].set_title("Relative units vs temperature")
    axes[1].set_xlabel("Average temperature (deg C)")
    axes[1].set_ylabel("Units / category mean")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "promo_and_temperature.png", dpi=120)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    categories, sales = load_data()

    print_overview("categories", categories)
    print_overview("weekly_sales", sales)
    data_quality_checks(sales)
    descriptive_stats(sales)
    tables = aggregations(sales, categories)

    plot_weekly_series(sales)
    plot_seasonal_heatmap(tables["seasonal_index"])
    plot_promo_and_temperature(sales)
    print(f"\nSaved 3 charts to {FIG_DIR.relative_to(PROJECT_DIR)}/")


if __name__ == "__main__":
    main()
