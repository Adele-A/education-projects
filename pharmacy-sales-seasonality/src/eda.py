"""Exploratory data analysis of the SYNTHETIC pharmacy sales data.

Reads data/categories.csv and data/weekly_sales.csv, prints data quality
checks, descriptive statistics and several aggregations, and saves charts
to figures/.

Run from the project root (after python src/data_generator.py):
    python src/eda.py
"""
import calendar
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
    # Replace underscores in category names for readable panel titles (display only)
    plot_df = sales.assign(category=sales["category"].str.replace("_", " ", regex=False))
    g = sns.FacetGrid(plot_df, col="category", col_wrap=4, height=2.6, aspect=1.5, sharey=False)
    g.map_dataframe(sns.lineplot, x="week_start", y="units_sold", linewidth=0.8)
    g.set_titles("{col_name}", size=10)
    g.set_axis_labels("Week start (Monday)", "Units sold per week")
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=45)
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    g.figure.suptitle("Weekly units sold by category, 2019-2023 (synthetic data)", y=1.02)
    g.savefig(FIG_DIR / "weekly_units_by_category.png", dpi=120, bbox_inches="tight")
    plt.close(g.figure)


def plot_seasonal_heatmap(seasonal_index: pd.DataFrame) -> None:
    """Heatmap of the monthly seasonal index per category."""
    # Month numbers become month names; category names lose underscores (display only)
    index_df = seasonal_index.rename(
        columns=lambda m: calendar.month_abbr[m],
        index=lambda s: s.replace("_", " "),
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(index_df, annot=True, fmt=".2f", cmap="coolwarm", center=1.0,
                linewidths=0.5, cbar_kws={"label": "Seasonal index (1.00 = category average)"}, ax=ax)
    ax.set_title("Seasonal index by month and medicine category (synthetic data)", pad=12)
    ax.set_xlabel("Month of year")
    ax.set_ylabel("Medicine category")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "seasonal_index_heatmap.png", dpi=120)
    plt.close(fig)


def plot_promo_and_temperature(sales: pd.DataFrame) -> None:
    """Promo vs non-promo units and units vs temperature for selected categories."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Units normalised by category mean so categories are comparable
    norm = sales.copy()
    norm["units_rel"] = norm["units_sold"] / norm.groupby("category")["units_sold"].transform("mean")
    sns.boxplot(data=norm, x="promo_flag", y="units_rel", ax=axes[0])
    axes[0].set_title("Weekly units in promo vs non-promo weeks")
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["No promotion", "Promotion"])
    axes[0].set_xlabel("Promotion status of the week")
    axes[0].set_ylabel("Units relative to category mean (1.00 = average)")

    subset = norm[norm["category"].isin(["cold_and_flu", "sun_care", "allergy"])].copy()
    subset["category"] = subset["category"].str.replace("_", " ", regex=False)
    sns.scatterplot(data=subset, x="avg_temp_c", y="units_rel", hue="category", s=16, alpha=0.7, ax=axes[1])
    axes[1].set_title("Weekly units vs average temperature")
    axes[1].set_xlabel("Average weekly temperature (deg C)")
    axes[1].set_ylabel("Units relative to category mean (1.00 = average)")
    axes[1].legend(title="Category", loc="upper left", bbox_to_anchor=(1.01, 1))

    fig.tight_layout()
    fig.savefig(FIG_DIR / "promo_and_temperature.png", dpi=120, bbox_inches="tight")
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
