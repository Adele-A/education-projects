"""Exploratory data analysis for the Delivery Time Analysis project.

Loads the SYNTHETIC tables from <project root>/data/, prints a data quality
overview, descriptive statistics and several aggregations, and saves charts
to <project root>/figures/.

Run from the project root (after python src/data_generator.py):
    python src/eda.py
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

WEATHER_ORDER = ["clear", "cloudy", "rain", "snow"]
VEHICLE_ORDER = ["bike", "scooter", "car"]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load couriers and orders from CSV files."""
    couriers_path = DATA_DIR / "couriers.csv"
    orders_path = DATA_DIR / "orders.csv"
    if not couriers_path.exists() or not orders_path.exists():
        raise FileNotFoundError(
            "CSV files not found in data/. Run: python src/data_generator.py"
        )
    couriers = pd.read_csv(couriers_path)
    orders = pd.read_csv(orders_path, parse_dates=["order_datetime"])
    return couriers, orders


def print_overview(name: str, df: pd.DataFrame) -> None:
    """Print shape, dtypes, missing values and duplicates for one table."""
    print(f"\n=== {name} ===")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nData types:")
    print(df.dtypes.to_string())
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    print(f"\nDuplicated rows: {df.duplicated().sum()}")


def print_statistics(couriers: pd.DataFrame, orders: pd.DataFrame) -> None:
    """Print descriptive statistics for numeric columns."""
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)
    print("\n=== Descriptive statistics: couriers ===")
    print(couriers.drop(columns="courier_id").describe().round(2).to_string())
    print("\n=== Descriptive statistics: orders ===")
    num_cols = ["distance_km", "courier_active_orders", "prep_time_min", "delivery_time_min"]
    print(orders[num_cols].describe().round(2).to_string())


def print_aggregations(orders: pd.DataFrame, couriers: pd.DataFrame) -> pd.DataFrame:
    """Print meaningful aggregations; return orders merged with couriers."""
    df = orders.merge(couriers, on="courier_id", how="left")
    df["hour"] = df["order_datetime"].dt.hour
    df["distance_bin"] = pd.cut(
        df["distance_km"], bins=[0, 2, 4, 6, 8, 15],
        labels=["0-2", "2-4", "4-6", "6-8", "8+"],
    )

    def summarize(by: str, order=None) -> pd.DataFrame:
        res = df.groupby(by, observed=True)["delivery_time_min"].agg(
            orders="count", mean_delivery="mean", median_delivery="median"
        ).round(2)
        return res.loc[order] if order is not None else res

    print("\n=== Delivery time by weather ===")
    print(summarize("weather", WEATHER_ORDER).to_string())

    print("\n=== Delivery time by courier load (active orders) ===")
    print(summarize("courier_active_orders").to_string())

    print("\n=== Delivery time by vehicle type ===")
    print(summarize("vehicle_type", VEHICLE_ORDER).to_string())

    print("\n=== Delivery time by distance band (km) ===")
    print(summarize("distance_bin").to_string())

    print("\n=== Delivery time by peak hour flag ===")
    print(summarize("is_peak_hour").to_string())

    print("\n=== Correlation with delivery_time_min (Pearson) ===")
    corr_cols = ["distance_km", "courier_active_orders", "prep_time_min",
                 "is_peak_hour", "experience_months", "avg_rating"]
    corr = df[corr_cols].corrwith(df["delivery_time_min"]).round(3)
    print(corr.sort_values(ascending=False).to_string())
    return df


def make_charts(df: pd.DataFrame) -> None:
    """Save charts to figures/."""
    os.makedirs(FIG_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Distribution of the target variable
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["delivery_time_min"], bins=40, kde=True, ax=ax)
    ax.set_title("Distribution of delivery time (synthetic data)")
    ax.set_xlabel("Delivery time, minutes")
    ax.set_ylabel("Number of orders")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "delivery_time_distribution.png", dpi=120)
    plt.close(fig)

    # 2. Delivery time vs distance, colored by weather
    fig, ax = plt.subplots(figsize=(8, 5))
    sample = df.sample(n=min(2000, len(df)), random_state=42)
    sns.scatterplot(data=sample, x="distance_km", y="delivery_time_min",
                    hue="weather", hue_order=WEATHER_ORDER, alpha=0.6, s=18, ax=ax)
    ax.set_title("Delivery time vs distance by weather (sample of 2000 orders)")
    ax.set_xlabel("Distance, km")
    ax.set_ylabel("Delivery time, minutes")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "delivery_vs_distance.png", dpi=120)
    plt.close(fig)

    # 3. Boxplots by weather and by courier load
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=df, x="weather", y="delivery_time_min", order=WEATHER_ORDER, ax=axes[0])
    axes[0].set_title("Delivery time by weather")
    axes[0].set_xlabel("Weather")
    axes[0].set_ylabel("Delivery time, minutes")
    sns.boxplot(data=df, x="courier_active_orders", y="delivery_time_min", ax=axes[1])
    axes[1].set_title("Delivery time by courier load")
    axes[1].set_xlabel("Active orders of the courier")
    axes[1].set_ylabel("Delivery time, minutes")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "delivery_by_weather_and_load.png", dpi=120)
    plt.close(fig)

    # 4. Average delivery time by hour of day
    hourly = df.groupby("hour")["delivery_time_min"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hourly.index, hourly.values, marker="o")
    ax.set_title("Average delivery time by hour of day")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean delivery time, minutes")
    ax.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "delivery_by_hour.png", dpi=120)
    plt.close(fig)

    print(f"\nSaved charts to {FIG_DIR}")


def main() -> None:
    couriers, orders = load_data()
    print_overview("couriers", couriers)
    print_overview("orders", orders)
    print_statistics(couriers, orders)
    merged = print_aggregations(orders, couriers)
    make_charts(merged)


if __name__ == "__main__":
    main()
