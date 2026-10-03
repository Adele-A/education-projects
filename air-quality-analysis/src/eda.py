"""Exploratory data analysis for the SYNTHETIC air quality data.

Loads data/stations.csv and data/daily_measurements.csv, prints data quality
checks, descriptive statistics and several aggregations, and saves charts to
figures/. All numbers shown come from the data at run time.

Run from the project root:
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

POLLUTANTS = ["pm25", "pm10", "no2", "o3"]
WEATHER = ["temperature_c", "humidity_pct", "wind_speed_ms", "precipitation_mm", "pressure_hpa"]


def load_data():
    """Load both tables and merge station metadata into the measurements.
    
    Raises:
        FileNotFoundError: if required data files do not exist. 
            Run src/data_generator.py first to create them.
    """
    stations_path = DATA_DIR / "stations.csv"
    meas_path = DATA_DIR / "daily_measurements.csv"
    
    if not stations_path.exists() or not meas_path.exists():
        raise FileNotFoundError(
            f"Data files not found in {DATA_DIR}.\n"
            "Please run the data generator first:\n"
            "  python src/data_generator.py"
        )
    
    stations = pd.read_csv(stations_path)
    meas = pd.read_csv(meas_path, parse_dates=["date"])
    df = meas.merge(stations, on="station_id", how="left")
    return stations, meas, df


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def data_quality(stations, meas):
    """Print shape, dtypes, missing values and duplicate checks."""
    section("1. SHAPE AND DTYPES")
    print(f"stations: {stations.shape}")
    print(f"daily_measurements: {meas.shape}")
    print("\nDtypes of daily_measurements:")
    print(meas.dtypes.to_string())

    section("2. MISSING VALUES")
    missing = pd.DataFrame(
        {"missing": meas.isna().sum(), "share_pct": (meas.isna().mean() * 100).round(2)}
    )
    print(missing.to_string())

    section("3. KEY INTEGRITY")
    dup = meas.duplicated(subset=["date", "station_id"]).sum()
    print(f"Duplicate (date, station_id) keys: {dup}")
    orphan = (~meas["station_id"].isin(stations["station_id"])).sum()
    print(f"Rows with unknown station_id: {orphan}")
    print(f"Date range: {meas['date'].min().date()} to {meas['date'].max().date()}")

    section("4. DESCRIPTIVE STATISTICS")
    print(meas[POLLUTANTS + WEATHER].describe().round(2).T.to_string())


def aggregations(df):
    """Print aggregations and return the tables used for plotting."""
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.dayofweek

    section("5. AGGREGATIONS")

    by_station = df.groupby(["station_id", "station_type"])[POLLUTANTS].mean().round(1)
    print("\nMean pollutant levels by station:")
    print(by_station.to_string())

    by_year = df.groupby("year")[POLLUTANTS].mean().round(1)
    print("\nMean pollutant levels by year:")
    print(by_year.to_string())

    by_month = df.groupby("month")[POLLUTANTS + ["temperature_c"]].mean().round(1)
    print("\nMean values by calendar month:")
    print(by_month.to_string())

    by_weekday = df.groupby("weekday")[POLLUTANTS].mean().round(1)
    print("\nMean pollutant levels by weekday (0 = Monday):")
    print(by_weekday.to_string())

    # Rain vs. dry days (precipitation above 1 mm counts as rainy).
    df["rainy"] = df["precipitation_mm"] > 1.0
    rain = df.groupby("rainy")[POLLUTANTS].mean().round(1)
    print("\nMean pollutant levels, dry vs rainy days (rainy = precipitation > 1 mm):")
    print(rain.to_string())

    corr = df[POLLUTANTS + WEATHER].corr(method="spearman").round(2)
    print("\nSpearman correlation matrix (pairwise complete observations):")
    print(corr.to_string())

    return by_month, corr


def make_charts(df, by_month, corr):
    """Save charts to figures/."""
    os.makedirs(FIG_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Monthly mean PM2.5 time series per station.
    monthly = (
        df.set_index("date")
        .groupby("station_name")["pm25"]
        .resample("MS")
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=monthly, x="date", y="pm25", hue="station_name", ax=ax)
    ax.set_title("Monthly mean PM2.5 by station (synthetic data)")
    ax.set_xlabel("Month")
    ax.set_ylabel("PM2.5, ug/m3")
    ax.legend(title="Station")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pm25_monthly_by_station.png", dpi=120)
    plt.close(fig)

    # 2. Seasonality: mean pollutants by calendar month.
    fig, ax = plt.subplots(figsize=(9, 5))
    for col in POLLUTANTS:
        ax.plot(by_month.index, by_month[col], marker="o", label=col)
    ax.set_title("Mean pollutant level by calendar month (synthetic data)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean concentration, ug/m3")
    ax.set_xticks(range(1, 13))
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "seasonality_by_month.png", dpi=120)
    plt.close(fig)

    # 3. PM2.5 distribution by station type.
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="station_type", y="pm25", ax=ax)
    ax.set_title("PM2.5 distribution by station type (synthetic data)")
    ax.set_xlabel("Station type")
    ax.set_ylabel("PM2.5, ug/m3")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pm25_by_station_type.png", dpi=120)
    plt.close(fig)

    # 4. Spearman correlation heatmap: pollutants vs weather.
    fig, ax = plt.subplots(figsize=(8, 6))
    sub = corr.loc[POLLUTANTS, WEATHER]
    sns.heatmap(sub, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Spearman correlation: pollutants vs weather (synthetic data)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correlation_heatmap.png", dpi=120)
    plt.close(fig)

    print(f"\nSaved 4 charts to {FIG_DIR}")


def main():
    stations, meas, df = load_data()
    data_quality(stations, meas)
    by_month, corr = aggregations(df)
    make_charts(df, by_month, corr)


if __name__ == "__main__":
    main()
