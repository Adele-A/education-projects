"""Exploratory data analysis for the SYNTHETIC airline delay dataset.

Loads the CSV tables from data/, prints a data quality summary, descriptive
statistics and several aggregations, and saves charts to figures/.

All numbers printed come from the data itself; the data is synthetic and
does not describe real flights or airlines.

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

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def load_data():
    """Load the three tables from data/."""
    airports = pd.read_csv(DATA_DIR / "airports.csv")
    # keep_default_na=False: carrier code "NA" must not be read as missing
    carriers = pd.read_csv(DATA_DIR / "carriers.csv", keep_default_na=False)
    flights = pd.read_csv(DATA_DIR / "flights.csv", keep_default_na=False,
                          na_values=[""])
    return airports, carriers, flights


def print_overview(name, df):
    """Print shape, dtypes and missing values of a table."""
    print(f"\n=== {name} ===")
    print("Shape:", df.shape)
    print("\nDtypes:")
    print(df.dtypes.to_string())
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    print("Duplicate rows:", int(df.duplicated().sum()))


def print_descriptives(flights):
    """Print descriptive statistics of the numeric flight columns."""
    cols = ["scheduled_dep_hour", "distance_miles", "weather_severity",
            "dep_delay_min", "arr_delay_min"]
    print("\n=== Descriptive statistics (flights) ===")
    print(flights[cols].describe().round(2).to_string())
    print("\nOverall delay rate (arr delay >= 15 min): "
          f"{flights['is_delayed'].mean():.3f}")
    print("Skewness of arr_delay_min:",
          round(float(flights["arr_delay_min"].skew()), 2))


def delay_rate_by(flights, column):
    """Flight count, delay rate and mean arrival delay by a column."""
    out = (flights.groupby(column)
           .agg(flights=("flight_id", "count"),
                delay_rate=("is_delayed", "mean"),
                mean_arr_delay=("arr_delay_min", "mean"))
           .round(3))
    return out


def print_aggregations(flights, carriers):
    """Print the main aggregations and return them for plotting."""
    by_airport = delay_rate_by(flights, "origin").sort_values(
        "delay_rate", ascending=False)
    print("\n=== Delay rate by origin airport ===")
    print(by_airport.to_string())

    by_carrier = delay_rate_by(flights, "carrier_code").join(
        carriers.set_index("carrier_code")[["carrier_name", "carrier_type"]]
    ).sort_values("delay_rate", ascending=False)
    print("\n=== Delay rate by carrier ===")
    print(by_carrier.to_string())

    by_hour = delay_rate_by(flights, "scheduled_dep_hour")
    print("\n=== Delay rate by scheduled departure hour ===")
    print(by_hour.to_string())

    by_month = delay_rate_by(flights, "month")
    print("\n=== Delay rate by month ===")
    print(by_month.to_string())

    # Weather bands: bin the continuous severity index
    bands = pd.cut(flights["weather_severity"], bins=[-0.01, 1, 2, 4, 10],
                   labels=["0-1", "1-2", "2-4", "4-10"])
    by_weather = (flights.assign(weather_band=bands)
                  .groupby("weather_band", observed=True)
                  .agg(flights=("flight_id", "count"),
                       delay_rate=("is_delayed", "mean"))
                  .round(3))
    print("\n=== Delay rate by weather severity band ===")
    print(by_weather.to_string())

    return by_airport, by_hour, by_month


def save_figure(fig, filename):
    """Save a figure to figures/ and close it."""
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=120)
    plt.close(fig)


def make_charts(flights, by_airport, by_hour, by_month):
    """Create and save the EDA charts."""
    os.makedirs(FIG_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Distribution of arrival delay
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(flights["arr_delay_min"], bins=60, ax=ax, color="steelblue")
    ax.axvline(15, color="red", linestyle="--", label="Delay threshold (15 min)")
    ax.set_title("Distribution of arrival delay (synthetic data)")
    ax.set_xlabel("Arrival delay (minutes)")
    ax.set_ylabel("Number of flights")
    ax.legend()
    save_figure(fig, "arrival_delay_distribution.png")

    # 2. Delay rate by origin airport
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=by_airport.index, y=by_airport["delay_rate"], ax=ax,
                color="darkorange")
    ax.set_title("Delay rate by origin airport (synthetic data)")
    ax.set_xlabel("Origin airport")
    ax.set_ylabel("Share of delayed flights")
    save_figure(fig, "delay_rate_by_airport.png")

    # 3. Delay rate by scheduled departure hour
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(x=by_hour.index, y=by_hour["delay_rate"], marker="o", ax=ax,
                 color="seagreen")
    ax.set_title("Delay rate by scheduled departure hour (synthetic data)")
    ax.set_xlabel("Scheduled departure hour")
    ax.set_ylabel("Share of delayed flights")
    save_figure(fig, "delay_rate_by_hour.png")

    # 4. Delay rate by month
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [MONTH_LABELS[m - 1] for m in by_month.index]
    sns.barplot(x=labels, y=by_month["delay_rate"], ax=ax, color="slateblue")
    ax.set_title("Delay rate by month (synthetic data)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Share of delayed flights")
    save_figure(fig, "delay_rate_by_month.png")

    print(f"\nCharts saved to: {FIG_DIR}")


def main():
    """Run the full EDA."""
    airports, carriers, flights = load_data()

    for name, df in [("airports", airports), ("carriers", carriers),
                     ("flights", flights)]:
        print_overview(name, df)

    print_descriptives(flights)
    by_airport, by_hour, by_month = print_aggregations(flights, carriers)
    make_charts(flights, by_airport, by_hour, by_month)


if __name__ == "__main__":
    main()
