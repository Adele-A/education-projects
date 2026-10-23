"""Exploratory data analysis for the SYNTHETIC household energy data.

Run from the project root:
    python src/eda.py

Reads the CSV files in data/ (create them first with src/data_generator.py),
prints data quality checks, descriptive statistics and a few aggregations,
and saves charts to figures/.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: write files, never open windows
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
FIG_DIR = PROJECT_DIR / "figures"

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
HEATING_LABELS = {1: "electric heating", 0: "no electric heating"}

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n=== {title} ===")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three CSV tables and parse timestamps."""
    households = pd.read_csv(DATA_DIR / "households.csv")
    weather = pd.read_csv(DATA_DIR / "weather_hourly.csv", parse_dates=["timestamp"])
    energy = pd.read_csv(DATA_DIR / "energy_hourly.csv", parse_dates=["timestamp"])
    return households, weather, energy


def quality_checks(tables: dict[str, pd.DataFrame]) -> None:
    """Print shape, dtypes, missing values and duplicates for each table."""
    for name, df in tables.items():
        section(f"{name}: structure")
        print(f"shape: {df.shape}")
        print("dtypes:")
        print(df.dtypes.to_string())
        print("missing values per column:")
        print(df.isna().sum().to_string())
        print(f"duplicated rows: {df.duplicated().sum()}")


def integrity_checks(households: pd.DataFrame, weather: pd.DataFrame,
                     energy: pd.DataFrame) -> None:
    """Check keys and that every household has a full hourly series."""
    section("Integrity checks")
    print("unknown household ids in energy:",
          (~energy["household_id"].isin(households["household_id"])).sum())
    print("energy timestamps missing in weather:",
          (~energy["timestamp"].isin(weather["timestamp"])).sum())
    print("duplicated (household_id, timestamp) pairs:",
          energy.duplicated(["household_id", "timestamp"]).sum())
    print("rows per household:")
    print(energy.groupby("household_id").size().to_string())


def build_frame(households: pd.DataFrame, weather: pd.DataFrame,
                energy: pd.DataFrame) -> pd.DataFrame:
    """Join the tables and add calendar features."""
    df = energy.merge(households, on="household_id", how="left")
    df = df.merge(weather, on="timestamp", how="left")
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0 = Monday
    df["day_name"] = df["day_of_week"].map(dict(enumerate(DAY_NAMES)))
    df["is_weekend"] = df["day_of_week"] >= 5
    df["day_type"] = df["is_weekend"].map({True: "weekend", False: "weekday"})
    df["heating"] = df["has_electric_heating"].map(HEATING_LABELS)
    return df


def descriptive_stats(df: pd.DataFrame) -> None:
    """Print key descriptive statistics for numeric variables."""
    section("Descriptive statistics")
    print(df[["kwh", "temperature_c"]].describe().round(3).to_string())
    print("\nkWh skewness:", round(df["kwh"].skew(), 3))


def aggregations(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute and print the main aggregations; return them for plotting."""
    out = {}

    # 1. Average hourly profile, weekday vs weekend
    profile = df.groupby(["hour", "day_type"])["kwh"].mean().unstack().round(3)
    out["profile"] = profile
    section("1. Mean kWh by hour of day (weekday vs weekend)")
    print(profile.to_string())
    print(f"peak hour weekday: {profile['weekday'].idxmax()}:00, "
          f"weekend: {profile['weekend'].idxmax()}:00")

    # 2. Day of week totals (mean kWh per household-hour)
    dow = (df.groupby(["day_of_week", "day_name"])["kwh"]
           .agg(["mean", "sum"]).round(3).reset_index()
           .set_index("day_name").drop(columns="day_of_week"))
    section("2. Consumption by day of week")
    print(dow.rename(columns={"mean": "mean_kwh", "sum": "total_kwh"}).to_string())

    # 3. Household comparison
    hh_hour = df.groupby(["household_id", "hour"])["kwh"].mean()
    summary = (df.groupby(["household_id", "dwelling_type", "n_residents",
                           "floor_area_m2", "has_electric_heating"])["kwh"]
               .agg(mean_kwh="mean", max_kwh="max", total_kwh="sum")
               .round(3).reset_index().set_index("household_id"))
    summary["peak_hour"] = hh_hour.groupby("household_id").idxmax().str[1]
    out["households"] = summary
    section("3. Household comparison")
    print(summary.to_string())

    # 4. Temperature bands (quantile-based) by heating type
    df = df.copy()
    df["temp_band"] = pd.qcut(df["temperature_c"], 5, duplicates="drop")
    bands = (df.groupby(["temp_band", "heating"], observed=True)["kwh"]
             .mean().unstack().round(3))
    section("4. Mean kWh by temperature band and heating type")
    print(bands.to_string())

    # 5. Correlation between temperature and kWh, by heating type
    corr = (df.groupby("heating")[["temperature_c", "kwh"]]
            .apply(lambda g: g["temperature_c"].corr(g["kwh"])).round(3))
    section("5. Correlation of temperature with kWh (hourly rows)")
    print(corr.to_string())
    return out


def plot_hourly_profile(profile: pd.DataFrame) -> None:
    """Line chart of the average daily profile."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for col in profile.columns:
        ax.plot(profile.index, profile[col], marker="o", label=col)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean consumption (kWh per household-hour)")
    ax.set_title("Average hourly profile: weekday vs weekend (synthetic data)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hourly_profile.png", dpi=120)
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of mean kWh by day of week and hour."""
    pivot = df.pivot_table(index="day_name", columns="hour", values="kwh",
                           aggfunc="mean").reindex(DAY_NAMES)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "Mean kWh per household-hour"})
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Day of week")
    ax.set_title("Mean consumption by day of week and hour (synthetic data)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "heatmap_day_hour.png", dpi=120)
    plt.close(fig)


def plot_temperature(df: pd.DataFrame) -> None:
    """Scatter of temperature vs mean kWh, split by heating type."""
    # Average over households of each heating type for every hour
    hourly = (df.groupby(["timestamp", "heating"])
              .agg(kwh=("kwh", "mean"), temperature_c=("temperature_c", "first"))
              .reset_index())
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(data=hourly, x="temperature_c", y="kwh", hue="heating",
                    alpha=0.5, s=18, ax=ax)
    ax.set_xlabel("Outdoor temperature (°C)")
    ax.set_ylabel("Mean consumption (kWh per household-hour)")
    ax.set_title("Temperature vs consumption by heating type (synthetic data)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "temperature_vs_kwh.png", dpi=120)
    plt.close(fig)


def plot_households(summary: pd.DataFrame) -> None:
    """Bar chart of mean consumption per household."""
    data = summary.reset_index()
    data["heating"] = data["has_electric_heating"].map(HEATING_LABELS)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=data, x="household_id", y="mean_kwh", hue="heating", ax=ax)
    ax.set_xlabel("Household")
    ax.set_ylabel("Mean consumption (kWh per hour)")
    ax.set_title("Mean hourly consumption per household (synthetic data)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "household_comparison.png", dpi=120)
    plt.close(fig)


def main() -> None:
    households, weather, energy = load_data()
    quality_checks({"households": households, "weather_hourly": weather,
                    "energy_hourly": energy})
    integrity_checks(households, weather, energy)

    df = build_frame(households, weather, energy)
    descriptive_stats(df)
    results = aggregations(df)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_hourly_profile(results["profile"])
    plot_heatmap(df)
    plot_temperature(df)
    plot_households(results["households"])

    section("Charts saved")
    for path in sorted(FIG_DIR.glob("*.png")):
        print(f"figures/{path.name}")


if __name__ == "__main__":
    main()
