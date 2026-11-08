"""Exploratory data analysis for the SYNTHETIC crypto price data.

Loads the CSV tables produced by src/data_generator.py, prints basic
checks (shape, dtypes, missing values, descriptive statistics, aggregations)
and saves a few charts to figures/.

The data is simulated and is NOT real market data.

Run from the project root:
    python src/eda.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "figures"

DAYS_PER_YEAR = 365
ROLLING_WINDOW = 30  # days


def load_data() -> dict:
    """Load the three CSV tables from data/."""
    needed = ["assets", "market_regimes", "daily_prices"]
    tables = {}
    for name in needed:
        path = DATA_DIR / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run 'python src/data_generator.py' first."
            )
        tables[name] = pd.read_csv(path)
    tables["market_regimes"]["date"] = pd.to_datetime(tables["market_regimes"]["date"])
    tables["daily_prices"]["date"] = pd.to_datetime(tables["daily_prices"]["date"])
    return tables


def print_overview(tables: dict) -> None:
    """Print shape, dtypes, missing values and duplicate checks per table."""
    for name, df in tables.items():
        print(f"\n=== Table: {name} ===")
        print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        print("Dtypes:")
        print(df.dtypes.to_string())
        print("Missing values per column:")
        print(df.isna().sum().to_string())

    prices = tables["daily_prices"]
    print("\n=== Data quality checks ===")
    print("Duplicate (date, asset) rows:",
          int(prices.duplicated(["date", "asset"]).sum()))
    print("Non-positive close prices:", int((prices["close_usd"] <= 0).sum()))
    print("Non-positive volumes:", int((prices["volume_usd"] <= 0).sum()))
    unknown = set(prices["asset"]) - set(tables["assets"]["asset"])
    print("Assets in prices but not in assets table:", sorted(unknown))
    missing_dates = set(prices["date"]) - set(tables["market_regimes"]["date"])
    print("Price dates missing from regimes table:", len(missing_dates))


def print_descriptive_stats(prices: pd.DataFrame) -> None:
    """Print descriptive statistics of the main numeric columns."""
    print("\n=== Descriptive statistics (all assets pooled) ===")
    print(prices[["close_usd", "log_return", "volume_usd"]].describe().to_string())


def per_asset_return_stats(prices: pd.DataFrame) -> pd.DataFrame:
    """Aggregation 1: return distribution statistics per asset."""
    def summarize(r: pd.Series) -> pd.Series:
        return pd.Series(
            {
                "mean_daily": r.mean(),
                "std_daily": r.std(),
                "ann_vol": r.std() * np.sqrt(DAYS_PER_YEAR),
                "skew": stats.skew(r),
                "excess_kurtosis": stats.kurtosis(r),  # Fisher: normal = 0
                "min": r.min(),
                "max": r.max(),
                "jarque_bera_p": stats.jarque_bera(r).pvalue,
            }
        )

    return prices.groupby("asset")["log_return"].apply(
        lambda r: summarize(r)
    ).unstack()


def regime_volatility(prices: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    """Aggregation 2: daily return std and mean volume per asset and regime."""
    merged = prices.merge(regimes, on="date", how="left")
    return merged.pivot_table(
        index="asset", columns="regime", values="log_return", aggfunc="std"
    )


def regime_shares(regimes: pd.DataFrame) -> pd.DataFrame:
    """Aggregation 3: number and share of days in each regime."""
    counts = regimes["regime"].value_counts()
    return pd.DataFrame({"days": counts, "share": counts / counts.sum()})


def volume_by_move_size(prices: pd.DataFrame) -> pd.DataFrame:
    """Aggregation 4: mean volume relative to the asset's average, by |return| quartile."""
    df = prices.copy()
    df["abs_return"] = df["log_return"].abs()
    # Quartiles within each asset so assets of different scale are comparable
    df["move_quartile"] = df.groupby("asset")["abs_return"].transform(
        lambda s: pd.qcut(s, 4, labels=["Q1 (small)", "Q2", "Q3", "Q4 (large)"])
    )
    df["rel_volume"] = df["volume_usd"] / df.groupby("asset")["volume_usd"].transform("mean")
    return df.groupby("move_quartile", observed=True)["rel_volume"].mean().to_frame()


def return_correlation(prices: pd.DataFrame) -> pd.DataFrame:
    """Aggregation 5: correlation matrix of daily log returns."""
    wide = prices.pivot(index="date", columns="asset", values="log_return")
    return wide.corr()


def plot_normalized_prices(prices: pd.DataFrame) -> None:
    """Chart 1: prices rebased to 100 on the first day (log scale)."""
    wide = prices.pivot(index="date", columns="asset", values="close_usd")
    rebased = wide / wide.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    rebased.plot(ax=ax)
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.3)
    ax.set_title("Simulated prices rebased to 100 (log scale, synthetic data)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rebased price index (first day = 100, log scale)")
    ax.legend(title="Asset")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "prices_rebased.png", dpi=120)
    plt.close(fig)


def plot_return_distributions(prices: pd.DataFrame) -> None:
    """Chart 2: histogram of standardized returns vs the normal density."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    grid = np.linspace(-6, 6, 300)
    for ax, (asset, grp) in zip(axes.ravel(), prices.groupby("asset")):
        r = grp["log_return"]
        z = (r - r.mean()) / r.std()
        ax.hist(z, bins=50, range=(-6, 6), density=True, alpha=0.6)
        ax.plot(grid, stats.norm.pdf(grid), color="black", lw=1,
                label="Normal")
        ax.set_title(asset)
        ax.set_xlabel("Standardized log return (z-score, unitless)")
        ax.set_ylabel("Density (per unit z)")
    axes[0, 0].legend()
    fig.suptitle("Distribution of standardized daily returns (synthetic data)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "return_distributions.png", dpi=120)
    plt.close(fig)


def plot_rolling_volatility(prices: pd.DataFrame, regimes: pd.DataFrame) -> None:
    """Chart 3: rolling annualized volatility with turbulent days shaded."""
    wide = prices.pivot(index="date", columns="asset", values="log_return")
    rolling = wide.rolling(ROLLING_WINDOW).std() * np.sqrt(DAYS_PER_YEAR)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    rolling.plot(ax=ax)

    # Shade turbulent days (one-day-wide spans, merged visually)
    turbulent = regimes.loc[regimes["regime"] == "turbulent", "date"]
    for d in turbulent:
        ax.axvspan(d, d + pd.Timedelta(days=1), color="red", alpha=0.08, lw=0)

    # Two-line title so it is not cut off at this figure width
    ax.set_title(f"{ROLLING_WINDOW}-day rolling annualized volatility\n"
                 "(red shading = turbulent days, synthetic data)")
    ax.set_xlabel("Date")
    # Ticks are shown in percent of annual volatility (0.5 -> 50%)
    ax.set_ylabel("Annualized volatility (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, alpha=0.3)
    ax.legend(title="Asset", ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rolling_volatility.png", dpi=120)
    plt.close(fig)


def plot_correlation_heatmap(corr: pd.DataFrame) -> None:
    """Chart 4: heatmap of the return correlation matrix."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, square=True, ax=ax,
                cbar_kws={"label": "Correlation coefficient (-1 to 1)"})
    ax.set_title("Correlation of daily log returns (synthetic data)")
    ax.set_xlabel("Asset")
    ax.set_ylabel("Asset")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "return_correlation.png", dpi=120)
    plt.close(fig)


def main() -> None:
    """Run the full EDA: printed summaries and saved charts."""
    FIGURES_DIR.mkdir(exist_ok=True)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

    tables = load_data()
    prices = tables["daily_prices"]
    regimes = tables["market_regimes"]

    print_overview(tables)
    print_descriptive_stats(prices)

    print("\n=== Aggregation 1: return distribution per asset ===")
    print(per_asset_return_stats(prices).to_string())

    print("\n=== Aggregation 2: daily return std by regime ===")
    print(regime_volatility(prices, regimes).to_string())

    print("\n=== Aggregation 3: days per regime ===")
    print(regime_shares(regimes).to_string())

    print("\n=== Aggregation 4: relative volume by |return| quartile ===")
    print(volume_by_move_size(prices).to_string())

    corr = return_correlation(prices)
    print("\n=== Aggregation 5: correlation of daily log returns ===")
    print(corr.to_string())

    plot_normalized_prices(prices)
    plot_return_distributions(prices)
    plot_rolling_volatility(prices, regimes)
    plot_correlation_heatmap(corr)
    print(f"\nSaved 4 charts to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
