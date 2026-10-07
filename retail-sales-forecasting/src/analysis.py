"""Seasonal decomposition and Holt-Winters forecasting of weekly sales.

For each product category the total weekly units (all stores) are:
  1. decomposed into trend, yearly seasonality and residual (multiplicative);
  2. forecast on a time-based holdout (last 26 weeks) with Holt-Winters,
     and compared with a naive and a seasonal-naive baseline.

Outputs: figures/decomposition_seasonal.png, figures/forecast_vs_actual.png,
reports/metrics.json. All data is synthetic.
Run from the project root: python src/analysis.py
(run python src/data_generator.py first).
"""

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

from utils import (SEASON_LENGTH, component_strength, evaluate_forecast,
                   holt_winters_forecast, naive_forecast,
                   seasonal_naive_forecast, time_split, weekly_category_series)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "figures"
REPORTS_DIR = PROJECT_DIR / "reports"

HORIZON = 26  # weeks held out for testing
MODELS = {
    "naive": naive_forecast,
    "seasonal_naive": seasonal_naive_forecast,
    "holt_winters": holt_winters_forecast,
}

sns.set_theme(style="whitegrid")
pd.set_option("display.width", 120)


def load_series() -> pd.DataFrame:
    """Load the sales table and build the weekly series per category."""
    path = DATA_DIR / "weekly_sales.csv"
    if not path.exists():
        sys.exit(f"Missing {path}. Run 'python src/data_generator.py' first.")
    sales = pd.read_csv(path, parse_dates=["week_start"])
    return weekly_category_series(sales)


def decompose_all(series: pd.DataFrame) -> tuple[dict, dict]:
    """Multiplicative decomposition per category; return results and strengths."""
    results, strengths = {}, {}
    for cat in series.columns:
        res = seasonal_decompose(series[cat], model="multiplicative",
                                 period=SEASON_LENGTH)
        results[cat] = res
        strengths[cat] = {
            "seasonal_strength": component_strength(res.seasonal, res.resid),
            "trend_strength": component_strength(res.trend, res.resid),
        }
    return results, strengths


def forecast_all(series: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    """Fit all models on the training period and score them on the holdout."""
    train, test = time_split(series, HORIZON)
    forecasts, metrics = {}, {}
    for cat in series.columns:
        forecasts[cat], metrics[cat] = {}, {}
        for name, func in MODELS.items():
            pred = func(train[cat], HORIZON)
            forecasts[cat][name] = pred
            metrics[cat][name] = evaluate_forecast(test[cat], pred)
    return forecasts, metrics, train, test


def plot_decomposition(results: dict) -> None:
    """Plot the estimated yearly seasonal index and the trend per category."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for cat, res in results.items():
        axes[0].plot(res.seasonal.index, res.seasonal.values, label=cat)
        axes[1].plot(res.trend.index, res.trend / res.trend.dropna().iloc[0], label=cat)
    axes[0].set_title("Seasonal index (multiplicative)")
    axes[0].set_ylabel("Index (1 = no seasonal effect)")
    axes[1].set_title("Trend relative to first available value")
    axes[1].set_ylabel("Relative trend level")
    for ax in axes:
        ax.set_xlabel("Week start")
        ax.legend()
    fig.suptitle("Seasonal decomposition of total weekly units (synthetic data)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "decomposition_seasonal.png", dpi=120)
    plt.close(fig)


def plot_forecasts(forecasts: dict, train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Plot actuals and forecasts on the holdout for each category."""
    cats = list(test.columns)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, cat in zip(axes.ravel(), cats):
        tail = train[cat].iloc[-52:]
        ax.plot(tail.index, tail.values, color="grey", label="Train (last 52 weeks)")
        ax.plot(test.index, test[cat].values, color="black", linewidth=1.8, label="Actual")
        ax.plot(test.index, forecasts[cat]["seasonal_naive"], "--", label="Seasonal naive")
        ax.plot(test.index, forecasts[cat]["holt_winters"], label="Holt-Winters")
        ax.set_title(cat, loc="left", fontsize=10)
        ax.set_ylabel("Units per week")
        ax.tick_params(axis="x", rotation=30)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Holdout forecasts, last {HORIZON} weeks (synthetic data)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "forecast_vs_actual.png", dpi=120)
    plt.close(fig)


def main() -> None:
    """Run decomposition and forecasting, print metrics and save outputs."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    series = load_series()
    print(f"Series: {series.shape[1]} categories x {len(series)} weeks "
          f"({series.index.min().date()} to {series.index.max().date()})")

    print("\n=== Seasonal decomposition (period = 52 weeks) ===")
    results, strengths = decompose_all(series)
    print(pd.DataFrame(strengths).T.round(3))

    forecasts, metrics, train, test = forecast_all(series)
    print(f"\n=== Holdout evaluation: train {len(train)} weeks, "
          f"test {len(test)} weeks ({test.index.min().date()} to {test.index.max().date()}) ===")
    rows = [{"category": c, "model": m, **vals}
            for c, per_model in metrics.items() for m, vals in per_model.items()]
    table = pd.DataFrame(rows)
    print(table.round(2).to_string(index=False))

    summary = table.groupby("model")[["mae", "rmse", "mape_pct"]].mean()
    print("\n=== Mean error across categories ===")
    print(summary.round(2))

    best = table.loc[table.groupby("category")["rmse"].idxmin(), ["category", "model"]]
    print("\n=== Best model per category (lowest RMSE) ===")
    print(best.to_string(index=False))

    plot_decomposition(results)
    plot_forecasts(forecasts, train, test)

    output = {
        "config": {"season_length": SEASON_LENGTH, "holdout_weeks": HORIZON,
                   "train_weeks": len(train)},
        "decomposition_strength": strengths,
        "holdout_metrics": metrics,
        "mean_across_categories": summary.to_dict(orient="index"),
        "best_model_by_rmse": dict(zip(best["category"], best["model"])),
    }
    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\nSaved reports/metrics.json, figures/decomposition_seasonal.png, "
          "figures/forecast_vs_actual.png")


if __name__ == "__main__":
    main()
