"""Next-day hourly forecast for the SYNTHETIC household energy data.

Run from the project root (after src/data_generator.py):
    python src/analysis.py

Compares two naive baselines (same hour yesterday, same hour last week)
with a Ridge regression and a gradient boosting model. The final days of the
period are held out as a test set (chronological split, no shuffling).
Writes charts to figures/ and metrics to reports/metrics.json.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
FIG_DIR = PROJECT_DIR / "figures"
REPORT_DIR = PROJECT_DIR / "reports"

RANDOM_STATE = 42
TEST_DAYS = 6

pd.set_option("display.width", 120)


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n=== {title} ===")


def load_prepared() -> pd.DataFrame:
    """Read the CSV files and build the feature table."""
    households = pd.read_csv(DATA_DIR / "households.csv")
    weather = pd.read_csv(DATA_DIR / "weather_hourly.csv")
    energy = pd.read_csv(DATA_DIR / "energy_hourly.csv")
    df = utils.build_dataset(households, weather, energy)
    return utils.add_features(df)


def fit_models(train: pd.DataFrame) -> dict[str, BaseEstimator]:
    """Train the regression models on the training period.

    Returns a dictionary mapping the model name to the fitted estimator.
    """
    X, y = train[utils.FEATURE_COLUMNS], train[utils.TARGET]
    models = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.8, random_state=RANDOM_STATE),
    }
    for model in models.values():
        model.fit(X, y)
    return models


def plot_forecast(test: pd.DataFrame, preds: dict[str, np.ndarray],
                  best_name: str) -> None:
    """Total consumption of all households: actual vs forecasts."""
    frame = test[["timestamp", "kwh"]].copy()
    frame["baseline_24h"] = preds["baseline_same_hour_yesterday"]
    frame["best_model"] = preds[best_name]
    total = frame.groupby("timestamp").sum()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(total.index, total["kwh"], color="black", label="actual", linewidth=1.5)
    ax.plot(total.index, total["baseline_24h"], color="tab:orange",
            linestyle="--", label="baseline: same hour yesterday")
    ax.plot(total.index, total["best_model"], color="tab:blue",
            label=f"model: {best_name}")
    ax.set_xlabel("Time (test period)")
    ax.set_ylabel("Total consumption of all households (kWh per hour)")
    ax.set_title("Next-day forecast vs actual on the held-out period (synthetic data)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "forecast_vs_actual.png", dpi=120)
    plt.close(fig)


def plot_error_by_hour(test: pd.DataFrame, preds: dict[str, np.ndarray],
                       best_name: str) -> None:
    """Mean absolute error per hour of day for baseline and best model."""
    errors = pd.DataFrame({
        "hour": test["hour"],
        "baseline: same hour yesterday":
            (test["kwh"] - preds["baseline_same_hour_yesterday"]).abs(),
        f"model: {best_name}": (test["kwh"] - preds[best_name]).abs(),
    }).groupby("hour").mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    for col in errors.columns:
        ax.plot(errors.index, errors[col], marker="o", label=col)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean absolute error (kWh per household-hour)")
    ax.set_title("Forecast error by hour of day, test period (synthetic data)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "error_by_hour.png", dpi=120)
    plt.close(fig)


def main() -> None:
    full = load_prepared()
    data = utils.drop_incomplete(full)
    train, test = utils.chronological_split(data, TEST_DAYS)

    section("Setup")
    print(f"rows with complete lag features: {len(data)} (first 7 days dropped)")
    print(f"train: {len(train)} rows, {train['date'].min().date()} to {train['date'].max().date()}")
    print(f"test:  {len(test)} rows, {test['date'].min().date()} to {test['date'].max().date()}")
    print(f"features: {len(utils.FEATURE_COLUMNS)}; random_state={RANDOM_STATE}")

    models = fit_models(train)
    y_test = test[utils.TARGET]
    preds = {
        "baseline_same_hour_yesterday": test["lag_24"].to_numpy(),
        "baseline_same_hour_last_week": test["lag_168"].to_numpy(),
    }
    for name, model in models.items():
        preds[name] = model.predict(test[utils.FEATURE_COLUMNS])

    metrics = {name: utils.evaluate(y_test, p) for name, p in preds.items()}

    section("Test-set metrics (kWh per household-hour)")
    table = pd.DataFrame(metrics).T
    print(table.round(4).to_string())

    best_baseline = min(
        ("baseline_same_hour_yesterday", "baseline_same_hour_last_week"),
        key=lambda n: metrics[n]["mae"])
    best_model = min(models, key=lambda n: metrics[n]["mae"])
    mae_gain = utils.improvement_pct(metrics[best_baseline]["mae"],
                                     metrics[best_model]["mae"])
    rmse_gain = utils.improvement_pct(metrics[best_baseline]["rmse"],
                                      metrics[best_model]["rmse"])

    section("Best model vs best baseline")
    print(f"best baseline (lowest MAE): {best_baseline}")
    print(f"best model (lowest MAE):    {best_model}")
    print(f"MAE reduction vs baseline:  {mae_gain:.1f}%")
    print(f"RMSE reduction vs baseline: {rmse_gain:.1f}%")

    importances = pd.Series(models["gradient_boosting"].feature_importances_,
                            index=utils.FEATURE_COLUMNS).sort_values(ascending=False)
    section("Gradient boosting feature importance (top 8)")
    print(importances.head(8).round(4).to_string())

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    plot_forecast(test, preds, best_model)
    plot_error_by_hour(test, preds, best_model)

    summary = {
        "data": "synthetic",
        "random_state": RANDOM_STATE,
        "split": {
            "type": "chronological",
            "test_days": TEST_DAYS,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_start": str(train["date"].min().date()),
            "train_end": str(train["date"].max().date()),
            "test_start": str(test["date"].min().date()),
            "test_end": str(test["date"].max().date()),
        },
        "features": utils.FEATURE_COLUMNS,
        "metrics": metrics,
        "best_baseline": best_baseline,
        "best_model": best_model,
        "mae_reduction_vs_best_baseline_pct": mae_gain,
        "rmse_reduction_vs_best_baseline_pct": rmse_gain,
        "gradient_boosting_feature_importance": {
            k: float(v) for k, v in importances.items()},
    }
    with open(REPORT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    section("Files saved")
    print("figures/forecast_vs_actual.png")
    print("figures/error_by_hour.png")
    print("reports/metrics.json")


if __name__ == "__main__":
    main()
