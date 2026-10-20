"""Seasonality analysis and SARIMAX forecast on SYNTHETIC pharmacy sales.

For every medicine category:
  1. Quantify seasonality on the training period (log-additive decomposition).
  2. Fit SARIMAX on log(weekly units) with exogenous drivers: promotion flag,
     temperature, a linear time term and yearly Fourier terms (a standard way
     to handle a 52-week season, since a seasonal ARIMA with period 52 is
     slow and unstable on 5 years of data). Errors follow an ARMA(1,1).
  3. Forecast the hold-out period (last 52 weeks) and compare with a
     seasonal naive baseline (the same week one year earlier).

The split is purely time-based, so no random_state is involved; the data
itself is generated with seed 42.

Run from the project root (after python src/data_generator.py):
    python src/analysis.py
"""
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import calendar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from utils import (
    build_exog,
    forecast_metrics,
    interval_coverage,
    seasonal_naive_forecast,
    seasonal_profile,
    temperature_climatology,
    time_split,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
FIG_DIR = PROJECT_DIR / "figures"
REPORT_DIR = PROJECT_DIR / "reports"

TEST_WEEKS = 52          # hold-out length: the last year
FOURIER_K = 3            # yearly harmonics used as exogenous terms
ARMA_ORDER = (1, 0, 1)   # (p, d, q) of the error process
ALPHA = 0.05             # 95% prediction interval

sns.set_theme(style="whitegrid")


def load_sales() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "weekly_sales.csv", parse_dates=["week_start"])


def fit_and_forecast(cat_df: pd.DataFrame) -> dict:
    """Split one category, fit SARIMAX and the baseline, return forecasts and metrics."""
    train, test = time_split(cat_df.sort_values("week_start"), TEST_WEEKS)
    train_idx = pd.DatetimeIndex(train["week_start"])
    test_idx = pd.DatetimeIndex(test["week_start"])
    origin = train_idx[0]

    # Promotions are planned, so they are known ahead; temperature is not,
    # so the test period uses the training climatology instead of actual values.
    x_train = build_exog(train_idx, train["promo_flag"], train["avg_temp_c"], origin, FOURIER_K)
    x_test = build_exog(test_idx, test["promo_flag"],
                        temperature_climatology(train, test_idx), origin, FOURIER_K)

    y_train = pd.Series(np.log(train["units_sold"].to_numpy(float)), index=train_idx)
    y_test = test["units_sold"].to_numpy(float)

    model = SARIMAX(y_train, exog=x_train, order=ARMA_ORDER, trend="n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        res = model.fit(disp=False, maxiter=300)

    fc = res.get_forecast(steps=len(test), exog=x_test)
    pred = np.exp(fc.predicted_mean.to_numpy())
    ci = np.exp(fc.conf_int(alpha=ALPHA).to_numpy())

    baseline = seasonal_naive_forecast(train["units_sold"].to_numpy(float), len(test))

    return {
        "train": train, "test": test,
        "pred": pred, "lower": ci[:, 0], "upper": ci[:, 1], "baseline": baseline,
        "metrics_model": forecast_metrics(y_test, pred),
        "metrics_baseline": forecast_metrics(y_test, baseline),
        "coverage": interval_coverage(y_test, ci[:, 0], ci[:, 1]),
        "promo_uplift_pct": float((np.exp(res.params["promo"]) - 1) * 100),
        "temp_coef": float(res.params["temp_c"]),
        "seasonality": seasonal_profile(pd.Series(train["units_sold"].to_numpy(float), index=train_idx)),
    }


def plot_forecasts(results: dict) -> None:
    """Actual vs SARIMAX vs baseline on the hold-out, one panel per category."""
    cats = list(results)
    fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharex=True)
    for ax, cat in zip(axes.flat, cats):
        r = results[cat]
        tail = r["train"].tail(26)
        ax.plot(tail["week_start"], tail["units_sold"], color="grey", lw=1, label="Train (last 26 weeks)")
        ax.plot(r["test"]["week_start"], r["test"]["units_sold"], color="black", lw=1.2, label="Actual")
        ax.plot(r["test"]["week_start"], r["pred"], color="tab:red", lw=1.2, label="SARIMAX")
        ax.plot(r["test"]["week_start"], r["baseline"], color="tab:blue", lw=1, ls="--",
                label="Seasonal naive")
        ax.fill_between(r["test"]["week_start"], r["lower"], r["upper"], color="tab:red", alpha=0.12)
        ax.set_title(cat.replace("_", " "), fontsize=10)
        ax.tick_params(axis="x", rotation=45)
    for ax in axes[:, 0]:
        ax.set_ylabel("Units sold per week")
    for ax in axes[-1, :]:
        ax.set_xlabel("Week start (Monday)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.suptitle("Hold-out forecast of weekly units by category, 95% interval shaded (synthetic data)")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(FIG_DIR / "forecast_vs_actual.png", dpi=120)
    plt.close(fig)


def plot_error_comparison(table: pd.DataFrame) -> None:
    """MAPE of SARIMAX vs the seasonal naive baseline per category."""
    plot_df = (
        table[["mape_model", "mape_baseline"]]
        .rename(columns={"mape_model": "SARIMAX", "mape_baseline": "Seasonal naive"})
        .rename(index=lambda s: s.replace("_", " "))
        .reset_index(names="category")
        .melt(id_vars="category", var_name="Method", value_name="MAPE (%)")
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_df, x="category", y="MAPE (%)", hue="Method", ax=ax)
    ax.set_title("Hold-out forecast error by category (synthetic data)")
    ax.set_xlabel("Medicine category")
    ax.set_ylabel("MAPE on the last 52 weeks (%)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "forecast_error_comparison.png", dpi=120)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    sales = load_sales()

    results = {cat: fit_and_forecast(g) for cat, g in sales.groupby("category")}

    # Seasonality table (training period only)
    seas = pd.DataFrame({c: r["seasonality"] for c, r in results.items()}).T
    seas["peak_month"] = seas["peak_month"].astype(int).map(lambda m: calendar.month_abbr[m])
    seas["trough_month"] = seas["trough_month"].astype(int).map(lambda m: calendar.month_abbr[m])
    seas = seas.sort_values("strength", ascending=False)
    print("=== Seasonality by category (training period, log-additive decomposition) ===")
    print(seas.to_string(float_format=lambda v: f"{v:.2f}"))

    # Forecast accuracy table
    table = pd.DataFrame(
        {
            c: {
                "mape_model": r["metrics_model"]["mape"],
                "mape_baseline": r["metrics_baseline"]["mape"],
                "rmse_model": r["metrics_model"]["rmse"],
                "rmse_baseline": r["metrics_baseline"]["rmse"],
                "interval_coverage": r["coverage"],
                "promo_uplift_pct": r["promo_uplift_pct"],
            }
            for c, r in results.items()
        }
    ).T
    print(f"\n=== Hold-out accuracy (last {TEST_WEEKS} weeks) ===")
    print(table.to_string(float_format=lambda v: f"{v:.2f}"))

    wins = int((table["mape_model"] < table["mape_baseline"]).sum())
    summary = {
        "mean_mape_model": float(table["mape_model"].mean()),
        "mean_mape_baseline": float(table["mape_baseline"].mean()),
        "mean_rmse_model": float(table["rmse_model"].mean()),
        "mean_rmse_baseline": float(table["rmse_baseline"].mean()),
        "mean_interval_coverage": float(table["interval_coverage"].mean()),
        "categories_where_sarimax_beats_baseline": wins,
        "n_categories": int(len(table)),
    }
    print("\n=== Summary over categories ===")
    print(f"Mean MAPE, SARIMAX:        {summary['mean_mape_model']:.2f}%")
    print(f"Mean MAPE, seasonal naive: {summary['mean_mape_baseline']:.2f}%")
    print(f"SARIMAX better in {wins} of {len(table)} categories (by MAPE)")
    print(f"Mean coverage of the {int((1 - ALPHA) * 100)}% interval: {summary['mean_interval_coverage']:.2f}")

    plot_forecasts(results)
    plot_error_comparison(table)
    print(f"\nSaved 2 charts to {FIG_DIR.relative_to(PROJECT_DIR)}/")

    report = {
        "settings": {
            "test_weeks": TEST_WEEKS,
            "fourier_k": FOURIER_K,
            "arma_order": list(ARMA_ORDER),
            "interval_alpha": ALPHA,
            "data": "synthetic",
        },
        "summary": summary,
        "seasonality": json.loads(seas.to_json(orient="index")),
        "per_category": json.loads(table.to_json(orient="index")),
    }
    path = REPORT_DIR / "metrics.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"Saved metrics to {path.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
