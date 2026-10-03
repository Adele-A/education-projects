"""Main analysis for the SYNTHETIC air quality data.

1. Spearman correlation between pollutants and weather (with p-values).
2. Linear regression for daily PM2.5 (log target) with weather, season and
   station type features, validated on a time-based split (train: before
   2024-01-01, test: 2024) and compared with a station-type mean baseline.
3. For comparison only: the same model on a random 80/20 split (random_state=42).

Saves charts to figures/ and metrics to reports/metrics.json.

Run from the project root:
    python src/analysis.py
"""
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (POLLUTANTS, WEATHER, load_data, prepare_model_data,  # noqa: E402
                   regression_metrics, smearing_factor, time_split)

PROJECT_DIR = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_DIR / "figures"
REPORT_DIR = PROJECT_DIR / "reports"

RANDOM_STATE = 42
SPLIT_CUTOFF = "2024-01-01"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def correlation_analysis(df):
    """Spearman rho and p-value for each pollutant-weather pair."""
    section("1. SPEARMAN CORRELATION: POLLUTANTS VS WEATHER")
    rows = []
    for p in POLLUTANTS:
        for w in WEATHER:
            pair = df[[p, w]].dropna()
            rho, pval = stats.spearmanr(pair[p], pair[w])
            rows.append({"pollutant": p, "weather": w, "rho": float(rho),
                         "p_value": float(pval), "n": int(len(pair))})
    res = pd.DataFrame(rows)
    print(res.pivot(index="pollutant", columns="weather", values="rho")
          .loc[POLLUTANTS, WEATHER].round(2).to_string())
    print("\nStrongest absolute correlations:")
    top = res.reindex(res["rho"].abs().sort_values(ascending=False).index).head(5)
    print(top.round(4).to_string(index=False))
    return res


def fit_log_model(X_train, y_train):
    """Standardised linear regression on log(PM2.5)."""
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(X_train, np.log(y_train))
    return model


def predict_pm25(model, X, smear):
    """Back-transform log predictions to ug/m3 using the smearing factor."""
    return np.exp(model.predict(X)) * smear


def model_analysis(df):
    """Train and evaluate the PM2.5 model; return metrics, test frame and model."""
    rows, X, y = prepare_model_data(df, target="pm25")
    train_mask, test_mask = time_split(rows, SPLIT_CUTOFF)
    X_tr, X_te, y_tr, y_te = X[train_mask], X[test_mask], y[train_mask], y[test_mask]

    section("2. PM2.5 REGRESSION (TIME-BASED SPLIT)")
    print(f"Usable rows: {len(rows)} | train: {len(X_tr)} (before {SPLIT_CUTOFF}) | test: {len(X_te)}")

    model = fit_log_model(X_tr, y_tr)
    resid = np.log(y_tr) - model.predict(X_tr)
    smear = smearing_factor(resid)
    pred_te = predict_pm25(model, X_te, smear)
    pred_tr = predict_pm25(model, X_tr, smear)

    # Baseline: mean PM2.5 of each station type in the training period.
    type_means = y_tr.groupby(rows.loc[train_mask, "station_type"]).mean()
    base_te = rows.loc[test_mask, "station_type"].map(type_means)

    m_model_test = regression_metrics(y_te, pred_te)
    m_model_train = regression_metrics(y_tr, pred_tr)
    m_base_test = regression_metrics(y_te, base_te)

    for name, m in [("Model (train)", m_model_train), ("Model (test)", m_model_test),
                    ("Baseline station-type mean (test)", m_base_test)]:
        print(f"{name:38s} MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  R2={m['r2']:.3f}")

    coefs = pd.Series(model.named_steps["linearregression"].coef_, index=X.columns)
    print("\nStandardised coefficients on log(PM2.5), sorted by absolute size:")
    print(coefs.reindex(coefs.abs().sort_values(ascending=False).index).round(3).to_string())

    # Comparison only: random split (ignores time order, so it can look optimistic).
    section("3. COMPARISON: RANDOM 80/20 SPLIT (random_state=42)")
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    m_r = fit_log_model(Xr_tr, yr_tr)
    smear_r = smearing_factor(np.log(yr_tr) - m_r.predict(Xr_tr))
    m_random = regression_metrics(yr_te, predict_pm25(m_r, Xr_te, smear_r))
    print(f"Random split test: MAE={m_random['mae']:.2f}  RMSE={m_random['rmse']:.2f}  R2={m_random['r2']:.3f}")

    metrics = {
        "split_cutoff": SPLIT_CUTOFF,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "smearing_factor": smear,
        "model_train": m_model_train,
        "model_test": m_model_test,
        "baseline_station_type_mean_test": m_base_test,
        "random_split_test": m_random,
        "standardised_coefficients_log_pm25": {k: float(v) for k, v in coefs.items()},
    }
    test_frame = rows.loc[test_mask].assign(pm25_pred=pred_te)
    return metrics, test_frame, coefs


def make_charts(test_frame, coefs):
    os.makedirs(FIG_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Chart 1: predicted vs actual PM2.5 on the test period.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(test_frame["pm25"], test_frame["pm25_pred"], s=8, alpha=0.4)
    lim = max(test_frame["pm25"].max(), test_frame["pm25_pred"].max())
    ax.plot([0, lim], [0, lim], color="red", linestyle="--", label="Perfect prediction")
    ax.set_title("Predicted vs actual PM2.5, test period (synthetic data)")
    ax.set_xlabel("Actual PM2.5, ug/m3")
    ax.set_ylabel("Predicted PM2.5, ug/m3")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pm25_pred_vs_actual.png", dpi=120)
    plt.close(fig)

    # Chart 2: standardised coefficients.
    fig, ax = plt.subplots(figsize=(8, 6))
    ordered = coefs.sort_values()
    ax.barh(ordered.index, ordered.values, color="steelblue")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Standardised coefficients, log(PM2.5) model (synthetic data)")
    ax.set_xlabel("Change in log(PM2.5) per 1 SD of the feature")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pm25_model_coefficients.png", dpi=120)
    plt.close(fig)
    print(f"\nSaved 2 charts to {FIG_DIR}")


def main():
    df = load_data()
    corr = correlation_analysis(df)
    metrics, test_frame, coefs = model_analysis(df)
    make_charts(test_frame, coefs)

    metrics["spearman_correlations"] = corr.to_dict(orient="records")
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(REPORT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {REPORT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
