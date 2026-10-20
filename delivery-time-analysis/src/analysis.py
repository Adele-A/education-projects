"""Main analysis for the Delivery Time Analysis project (SYNTHETIC data).

Steps:
    1. Load and merge the orders and couriers tables.
    2. Split into train (80%) and test (20%) with random_state=42.
    3. Fit a mean baseline, a one-feature linear model, a multiple linear
       regression and a random forest; evaluate all on the hold-out set.
    4. Estimate the effect of distance, weather and courier load with OLS
       (statsmodels) on the training set.
    5. Save charts to figures/ and metrics to reports/metrics.json.

Run from the project root (after python src/data_generator.py):
    python src/analysis.py
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split

from utils import load_merged_data, prepare_features, regression_metrics

PROJECT_DIR = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_DIR / "figures"
REPORT_DIR = PROJECT_DIR / "reports"

SEED = 42
TEST_SIZE = 0.2


def fit_and_evaluate(X_train, X_test, y_train, y_test) -> tuple[dict, dict]:
    """Fit all models and compute hold-out metrics.

    Args:
        X_train, X_test: Feature matrices.
        y_train, y_test: Target vectors.

    Returns:
        A tuple (metrics, predictions): metrics maps model name to a metrics
        dict; predictions maps model name to test-set predictions.
    """
    models = {
        "baseline_mean": (DummyRegressor(strategy="mean"), list(X_train.columns)),
        "linear_distance_only": (LinearRegression(), ["distance_km"]),
        "linear_multiple": (LinearRegression(), list(X_train.columns)),
        "random_forest": (
            RandomForestRegressor(
                n_estimators=200, min_samples_leaf=5, random_state=SEED, n_jobs=-1
            ),
            list(X_train.columns),
        ),
    }
    metrics, predictions = {}, {}
    for name, (model, cols) in models.items():
        model.fit(X_train[cols], y_train)
        pred = model.predict(X_test[cols])
        predictions[name] = pred
        metrics[name] = regression_metrics(y_test, pred)
    return metrics, predictions


def ols_effects(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Fit OLS on the training set and return coefficients with 95% CIs.

    Args:
        X_train: Training features.
        y_train: Training target.

    Returns:
        DataFrame indexed by feature with coef, ci_low, ci_high, p_value.
    """
    fit = sm.OLS(y_train, sm.add_constant(X_train)).fit()
    ci = fit.conf_int()
    return pd.DataFrame({
        "coef": fit.params,
        "ci_low": ci[0],
        "ci_high": ci[1],
        "p_value": fit.pvalues,
    })


def make_charts(effects: pd.DataFrame, y_test, y_pred, model_name: str) -> None:
    """Save the coefficient chart and the predicted-vs-actual chart.

    Args:
        effects: Output of ols_effects().
        y_test: True test targets.
        y_pred: Predictions of the best model on the test set.
        model_name: Name of the best model, used in the chart title.
    """
    os.makedirs(FIG_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. OLS coefficients with 95% confidence intervals (without intercept)
    eff = effects.drop(index="const").sort_values("coef")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(eff["coef"], range(len(eff)),
                xerr=[eff["coef"] - eff["ci_low"], eff["ci_high"] - eff["coef"]],
                fmt="o", capsize=3)
    ax.axvline(0, color="grey", linestyle="--")
    ax.set_yticks(range(len(eff)))
    ax.set_yticklabels(eff.index)
    ax.set_xlabel("Change in delivery time, minutes (per unit of the feature)")
    ax.set_title("Linear regression coefficients with 95% CI (synthetic data)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_coefficients.png", dpi=120)
    plt.close(fig)

    # 2. Predicted vs actual on the hold-out set
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, s=10, alpha=0.5)
    lims = [min(np.min(y_test), np.min(y_pred)), max(np.max(y_test), np.max(y_pred))]
    ax.plot(lims, lims, color="red", linestyle="--", label="Perfect prediction")
    ax.set_xlabel("Actual delivery time, minutes")
    ax.set_ylabel("Predicted delivery time, minutes")
    ax.set_title(f"Predicted vs actual on test set ({model_name})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "predicted_vs_actual.png", dpi=120)
    plt.close(fig)


def main() -> None:
    """Run the analysis, print key metrics and save charts and metrics.json."""
    df = load_merged_data()
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )
    print(f"Rows: {len(df)} | train: {len(X_train)} | test: {len(X_test)}")
    print(f"Features ({X.shape[1]}): {', '.join(X.columns)}")

    metrics, predictions = fit_and_evaluate(X_train, X_test, y_train, y_test)

    # 5-fold cross-validated R2 of the multiple linear model (train set only)
    cv_r2 = cross_val_score(LinearRegression(), X_train, y_train, cv=5, scoring="r2")

    print("\n=== Hold-out test metrics (minutes for MAE / RMSE) ===")
    table = pd.DataFrame(metrics).T.round(3)
    print(table.to_string())
    print(f"\nLinear multiple, 5-fold CV R2 on train: "
          f"mean={cv_r2.mean():.3f}, std={cv_r2.std():.3f}")

    effects = ols_effects(X_train, y_train)
    print("\n=== OLS effects on delivery time (minutes, train set) ===")
    print(effects.round(3).to_string())

    # Best model by test RMSE
    best = min(metrics, key=lambda k: metrics[k]["rmse"])
    print(f"\nBest model by test RMSE: {best}")

    importances = pd.Series(
        RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                              random_state=SEED, n_jobs=-1)
        .fit(X_train, y_train).feature_importances_,
        index=X.columns,
    ).sort_values(ascending=False)
    print("\n=== Random forest feature importance (train set) ===")
    print(importances.round(3).to_string())

    make_charts(effects, y_test, predictions[best], best)

    os.makedirs(REPORT_DIR, exist_ok=True)
    summary = {
        "data": "synthetic",
        "seed": SEED,
        "test_size": TEST_SIZE,
        "n_rows": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": list(X.columns),
        "test_metrics": metrics,
        "linear_multiple_cv_r2_mean": float(cv_r2.mean()),
        "linear_multiple_cv_r2_std": float(cv_r2.std()),
        "best_model_by_rmse": best,
        "ols_effects": {k: {c: float(v) for c, v in row.items()}
                        for k, row in effects.to_dict(orient="index").items()},
        "random_forest_importance": {k: float(v) for k, v in importances.items()},
    }
    with open(REPORT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metrics to {REPORT_DIR / 'metrics.json'}")
    print(f"Saved charts to {FIG_DIR}")


if __name__ == "__main__":
    main()
