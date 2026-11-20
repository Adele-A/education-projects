"""Main analysis of the SYNTHETIC warehouse order picking data.

1. Regression: which factors drive picking time (linear model on log time and a
   gradient boosting model, evaluated on a held-out test set).
2. Clustering: k-means on scaled order characteristics to find order profiles.

Prints key metrics, saves charts to figures/ and metrics to reports/metrics.json.
Run from the project root:  python src/analysis.py
(Run src/data_generator.py first.)
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import (
    build_regression_data,
    choose_k,
    evaluate_k_range,
    load_merged,
    prepare_cluster_features,
    profile_clusters,
    regression_metrics,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_DIR / "figures"
REPORTS_DIR = PROJECT_DIR / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.2
# At least three profiles so the result is useful for planning
K_RANGE = range(3, 9)


def run_regression(merged: pd.DataFrame) -> tuple[dict, plt.Figure]:
    """Train and evaluate the regression models; return metrics and a chart."""
    X, y = build_regression_data(merged)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\n=== Regression: train {len(X_train)} rows, test {len(X_test)} rows ===")

    results: dict[str, dict] = {}

    # Baseline: always predict the training mean
    baseline_pred = np.full(len(y_test), y_train.mean())
    results["baseline_mean"] = regression_metrics(y_test, baseline_pred)

    # Linear regression on log(time): effects are roughly multiplicative
    linear = make_pipeline(StandardScaler(), LinearRegression())
    linear.fit(X_train, np.log1p(y_train))
    linear_pred = np.expm1(linear.predict(X_test)).clip(min=0)
    results["linear_log_target"] = regression_metrics(y_test, linear_pred)

    # Gradient boosting on the raw target
    gbr = GradientBoostingRegressor(random_state=RANDOM_STATE)
    gbr.fit(X_train, y_train)
    gbr_pred = gbr.predict(X_test)
    results["gradient_boosting"] = regression_metrics(y_test, gbr_pred)

    # 5-fold cross-validated R2 on the training set
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results["linear_log_target"]["cv_r2_train"] = float(
        cross_val_score(linear, X_train, np.log1p(y_train), cv=cv, scoring="r2").mean()
    )
    results["gradient_boosting"]["cv_r2_train"] = float(
        cross_val_score(gbr, X_train, y_train, cv=cv, scoring="r2").mean()
    )

    print("\nTest-set metrics (minutes; R2 unitless):")
    table = pd.DataFrame(results).T
    print(table.round(3).to_string())

    # Drivers: standardized linear coefficients (effect on log time) and GBR importances
    coefs = pd.Series(linear[-1].coef_, index=X.columns).sort_values(
        key=np.abs, ascending=False
    )
    print("\nLinear model: standardized coefficients on log(1 + time):")
    print(coefs.round(3).to_string())

    importances = pd.Series(gbr.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    print("\nGradient boosting: feature importances:")
    print(importances.round(3).to_string())

    # Chart: feature importances and actual vs predicted
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    importances.sort_values().plot.barh(ax=axes[0], color="steelblue")
    axes[0].set_title("Gradient boosting feature importances")
    axes[0].set_xlabel("Importance")

    axes[1].scatter(y_test, gbr_pred, s=10, alpha=0.4)
    lim = [0, float(max(y_test.max(), gbr_pred.max()))]
    axes[1].plot(lim, lim, color="red", linewidth=1, label="Perfect prediction")
    axes[1].set_title("Test set: actual vs predicted picking time")
    axes[1].set_xlabel("Actual picking time (min)")
    axes[1].set_ylabel("Predicted picking time (min)")
    axes[1].legend()
    fig.suptitle("Picking time model (synthetic data)")
    fig.tight_layout()

    out = {
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "models": results,
        "linear_standardized_coefficients": {k: float(v) for k, v in coefs.items()},
        "gradient_boosting_importances": {k: float(v) for k, v in importances.items()},
    }
    return out, fig


def run_clustering(merged: pd.DataFrame) -> tuple[dict, plt.Figure]:
    """Cluster orders with k-means; return metrics and a chart."""
    X, _ = prepare_cluster_features(merged)

    print("\n=== K-means: choice of k ===")
    scores = evaluate_k_range(X, K_RANGE, random_state=RANDOM_STATE)
    print(scores.round(3).to_string(index=False))
    best_k = choose_k(scores)
    print(f"\nSelected k (highest silhouette within {K_RANGE.start}-{K_RANGE.stop - 1}): {best_k}")

    km = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(X)
    profiles = profile_clusters(merged, labels)
    print("\nCluster profiles (means; sorted by picking time):")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(profiles.to_string())

    # Chart: clusters projected on the first two principal components
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x=coords[:, 0],
        y=coords[:, 1],
        hue=labels,
        palette="tab10",
        s=14,
        alpha=0.6,
        ax=ax,
    )
    ax.set_title("Order clusters in PCA space (synthetic data)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% of variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% of variance)")
    ax.legend(title="Cluster", fontsize=8)
    fig.tight_layout()

    out = {
        "k_scores": scores.to_dict(orient="records"),
        "selected_k": best_k,
        "cluster_profiles": {
            str(idx): {col: float(val) for col, val in row.items()}
            for idx, row in profiles.iterrows()
        },
    }
    return out, fig


def main() -> None:
    """Run regression and clustering, save charts and metrics."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    merged = load_merged()
    print(f"Loaded {len(merged)} orders with picker information")

    reg_metrics, reg_fig = run_regression(merged)
    reg_fig.savefig(FIGURES_DIR / "regression_results.png", dpi=120)
    plt.close(reg_fig)

    clu_metrics, clu_fig = run_clustering(merged)
    clu_fig.savefig(FIGURES_DIR / "cluster_profiles_pca.png", dpi=120)
    plt.close(clu_fig)

    metrics = {"regression": reg_metrics, "clustering": clu_metrics}
    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved charts to {FIGURES_DIR}")
    print(f"Saved metrics to {REPORTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
