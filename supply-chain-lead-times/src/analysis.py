"""Main analysis of supplier on-time delivery and lead time variability.

Steps:
1. Supplier, segment and country KPIs (on-time rate, lead time mean/std/CV).
2. Statistical tests of segment differences (chi-square, Kruskal-Wallis).
3. Models of late deliveries: logistic regression and gradient boosting
   (classification), plus linear regression of delay days.
4. Permutation importance on the test set to rank drivers of lateness.

Charts go to figures/ and a metrics summary to reports/metrics.json.
All data is SYNTHETIC.

Run from the project root:
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
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import (
    chi_square_on_time,
    classification_metrics,
    kruskal_delay,
    majority_baseline_accuracy,
    prepare_features,
    regression_metrics,
    segment_kpis,
    supplier_kpis,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
FIG_DIR = PROJECT_DIR / "figures"
REPORT_DIR = PROJECT_DIR / "reports"

RANDOM_STATE = 42
MIN_ORDERS = 30  # minimum orders for a supplier to be ranked


def load_merged() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load both tables and return (suppliers, orders, merged)."""
    suppliers = pd.read_csv(DATA_DIR / "suppliers.csv")
    orders = pd.read_csv(
        DATA_DIR / "purchase_orders.csv",
        parse_dates=["order_date", "promised_date", "delivery_date"],
    )
    merged = orders.merge(suppliers, on="supplier_id", how="left")
    return suppliers, orders, merged


def kpi_section(suppliers: pd.DataFrame, orders: pd.DataFrame,
                merged: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Print KPI tables and return supplier KPIs plus a summary dict."""
    sup = supplier_kpis(orders, suppliers)
    ranked = sup[sup["orders"] >= MIN_ORDERS]

    print("\n=== KPIs by country ===")
    print(segment_kpis(merged, "country").round(3).to_string())
    print("\n=== KPIs by reliability tier ===")
    print(segment_kpis(merged, "reliability_tier").round(3).to_string())

    cols = ["orders", "on_time_rate", "mean_delay", "cv_lead_time",
            "country", "reliability_tier"]
    print(f"\n=== Five best suppliers by on-time rate (>= {MIN_ORDERS} orders) ===")
    print(ranked.nlargest(5, "on_time_rate")[cols].round(3).to_string())
    print(f"\n=== Five worst suppliers by on-time rate (>= {MIN_ORDERS} orders) ===")
    print(ranked.nsmallest(5, "on_time_rate")[cols].round(3).to_string())

    rho, p_val = stats.spearmanr(ranked["cv_lead_time"], ranked["on_time_rate"])
    print(f"\nSpearman correlation, supplier lead time CV vs on-time rate "
          f"(n={len(ranked)}): rho={rho:.3f}, p={p_val:.4g}")

    summary = {
        "overall_on_time_rate": float(orders["on_time"].mean()),
        "mean_delay_days": float(orders["delay_days"].mean()),
        "n_suppliers_ranked": int(len(ranked)),
        "min_orders_for_ranking": MIN_ORDERS,
        "spearman_cv_vs_on_time": {"rho": float(rho), "p_value": float(p_val)},
    }
    return sup, summary


def test_section(merged: pd.DataFrame) -> dict:
    """Run and print statistical tests of segment differences."""
    print("\n=== Statistical tests ===")
    results = {}
    for by in ["reliability_tier", "transport_mode", "region", "category"]:
        res = chi_square_on_time(merged, by)
        results[f"chi2_on_time_by_{by}"] = res
        print(f"Chi-square on-time vs {by}: chi2={res['chi2']:.2f}, "
              f"dof={res['dof']}, p={res['p_value']:.4g}")

    q4 = merged.assign(season=np.where(merged["order_date"].dt.month >= 10, "Q4", "Q1-Q3"))
    res = chi_square_on_time(q4, "season")
    results["chi2_on_time_by_season"] = res
    print(f"Chi-square on-time vs season (Q4 vs rest): chi2={res['chi2']:.2f}, "
          f"dof={res['dof']}, p={res['p_value']:.4g}")

    res = kruskal_delay(merged, "reliability_tier")
    results["kruskal_delay_by_reliability_tier"] = res
    print(f"Kruskal-Wallis delay_days by reliability_tier: H={res['h_stat']:.2f}, "
          f"p={res['p_value']:.4g}")
    return results


def model_section(merged: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Train and evaluate models; return metrics and permutation importance."""
    X, y_late, y_delay = prepare_features(merged)
    X_train, X_test, yl_train, yl_test, yd_train, yd_test = train_test_split(
        X, y_late, y_delay, test_size=0.25, random_state=RANDOM_STATE,
        stratify=y_late,
    )
    print("\n=== Modelling ===")
    print(f"Train rows: {len(X_train)}, test rows: {len(X_test)}, "
          f"features: {X.shape[1]}")
    print(f"Late share in train: {yl_train.mean():.3f}, test: {yl_test.mean():.3f}")

    metrics = {"n_train": int(len(X_train)), "n_test": int(len(X_test)),
               "n_features": int(X.shape[1])}
    baseline_acc = majority_baseline_accuracy(yl_train, yl_test)
    metrics["baseline_majority_accuracy"] = baseline_acc
    print(f"Baseline accuracy (always predict majority class): {baseline_acc:.3f}")

    # Logistic regression on standardised features
    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    logit.fit(X_train, yl_train)
    logit_metrics = classification_metrics(yl_test, logit.predict_proba(X_test)[:, 1])
    metrics["logistic_regression"] = logit_metrics
    print("\nLogistic regression (target: late delivery), test set:")
    for k, v in logit_metrics.items():
        print(f"  {k}: {v:.3f}")

    odds = pd.Series(
        np.exp(logit[-1].coef_[0]), index=X.columns, name="odds_ratio_per_sd"
    ).sort_values(ascending=False)
    print("\nLogistic regression odds ratios per 1 SD of feature (top 5 / bottom 5):")
    print(pd.concat([odds.head(5), odds.tail(5)]).round(3).to_string())
    metrics["logistic_odds_ratios_per_sd"] = {k: float(v) for k, v in odds.items()}

    # Gradient boosting classifier
    gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
    gb.fit(X_train, yl_train)
    gb_metrics = classification_metrics(yl_test, gb.predict_proba(X_test)[:, 1])
    metrics["gradient_boosting"] = gb_metrics
    print("\nGradient boosting (target: late delivery), test set:")
    for k, v in gb_metrics.items():
        print(f"  {k}: {v:.3f}")

    # Permutation importance on the test set (ROC AUC drop)
    perm = permutation_importance(
        gb, X_test, yl_test, scoring="roc_auc", n_repeats=10,
        random_state=RANDOM_STATE,
    )
    importance = pd.DataFrame(
        {"feature": X.columns, "importance": perm.importances_mean,
         "std": perm.importances_std}
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    print("\nPermutation importance of gradient boosting (ROC AUC drop, test set):")
    print(importance.head(8).round(4).to_string(index=False))
    metrics["permutation_importance_top"] = {
        r.feature: float(r.importance) for r in importance.head(8).itertuples()
    }

    # Linear regression of delay days vs a mean-only baseline
    lin = LinearRegression().fit(X_train, yd_train)
    lin_metrics = regression_metrics(yd_test, lin.predict(X_test))
    base_pred = np.full(len(yd_test), yd_train.mean())
    base_metrics = regression_metrics(yd_test, base_pred)
    metrics["linear_regression_delay"] = lin_metrics
    metrics["delay_mean_baseline"] = base_metrics
    print("\nLinear regression of delay_days, test set:")
    for k, v in lin_metrics.items():
        print(f"  {k}: {v:.3f}")
    print("Mean-only baseline for delay_days, test set:")
    for k, v in base_metrics.items():
        print(f"  {k}: {v:.3f}")
    return metrics, importance


def save_charts(sup: pd.DataFrame, importance: pd.DataFrame) -> None:
    """Save the supplier scatter chart and the importance chart."""
    os.makedirs(FIG_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Supplier on-time rate vs lead time variability
    ranked = sup[sup["orders"] >= MIN_ORDERS].reset_index()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(data=ranked, x="cv_lead_time", y="on_time_rate",
                    hue="reliability_tier", hue_order=["A", "B", "C"],
                    size="orders", sizes=(30, 220), ax=ax)
    ax.set_title(f"Supplier on-time rate vs lead time variability\n"
                 f"(suppliers with >= {MIN_ORDERS} orders, synthetic data)")
    ax.set_xlabel("Coefficient of variation of actual lead time")
    ax.set_ylabel("On-time rate")
    ax.legend(title="Tier / orders", fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "supplier_otd_vs_variability.png", dpi=120)
    plt.close(fig)

    # 2. Permutation importance
    top = importance.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color="steelblue")
    ax.set_title("Drivers of late delivery: permutation importance\n"
                 "(gradient boosting, test set, synthetic data)")
    ax.set_xlabel("Mean drop in ROC AUC when feature is shuffled")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "late_delivery_feature_importance.png", dpi=120)
    plt.close(fig)
    print(f"\nSaved 2 charts to {FIG_DIR}")


def main() -> None:
    """Run the full analysis and write reports/metrics.json."""
    suppliers, orders, merged = load_merged()
    sup, summary = kpi_section(suppliers, orders, merged)
    tests = test_section(merged)
    model_metrics, importance = model_section(merged)
    save_charts(sup, importance)

    report = {
        "data_note": "Synthetic data generated by src/data_generator.py",
        "random_state": RANDOM_STATE,
        "kpis": summary,
        "statistical_tests": tests,
        "models": model_metrics,
    }
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(REPORT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved metrics summary to {REPORT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
