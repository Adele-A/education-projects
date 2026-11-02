"""Main analysis: estimate price elasticity with log-log regressions.

Steps:
    1. Load the synthetic tables and build regression features.
    2. Fit a naive and a controlled pooled model (product fixed effects).
    3. Fit per-category models, naive vs. controlled, and compare the
       estimates with the elasticities used by the data generator.
    4. Evaluate out-of-sample fit with an 80/20 split (random_state=42).
    5. Save charts to figures/ and metrics to reports/metrics.json.

Run from the project root:
    python src/analysis.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from data_generator import CATEGORY_ELASTICITY
from utils import (
    classify_elasticity,
    extract_price_effect,
    fit_log_log_model,
    implied_revenue_change_pct,
    prepare_model_data,
    regression_metrics,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "figures"
REPORTS_DIR = PROJECT_DIR / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.2
PRICE_CHANGE_PCT = -10.0  # illustrative price cut for the revenue calculation

sns.set_theme(style="whitegrid")
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    products_path = DATA_DIR / "products.csv"
    sales_path = DATA_DIR / "weekly_sales.csv"
    if not products_path.exists() or not sales_path.exists():
        sys.exit("Data files not found. Run `python src/data_generator.py` first.")
    return pd.read_csv(products_path), pd.read_csv(sales_path, parse_dates=["week_start"])


def pooled_models(df: pd.DataFrame) -> dict:
    """Fit one elasticity for all categories, without and with controls."""
    out = {}
    for name, controls in [("naive", False), ("controlled", True)]:
        res = fit_log_log_model(df, controls=controls)
        out[name] = {**extract_price_effect(res), "r2": float(res.rsquared)}
    return out


def category_models(df: pd.DataFrame) -> pd.DataFrame:
    """Fit naive and controlled models per category and compare with truth."""
    rows = []
    for category, group in df.groupby("category"):
        naive = extract_price_effect(fit_log_log_model(group, controls=False))
        ctrl = extract_price_effect(fit_log_log_model(group, controls=True))
        true_e = CATEGORY_ELASTICITY[category]
        rows.append(
            {
                "category": category,
                "true_elasticity": true_e,
                "naive_elasticity": naive["elasticity"],
                "controlled_elasticity": ctrl["elasticity"],
                "std_err": ctrl["std_err"],
                "ci_low": ctrl["ci_low"],
                "ci_high": ctrl["ci_high"],
                "p_value": ctrl["p_value"],
                "n_obs": ctrl["n_obs"],
                "abs_error_controlled": abs(ctrl["elasticity"] - true_e),
                "abs_error_naive": abs(naive["elasticity"] - true_e),
                "true_in_ci": bool(ctrl["ci_low"] <= true_e <= ctrl["ci_high"]),
                "demand_type": classify_elasticity(ctrl["elasticity"]),
                "revenue_change_pct": implied_revenue_change_pct(ctrl["elasticity"], PRICE_CHANGE_PCT),
            }
        )
    return pd.DataFrame(rows).sort_values("controlled_elasticity").reset_index(drop=True)


def holdout_evaluation(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Fit per-category controlled models on a train split, score the test split."""
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    preds = pd.Series(index=test.index, dtype=float)
    for category, g_train in train.groupby("category"):
        model = fit_log_log_model(g_train, controls=True)
        g_test = test[test["category"] == category]
        preds.loc[g_test.index] = model.predict(g_test)

    # Baseline: predict each product's mean log units from the training data.
    product_mean = train.groupby("product_id")["log_units"].mean()
    baseline = test["product_id"].map(product_mean)

    metrics = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "model": regression_metrics(test["log_units"], preds),
        "product_mean_baseline": regression_metrics(test["log_units"], baseline),
    }
    eval_df = test[["category", "log_units"]].assign(predicted=preds)
    return metrics, eval_df


def plot_elasticity_estimates(summary: pd.DataFrame) -> None:
    """Estimated elasticities (with 95% CI) vs. naive and generator values."""
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(summary))
    xerr = np.vstack(
        [
            summary["controlled_elasticity"] - summary["ci_low"],
            summary["ci_high"] - summary["controlled_elasticity"],
        ]
    )
    ax.errorbar(summary["controlled_elasticity"], y, xerr=xerr, fmt="o", color="navy",
                capsize=4, label="Controlled estimate (95% CI)")
    ax.scatter(summary["naive_elasticity"], y, marker="s", color="darkorange", label="Naive estimate")
    ax.scatter(summary["true_elasticity"], y, marker="D", color="crimson", label="Generator value")
    ax.axvline(-1, color="grey", linestyle="--", linewidth=1)
    ax.text(-1, len(summary) - 0.4, " unit elasticity", color="grey", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(summary["category"])
    ax.set_xlabel("Price elasticity of demand")
    ax.set_title("Estimated price elasticity by category")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "elasticity_estimates.png", dpi=120)
    plt.close(fig)


def plot_holdout_fit(eval_df: pd.DataFrame) -> None:
    """Actual vs. predicted log units on the held-out test split."""
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(data=eval_df, x="predicted", y="log_units", hue="category", s=14, alpha=0.6, ax=ax)
    lims = [eval_df[["predicted", "log_units"]].min().min(), eval_df[["predicted", "log_units"]].max().max()]
    ax.plot(lims, lims, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted log(units)")
    ax.set_ylabel("Actual log(units)")
    ax.set_title("Hold-out test set: actual vs. predicted")
    ax.legend(title="Category", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "holdout_actual_vs_predicted.png", dpi=120)
    plt.close(fig)


def main() -> None:
    products, sales = load_data()
    df = prepare_model_data(products, sales)

    section("Data used for modelling")
    print(f"Rows in weekly_sales: {len(sales)}")
    print(f"Rows used (units_sold > 0): {len(df)}")
    print(f"Rows dropped (zero units, log undefined): {len(sales) - len(df)}")

    section("Pooled elasticity (all categories, product fixed effects)")
    pooled = pooled_models(df)
    for name, res in pooled.items():
        print(f"{name:>10}: elasticity = {res['elasticity']:.3f} "
              f"(SE {res['std_err']:.3f}, 95% CI [{res['ci_low']:.3f}, {res['ci_high']:.3f}]), "
              f"R2 = {res['r2']:.3f}")

    section("Elasticity by category: naive vs. controlled vs. generator value")
    summary = category_models(df)
    cols = ["category", "true_elasticity", "naive_elasticity", "controlled_elasticity",
            "std_err", "ci_low", "ci_high", "true_in_ci", "demand_type"]
    print(summary[cols].round(3).to_string(index=False))
    print(f"\nMean absolute error vs. generator value - naive: {summary['abs_error_naive'].mean():.3f}, "
          f"controlled: {summary['abs_error_controlled'].mean():.3f}")

    section(f"Implied revenue effect of a {PRICE_CHANGE_PCT:.0f}% price change (controlled estimates)")
    print(summary[["category", "controlled_elasticity", "demand_type", "revenue_change_pct"]]
          .round(3).to_string(index=False))
    print("Note: assumes a constant elasticity and no change in competitor prices or promotions.")

    section(f"Hold-out evaluation (test size {TEST_SIZE}, random_state={RANDOM_STATE})")
    holdout, eval_df = holdout_evaluation(df)
    print(f"Train rows: {holdout['n_train']}, test rows: {holdout['n_test']}")
    for name in ["model", "product_mean_baseline"]:
        m = holdout[name]
        print(f"{name:>22}: RMSE = {m['rmse']:.3f}, MAE = {m['mae']:.3f}, R2 = {m['r2']:.3f} (log units)")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_elasticity_estimates(summary)
    plot_holdout_fit(eval_df)

    metrics = {
        "random_state": RANDOM_STATE,
        "rows_used": int(len(df)),
        "rows_dropped_zero_units": int(len(sales) - len(df)),
        "pooled": pooled,
        "by_category": summary.to_dict(orient="records"),
        "mean_abs_error_naive": float(summary["abs_error_naive"].mean()),
        "mean_abs_error_controlled": float(summary["abs_error_controlled"].mean()),
        "holdout": holdout,
    }
    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    section("Outputs saved")
    for path in [FIGURES_DIR / "elasticity_estimates.png",
                 FIGURES_DIR / "holdout_actual_vs_predicted.png",
                 REPORTS_DIR / "metrics.json"]:
        print(path.relative_to(PROJECT_DIR))


if __name__ == "__main__":
    main()
