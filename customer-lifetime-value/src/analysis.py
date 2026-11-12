"""Main CLV analysis on the SYNTHETIC dataset.

Steps:
  1. Validation: fit a BG/NBD model on orders up to a cutoff date and compare
     predictions with the orders that actually happened afterwards (holdout).
  2. Final model: refit on all data and predict revenue for the next 12 months.
  3. Compare historical and predicted CLV by segment and list top customers.
  4. Save charts to figures/ and metrics to reports/metrics.json.

The split is by time (not random) because the task is forecasting; no random
split is needed, so there is no random_state to set.

Run from the project root:
    python src/analysis.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: figures are only saved to files
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from utils import (  # noqa: E402
    OBSERVATION_END,
    build_rfm,
    expected_purchases,
    fit_bgnbd,
    holdout_actuals,
    load_data,
    naive_expected_purchases,
    prob_alive,
    rank_correlation,
    regression_metrics,
    segment_summary,
    shrunk_avg_order_value,
    top_share_captured,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "figures"
REPORTS_DIR = PROJECT_DIR / "reports"

HOLDOUT_DAYS = 182  # 26 weeks held out for validation
FORECAST_DAYS = 365  # prediction horizon for the final CLV

sns.set_theme(style="whitegrid")


def validate_model(customers: pd.DataFrame, transactions: pd.DataFrame):
    """Fit on the calibration period and evaluate on the holdout period."""
    cutoff = OBSERVATION_END - pd.Timedelta(days=HOLDOUT_DAYS)
    horizon_weeks = HOLDOUT_DAYS / 7.0

    rfm = build_rfm(customers, transactions, cutoff)
    params = fit_bgnbd(rfm)

    x, t_x, T = rfm["frequency"], rfm["recency_weeks"], rfm["T_weeks"]
    aov = shrunk_avg_order_value(rfm)

    rfm["p_alive"] = prob_alive(params, x, t_x, T)
    rfm["pred_orders_bgnbd"] = expected_purchases(params, horizon_weeks, x, t_x, T)
    rfm["pred_orders_naive"] = naive_expected_purchases(horizon_weeks, x, T)
    rfm["pred_revenue_bgnbd"] = rfm["pred_orders_bgnbd"] * aov
    rfm["pred_revenue_naive"] = rfm["pred_orders_naive"] * aov

    actual = holdout_actuals(customers, transactions, cutoff, OBSERVATION_END)
    rfm = rfm.merge(actual, on="customer_id", how="left")

    results = {
        "cutoff_date": str(cutoff.date()),
        "holdout_days": HOLDOUT_DAYS,
        "n_customers": int(len(rfm)),
        "bgnbd_params": params,
        "orders_bgnbd": regression_metrics(rfm["actual_orders"], rfm["pred_orders_bgnbd"]),
        "orders_naive": regression_metrics(rfm["actual_orders"], rfm["pred_orders_naive"]),
        "revenue_bgnbd": regression_metrics(rfm["actual_revenue"], rfm["pred_revenue_bgnbd"]),
        "revenue_naive": regression_metrics(rfm["actual_revenue"], rfm["pred_revenue_naive"]),
        "spearman_revenue_bgnbd": rank_correlation(rfm["actual_revenue"], rfm["pred_revenue_bgnbd"]),
        "spearman_revenue_naive": rfm_corr_safe(rfm),
        "top10pct_share_captured_bgnbd": top_share_captured(
            rfm["actual_revenue"], rfm["pred_revenue_bgnbd"]
        ),
        "top10pct_share_captured_naive": top_share_captured(
            rfm["actual_revenue"], rfm["pred_revenue_naive"]
        ),
        "top10pct_share_best_possible": top_share_captured(
            rfm["actual_revenue"], rfm["actual_revenue"]
        ),
    }
    return rfm, results


def rfm_corr_safe(rfm: pd.DataFrame) -> float:
    """Spearman correlation for the naive baseline (can be NaN if predictions are constant)."""
    return rank_correlation(rfm["actual_revenue"], rfm["pred_revenue_naive"])


def print_validation(results: dict) -> None:
    """Print the validation metrics with clear labels."""
    p = results["bgnbd_params"]
    print("\n=== Validation: BG/NBD fitted on the calibration period ===")
    print(f"Calibration ends: {results['cutoff_date']}  |  holdout: {results['holdout_days']} days")
    print(f"Customers in validation: {results['n_customers']}")
    print(
        f"Fitted parameters (time in weeks): r={p['r']:.4f}, alpha={p['alpha']:.4f}, "
        f"a={p['a']:.4f}, b={p['b']:.4f}"
    )
    print(f"Log-likelihood: {p['log_likelihood']:.2f}  |  optimizer converged: {p['converged']}")
    if not p["converged"]:
        print("WARNING: the optimizer did not report convergence; interpret results with care.")

    for label, key in (
        ("Holdout orders per customer", "orders"),
        ("Holdout revenue per customer", "revenue"),
    ):
        print(f"\n{label}:")
        table = pd.DataFrame(
            {"BG/NBD": results[f"{key}_bgnbd"], "Naive rate": results[f"{key}_naive"]}
        )
        print(table.round(3).to_string())

    print("\nRanking quality for holdout revenue:")
    print(f"Spearman correlation, BG/NBD:   {results['spearman_revenue_bgnbd']:.3f}")
    print(f"Spearman correlation, naive:    {results['spearman_revenue_naive']:.3f}")
    print(
        "Share of actual holdout revenue from the top 10% of customers by predicted revenue: "
        f"BG/NBD {results['top10pct_share_captured_bgnbd']:.1%}, "
        f"naive {results['top10pct_share_captured_naive']:.1%}, "
        f"best possible {results['top10pct_share_best_possible']:.1%}"
    )


def final_clv(customers: pd.DataFrame, transactions: pd.DataFrame):
    """Refit on all data and predict revenue for the next 12 months."""
    rfm = build_rfm(customers, transactions, OBSERVATION_END)
    params = fit_bgnbd(rfm)
    x, t_x, T = rfm["frequency"], rfm["recency_weeks"], rfm["T_weeks"]

    rfm["p_alive"] = prob_alive(params, x, t_x, T)
    rfm["predicted_12m_orders"] = expected_purchases(params, FORECAST_DAYS / 7.0, x, t_x, T)
    rfm["predicted_12m_revenue"] = rfm["predicted_12m_orders"] * shrunk_avg_order_value(rfm)
    return rfm, params


def print_final(rfm: pd.DataFrame, params: dict) -> dict:
    """Print the final CLV results and return summary numbers for the JSON report."""
    print("\n=== Final model: fitted on all data, forecast for the next 12 months ===")
    print(
        f"Fitted parameters (weeks): r={params['r']:.4f}, alpha={params['alpha']:.4f}, "
        f"a={params['a']:.4f}, b={params['b']:.4f}  |  converged: {params['converged']}"
    )
    hist_total = float(rfm["revenue"].sum())
    pred_total = float(rfm["predicted_12m_revenue"].sum())
    print(f"Historical revenue to date (all customers): {hist_total:,.2f}")
    print(f"Predicted revenue, next {FORECAST_DAYS} days (all customers): {pred_total:,.2f}")
    print(f"Mean probability of being alive: {rfm['p_alive'].mean():.3f}")

    for by, title in (("acquisition_channel", "channel"), ("country", "country")):
        print(f"\n--- Historical vs predicted CLV by {title} ---")
        print(segment_summary(rfm, by).to_string())

    print("\n--- Top 10 customers by predicted 12-month revenue ---")
    cols = [
        "customer_id", "acquisition_channel", "country", "n_orders",
        "revenue", "p_alive", "predicted_12m_revenue",
    ]
    top = rfm.sort_values("predicted_12m_revenue", ascending=False).head(10)[cols]
    print(top.round(2).to_string(index=False))

    ranked = rfm["predicted_12m_revenue"].sort_values(ascending=False)
    n_top = int(np.ceil(0.10 * len(ranked)))
    top_share = float(ranked.head(n_top).sum() / ranked.sum())
    print(f"\nTop 10% of customers ({n_top}) account for {top_share:.1%} of predicted 12-month revenue")

    return {
        "forecast_days": FORECAST_DAYS,
        "bgnbd_params": params,
        "historical_revenue_total": hist_total,
        "predicted_12m_revenue_total": pred_total,
        "mean_p_alive": float(rfm["p_alive"].mean()),
        "top10pct_share_of_predicted_revenue": top_share,
        "by_channel": segment_summary(rfm, "acquisition_channel").reset_index().to_dict("records"),
    }


def plot_decile_calibration(rfm: pd.DataFrame) -> None:
    """Mean predicted vs actual holdout revenue per predicted-revenue decile."""
    df = rfm.copy()
    df["decile"] = pd.qcut(df["pred_revenue_bgnbd"].rank(method="first"), 10, labels=False) + 1
    table = (
        df.groupby("decile")[["pred_revenue_bgnbd", "actual_revenue"]]
        .mean()
        .rename(columns={"pred_revenue_bgnbd": "Predicted", "actual_revenue": "Actual"})
        .reset_index()
        .melt(id_vars="decile", var_name="Type", value_name="Mean revenue per customer")
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=table, x="decile", y="Mean revenue per customer", hue="Type", ax=ax)
    ax.set_title("Holdout revenue: predicted vs actual by predicted decile (synthetic data)")
    ax.set_xlabel("Decile of predicted revenue (1 = lowest, 10 = highest)")
    ax.set_ylabel("Mean revenue per customer in holdout")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "holdout_calibration.png", dpi=120)
    plt.close(fig)


def plot_channel_clv(rfm: pd.DataFrame) -> None:
    """Average historical revenue vs predicted 12-month revenue by channel."""
    table = (
        segment_summary(rfm, "acquisition_channel")[
            ["avg_historical_revenue", "avg_predicted_12m_revenue"]
        ]
        .rename(
            columns={
                "avg_historical_revenue": "Historical revenue",
                "avg_predicted_12m_revenue": "Predicted next 12 months",
            }
        )
        .sort_values("Predicted next 12 months", ascending=False)
        .reset_index()
        .melt(id_vars="acquisition_channel", var_name="Measure", value_name="Average per customer")
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=table, x="acquisition_channel", y="Average per customer", hue="Measure", ax=ax)
    ax.set_title("Historical vs predicted CLV by acquisition channel (synthetic data)")
    ax.set_xlabel("Acquisition channel")
    ax.set_ylabel("Average revenue per customer")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "clv_by_channel.png", dpi=120)
    plt.close(fig)


def main() -> None:
    customers, transactions = load_data(DATA_DIR)
    print("NOTE: all data is synthetic, generated by src/data_generator.py.")

    val_rfm, validation = validate_model(customers, transactions)
    print_validation(validation)

    final_rfm, params = final_clv(customers, transactions)
    final_summary = print_final(final_rfm, params)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_decile_calibration(val_rfm)
    plot_channel_clv(final_rfm)

    report = {
        "data_note": "Synthetic data generated by src/data_generator.py",
        "validation": validation,
        "final_model": final_summary,
    }
    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved 2 charts to {FIGURES_DIR}")
    print(f"Saved metrics to {REPORTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
