"""Core analysis: MRR bridge, churn, retention, segments and cohort revenue.

Reads data/customers.csv and data/subscription_events.csv, prints the key
metrics, saves two charts to figures/ and a metrics summary to
reports/metrics.json.

All data is SYNTHETIC, produced by src/data_generator.py.

Run from the project root:
    python src/analysis.py
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from utils import (  # noqa: E402
    add_monthly_rates,
    bridge_reconciles,
    cohort_retention_pct,
    cohort_revenue,
    customer_mrr_matrix,
    load_data,
    monthly_mrr_bridge,
    pooled_cohort_retention,
    segment_retention,
    summarize_retention,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_DIR / "figures"
REPORTS_DIR = PROJECT_DIR / "reports"

SEGMENT_COLUMNS = ["initial_plan", "acquisition_channel", "company_size"]
COHORT_OFFSETS = [1, 3, 6, 12]


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def clean(obj):
    """Make nested structures JSON safe (NaN -> None, numpy -> python)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if hasattr(obj, "item"):
        obj = obj.item()
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def plot_bridge(bridge) -> None:
    """Stacked bars of MRR movements per month with the net change line."""
    months = bridge.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(months, bridge["new_mrr"], width=20, label="New", color="tab:green")
    ax.bar(months, bridge["expansion_mrr"], width=20, bottom=bridge["new_mrr"],
           label="Expansion", color="tab:cyan")
    ax.bar(months, bridge["contraction_mrr"], width=20, label="Contraction", color="tab:orange")
    ax.bar(months, bridge["churned_mrr"], width=20, bottom=bridge["contraction_mrr"],
           label="Churned", color="tab:red")
    net = bridge[["new_mrr", "expansion_mrr", "contraction_mrr", "churned_mrr"]].sum(axis=1)
    ax.plot(months, net, color="black", marker="o", markersize=3, label="Net change")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_title("Monthly MRR movements (synthetic data)")
    ax.set_xlabel("Month")
    ax.set_ylabel("MRR change, USD")
    ax.legend(ncol=5, loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mrr_bridge_monthly.png", dpi=120)
    plt.close(fig)


def plot_cohorts(retention_pct) -> None:
    """Heatmap of cohort MRR retention by months since signup."""
    data = retention_pct.copy()
    data.index = [str(p) for p in data.index]
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(data, cmap="YlGnBu", ax=ax, cbar_kws={"label": "MRR vs signup month, %"})
    ax.set_title("Cohort MRR retention by signup month (synthetic data)")
    ax.set_xlabel("Months since signup")
    ax.set_ylabel("Signup cohort")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cohort_mrr_retention.png", dpi=120)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    customers, events = load_data()

    # 1. MRR bridge and reconciliation
    bridge = monthly_mrr_bridge(events)
    matrix = customer_mrr_matrix(events, bridge.index)
    reconciles = bridge_reconciles(bridge, matrix)
    with_rates = add_monthly_rates(bridge)

    section("MRR BRIDGE (USD, by month)")
    cols = ["opening_mrr", "new_mrr", "expansion_mrr", "contraction_mrr",
            "churned_mrr", "closing_mrr"]
    print(bridge[cols].round(2).to_string())
    print(f"\nBridge reconciles with customer-level MRR: {reconciles}")

    # 2. Monthly rates
    section("MONTHLY CHURN AND RETENTION RATES")
    rate_cols = ["customer_churn_rate", "revenue_churn_rate", "grr", "nrr"]
    print(with_rates[rate_cols].round(4).to_string())

    overall = summarize_retention(bridge)
    section("OVERALL RETENTION (opening-MRR weighted, monthly)")
    print(f"Months used: {overall['months_used']}")
    print(f"Net revenue retention (NRR): {overall['avg_monthly_nrr']:.4f}")
    print(f"Gross revenue retention (GRR): {overall['avg_monthly_grr']:.4f}")
    print(f"Customer churn rate: {overall['avg_monthly_customer_churn_rate']:.4f}")
    print(f"Revenue churn rate: {overall['avg_monthly_revenue_churn_rate']:.4f}")

    final_mrr = float(bridge["closing_mrr"].iloc[-1])
    totals = {
        "total_new_mrr": float(bridge["new_mrr"].sum()),
        "total_expansion_mrr": float(bridge["expansion_mrr"].sum()),
        "total_contraction_mrr": float(bridge["contraction_mrr"].sum()),
        "total_churned_mrr": float(bridge["churned_mrr"].sum()),
        "final_mrr": final_mrr,
        "final_active_customers": int(bridge["active_end"].iloc[-1]),
    }
    section("PERIOD TOTALS (USD)")
    for key, value in totals.items():
        print(f"{key}: {value:,.2f}")

    # 3. Segments
    segments = {}
    for col in SEGMENT_COLUMNS:
        seg = segment_retention(events, customers, col)
        segments[col] = seg.to_dict(orient="index")
        section(f"SEGMENT COMPARISON: {col}")
        print(seg.round(4).to_string())

    # 4. Cohorts
    cohort_mrr = cohort_revenue(customers, matrix)
    retention_pct = cohort_retention_pct(cohort_mrr)
    pooled = {f"month_{k}": pooled_cohort_retention(cohort_mrr, k) for k in COHORT_OFFSETS}
    section("COHORT MRR RETENTION (pooled over cohorts that reached the offset)")
    for key, value in pooled.items():
        print(f"{key}: {value:.2f}%")

    # 5. Charts and report
    plot_bridge(bridge)
    plot_cohorts(retention_pct)
    print(f"\nSaved 2 charts to {FIGURES_DIR}")

    metrics = {
        "data_note": "Synthetic data generated by src/data_generator.py",
        "n_customers": int(len(customers)),
        "n_events": int(len(events)),
        "bridge_reconciles": reconciles,
        "overall_monthly_retention": overall,
        "period_totals": totals,
        "segments": segments,
        "pooled_cohort_mrr_retention_pct": pooled,
    }
    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(clean(metrics), f, indent=2)
    print(f"Saved metrics summary to {REPORTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
