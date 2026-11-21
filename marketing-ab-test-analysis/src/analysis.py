"""Main analysis of the SYNTHETIC landing page A/B test.

Steps: sample ratio check, conversion comparison (z-test, chi-square,
confidence intervals, effect size), power and sample size, revenue per
visitor, covariate-adjusted logistic regression and segment breakdowns.
Prints labelled results, saves charts to figures/ and metrics to
reports/metrics.json.

This is a randomized experiment analysed with inferential statistics, so
no train/test split is used.

Run from the project root (after data_generator.py):
    python src/analysis.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils  # noqa: E402
from utils import section  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_DIR / "figures"
REPORT_DIR = PROJECT_DIR / "reports"

ALPHA = 0.05
POWER = 0.80
RELATIVE_LIFTS = [0.05, 0.10, 0.15, 0.20]  # lifts used for the sample size table


def plot_conversion(counts: dict, path: Path) -> None:
    """Bar chart of conversion rate per group with 95% Wilson intervals."""
    groups = ["control", "variant"]
    rates, lows, highs = [], [], []
    for g in groups:
        x, n = counts[g]["x"], counts[g]["n"]
        lo, hi = utils.wilson_ci(x, n, ALPHA)
        rates.append(x / n)
        lows.append(x / n - lo)
        highs.append(hi - x / n)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(groups, rates, yerr=[lows, highs], capsize=6,
           color=["#8c8c8c", "#2a7ab0"])
    ax.set_title("Conversion rate with 95% CI (synthetic data)")
    ax.set_xlabel("Group")
    ax.set_ylabel("Conversion rate")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_segments(seg: pd.DataFrame, overall_diff: float, path: Path) -> None:
    """Forest plot of the conversion difference (variant - control) by segment."""
    seg = seg.iloc[::-1].reset_index(drop=True)
    y = range(len(seg))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        seg["diff"] * 100, list(y),
        xerr=[(seg["diff"] - seg["ci_low"]) * 100, (seg["ci_high"] - seg["diff"]) * 100],
        fmt="o", color="#2a7ab0", ecolor="#8c8c8c", capsize=3,
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(overall_diff * 100, color="#d95f02", linestyle="--",
               label="Overall difference")
    ax.set_yticks(list(y))
    ax.set_yticklabels(seg["segment"])
    ax.set_xlabel("Conversion difference, variant - control (percentage points)")
    ax.set_title("Difference by segment with 95% CI (synthetic data)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    users, orders = utils.load_data()
    df = utils.add_revenue(utils.prepare_features(users), orders)
    metrics: dict = {"alpha": ALPHA, "target_power": POWER}

    # 1. Sample ratio check
    section("1. Sample ratio check (intended 50/50 split)")
    counts = utils.group_counts(df)
    n1, x1 = counts["control"]["n"], counts["control"]["x"]
    n2, x2 = counts["variant"]["n"], counts["variant"]["x"]
    srm_p = utils.sample_ratio_test(n1, n2)
    print(f"Control visitors: {n1}")
    print(f"Variant visitors: {n2}")
    print(f"SRM chi-square p-value: {srm_p:.4f}")
    metrics["sample_ratio"] = {"n_control": n1, "n_variant": n2, "srm_p_value": srm_p}

    # 2. Conversion comparison
    section("2. Conversion rate comparison")
    p1, p2 = x1 / n1, x2 / n2
    z, p_z = utils.two_proportion_ztest(x1, n1, x2, n2)
    chi2, p_chi = utils.chi_square_test(x1, n1, x2, n2)
    diff, d_lo, d_hi = utils.diff_confidence_interval(x1, n1, x2, n2, ALPHA)
    upl, u_lo, u_hi = utils.relative_uplift_ci(x1, n1, x2, n2, ALPHA)
    h = utils.cohens_h(p1, p2)
    c_lo, c_hi = utils.wilson_ci(x1, n1, ALPHA)
    v_lo, v_hi = utils.wilson_ci(x2, n2, ALPHA)
    print(f"Control: {x1}/{n1} buyers, conversion = {p1:.4f} (95% CI {c_lo:.4f} to {c_hi:.4f})")
    print(f"Variant: {x2}/{n2} buyers, conversion = {p2:.4f} (95% CI {v_lo:.4f} to {v_hi:.4f})")
    print(f"Absolute difference: {diff * 100:.2f} pp (95% CI {d_lo * 100:.2f} to {d_hi * 100:.2f} pp)")
    print(f"Relative uplift: {upl * 100:.1f}% (95% CI {u_lo * 100:.1f}% to {u_hi * 100:.1f}%)")
    print(f"Two-proportion z-test: z = {z:.3f}, p-value = {p_z:.4f}")
    print(f"Chi-square test: chi2 = {chi2:.3f}, p-value = {p_chi:.4f}")
    print(f"Cohen's h: {h:.4f}")
    significant = p_z < ALPHA
    verdict = "statistically significant" if significant else "NOT statistically significant"
    print(f"Verdict at alpha = {ALPHA}: the difference is {verdict}.")
    metrics["conversion"] = {
        "rate_control": p1, "rate_variant": p2,
        "ci_control": [c_lo, c_hi], "ci_variant": [v_lo, v_hi],
        "abs_difference": diff, "abs_diff_ci": [d_lo, d_hi],
        "relative_uplift": upl, "relative_uplift_ci": [u_lo, u_hi],
        "z_stat": z, "z_p_value": p_z,
        "chi2_stat": chi2, "chi2_p_value": p_chi,
        "cohens_h": h, "significant": bool(significant),
    }

    # 3. Power and sample size
    section("3. Power and sample size")
    mde_rate = utils.minimum_detectable_rate(p1, min(n1, n2), ALPHA, POWER)
    obs_power = utils.achieved_power(p1, p2, n1, n2, ALPHA)
    print(f"Minimum detectable variant rate at {POWER:.0%} power with n = {min(n1, n2)} per group: "
          f"{mde_rate:.4f} (absolute lift {(mde_rate - p1) * 100:.2f} pp, "
          f"relative {(mde_rate / p1 - 1) * 100:.1f}%)")
    print(f"Post-hoc power at the observed effect: {obs_power:.3f} "
          "(informational only; it restates the p-value)")
    print(f"\nRequired visitors per group ({POWER:.0%} power, alpha = {ALPHA}), "
          f"baseline = control rate {p1:.4f}:")
    size_table = []
    for lift in RELATIVE_LIFTS:
        n_req = utils.required_sample_size(p1, p1 * (1 + lift), ALPHA, POWER)
        size_table.append({"relative_lift": lift, "n_per_group": n_req})
        print(f"  relative lift {lift * 100:>4.0f}%: {n_req} per group")
    metrics["power"] = {
        "min_detectable_rate": mde_rate,
        "min_detectable_relative_lift": mde_rate / p1 - 1,
        "post_hoc_power": obs_power,
        "required_sample_sizes": size_table,
    }

    # 4. Revenue per visitor
    section("4. Revenue per visitor (secondary metric)")
    rev = utils.revenue_comparison(df, seed=42, alpha=ALPHA)
    print(f"Revenue per visitor, control: {rev['rpv_control']:.2f}")
    print(f"Revenue per visitor, variant: {rev['rpv_variant']:.2f}")
    print(f"Difference: {rev['rpv_difference']:.2f} "
          f"(bootstrap 95% CI {rev['rpv_diff_ci_low']:.2f} to {rev['rpv_diff_ci_high']:.2f})")
    print(f"Welch t-test: t = {rev['welch_t']:.3f}, p-value = {rev['welch_p_value']:.4f}")
    aov = df[df["converted"] == 1].groupby("group")["revenue"].mean()
    print("\nAverage order value (buyers only):")
    print(aov.round(2).to_string())
    metrics["revenue"] = rev
    metrics["avg_order_value"] = {k: float(v) for k, v in aov.items()}

    # 5. Logistic regression
    section("5. Logistic regression (adjusted for covariates)")
    model = utils.fit_logistic(df)
    orr = utils.variant_odds_ratio(model, ALPHA)
    print(f"Variant odds ratio: {orr['odds_ratio']:.3f} "
          f"(95% CI {orr['or_ci_low']:.3f} to {orr['or_ci_high']:.3f}), "
          f"p-value = {orr['p_value']:.4f}")
    metrics["logistic_regression"] = orr

    # 6. Segment breakdowns
    section("6. Segment breakdowns (variant vs control)")
    seg = pd.concat(
        [utils.segment_breakdown(df, c, ALPHA) for c in ["device", "traffic_source", "country"]],
        ignore_index=True,
    )
    seg["p_holm"] = utils.holm_adjust(seg["p_value"])
    show = seg.copy()
    for c in ["rate_control", "rate_variant", "diff", "ci_low", "ci_high"]:
        show[c] = show[c].round(4)
    show["p_value"] = show["p_value"].round(4)
    show["p_holm"] = show["p_holm"].round(4)
    print(show.to_string(index=False))
    print("\np_holm adjusts for the number of segments tested; segment results are "
          "exploratory and have low power.")
    metrics["segments"] = seg.to_dict(orient="records")

    # Charts and report
    FIG_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")
    plot_conversion(counts, FIG_DIR / "analysis_conversion_ci.png")
    plot_segments(seg, diff, FIG_DIR / "analysis_segment_differences.png")
    with open(REPORT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved 2 charts to {FIG_DIR}")
    print(f"Saved metrics to {REPORT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
