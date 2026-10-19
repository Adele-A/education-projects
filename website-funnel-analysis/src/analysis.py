"""Main analysis for the Website Funnel Analysis project (SYNTHETIC data).

Steps:
  1. Drop-off analysis: overall funnel and conditional step rates by segment.
  2. Statistical tests: chi-square per segment and two-proportion z-tests.
  3. Logistic regression of purchase on source, device and user type
     (stratified 80/20 train/test split, random_state=42).

Saves two charts to figures/ and a metrics summary to reports/metrics.json.
Run from the project root:  python src/analysis.py
(Run python src/data_generator.py first to create the CSV files.)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: files only
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from utils import (
    baseline_metrics,
    build_design_matrix,
    chi_square_test,
    classification_metrics,
    compare_groups,
    segment_step_conversion,
    step_transitions,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
FIG_DIR = PROJECT_DIR / "figures"
REPORT_DIR = PROJECT_DIR / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.2
SEGMENTS = ["traffic_source", "device", "user_type", "country"]


def load_sessions() -> pd.DataFrame:
    """Load the sessions table."""
    path = DATA_DIR / "sessions.csv"
    if not path.exists():
        raise FileNotFoundError("data/sessions.csv not found. Run: python src/data_generator.py")
    return pd.read_csv(path, parse_dates=["session_date"])


def _json_default(obj):
    """Make numpy types JSON serialisable."""
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Not serialisable: {type(obj)}")


# --- 1. Drop-off analysis ----------------------------------------------------

def run_dropoff(sessions: pd.DataFrame) -> dict:
    """Print and return the overall funnel and the biggest drop-off points."""
    trans = step_transitions(sessions)
    print("\n=== 1. Overall funnel transitions ===")
    print(trans.round(3).to_string(index=False))

    worst_rate = trans.loc[trans["drop_off"].idxmax()]
    worst_abs = trans.loc[trans["lost"].idxmax()]
    print(f"\nHighest drop-off rate : {worst_rate['transition']} "
          f"({worst_rate['drop_off']:.1%} of entering sessions lost)")
    print(f"Most sessions lost    : {worst_abs['transition']} "
          f"({int(worst_abs['lost'])} sessions)")

    for col in ["traffic_source", "device", "user_type"]:
        print(f"\nConditional step conversion by {col}:")
        print(segment_step_conversion(sessions, col).round(3).to_string())

    return {
        "transitions": trans.to_dict(orient="records"),
        "highest_drop_off_rate_transition": worst_rate["transition"],
        "most_sessions_lost_transition": worst_abs["transition"],
    }


def plot_source_dropoff(sessions: pd.DataFrame) -> None:
    """Heatmap of conditional step conversion by traffic source."""
    seg = segment_step_conversion(sessions, "traffic_source")
    cols = ["session_to_view", "view_to_cart", "cart_to_checkout", "checkout_to_purchase"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(seg[cols], annot=True, fmt=".1%", cmap="YlGnBu", cbar_kws={"label": "Step conversion"}, ax=ax)
    ax.set_title("Step conversion by traffic source (synthetic data)")
    ax.set_xlabel("Funnel transition")
    ax.set_ylabel("Traffic source")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "step_conversion_by_source.png", dpi=120)
    plt.close(fig)


# --- 2. Statistical tests ----------------------------------------------------

def run_tests(sessions: pd.DataFrame) -> dict:
    """Chi-square tests per segment and selected two-proportion tests."""
    print("\n=== 2. Statistical tests on purchase (purchased = 1) ===")
    chi_rows = [chi_square_test(sessions, seg) for seg in SEGMENTS]
    print("\nChi-square tests of independence:")
    print(pd.DataFrame(chi_rows).round(4).to_string(index=False))

    pairs = [
        ("device", "mobile", "desktop"),
        ("user_type", "returning", "new"),
        ("traffic_source", "email", "organic_search"),
        ("traffic_source", "social", "organic_search"),
        ("traffic_source", "paid_search", "organic_search"),
    ]
    prop_rows = [compare_groups(sessions, c, a, b) for c, a, b in pairs]
    print("\nTwo-proportion z-tests (difference = group_1 - group_2 purchase rate):")
    show = ["column", "group_1", "group_2", "rate_1", "rate_2", "difference",
            "ci_low", "ci_high", "p_value"]
    print(pd.DataFrame(prop_rows)[show].round(4).to_string(index=False))
    print("\nNote: several tests are run; p-values are not adjusted for multiple comparisons.")
    return {"chi_square": chi_rows, "two_proportion": prop_rows}


# --- 3. Logistic regression --------------------------------------------------

def run_model(sessions: pd.DataFrame) -> dict:
    """Fit and evaluate a logistic regression of purchase probability."""
    print("\n=== 3. Logistic regression: purchase ~ source + device + user type ===")
    X = build_design_matrix(sessions)
    y = sessions["purchased"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train rows: {len(X_train)}, test rows: {len(X_test)}")
    print(f"Purchase rate train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")

    # Predictive model (sklearn) evaluated on the held-out test set
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]

    # Threshold 0.5 is unhelpful for a rare outcome, so also use the train purchase rate
    threshold = float(y_train.mean())
    metrics_default = classification_metrics(y_test, prob, threshold=0.5)
    metrics_prev = classification_metrics(y_test, prob, threshold=threshold)
    base = baseline_metrics(y_train, y_test)

    print("\nTest metrics (threshold 0.5):")
    print(pd.Series(metrics_default).round(4).to_string())
    print(f"\nTest metrics (threshold = train purchase rate {threshold:.3f}):")
    print(pd.Series(metrics_prev).round(4).to_string())
    print("\nBaseline (constant prediction = train purchase rate):")
    print(pd.Series(base).round(4).to_string())

    # Interpretation model (statsmodels, same training data): odds ratios with 95% CI
    sm_model = sm.Logit(y_train, sm.add_constant(X_train.astype(float))).fit(disp=0)
    ci = sm_model.conf_int()
    odds = pd.DataFrame(
        {
            "odds_ratio": np.exp(sm_model.params),
            "ci_low": np.exp(ci[0]),
            "ci_high": np.exp(ci[1]),
            "p_value": sm_model.pvalues,
        }
    ).drop(index="const")
    print("\nOdds ratios (training data; reference: organic_search, desktop, new):")
    print(odds.round(3).to_string())

    plot_odds_ratios(odds)
    return {
        "split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE,
                  "n_train": len(X_train), "n_test": len(X_test)},
        "test_metrics_threshold_0_5": metrics_default,
        "test_metrics_threshold_train_rate": metrics_prev,
        "baseline": base,
        "odds_ratios": odds.reset_index(names="feature").to_dict(orient="records"),
    }


def plot_odds_ratios(odds: pd.DataFrame) -> None:
    """Forest plot of odds ratios with 95% confidence intervals (log scale)."""
    data = odds.sort_values("odds_ratio")
    y_pos = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        data["odds_ratio"], y_pos,
        xerr=[data["odds_ratio"] - data["ci_low"], data["ci_high"] - data["odds_ratio"]],
        fmt="o", color="steelblue", ecolor="gray", capsize=3,
    )
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(data.index)
    ax.set_title("Odds ratios for purchase, 95% CI (synthetic data)")
    ax.set_xlabel("Odds ratio vs reference level (log scale)")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "odds_ratios.png", dpi=120)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    sns.set_theme(style="whitegrid")

    sessions = load_sessions()
    print(f"Loaded {len(sessions)} synthetic sessions.")

    results = {"n_sessions": len(sessions)}
    results["dropoff"] = run_dropoff(sessions)
    plot_source_dropoff(sessions)
    results["tests"] = run_tests(sessions)
    results["model"] = run_model(sessions)

    out_path = REPORT_DIR / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"\nSaved charts to {FIG_DIR}")
    print(f"Saved metrics summary to {out_path}")


if __name__ == "__main__":
    main()
