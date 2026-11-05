"""Main analysis: attrition drivers and a logistic regression model.

Steps:
    1. Load and join the synthetic tables.
    2. Univariate statistical tests (chi-square, Mann-Whitney U).
    3. Train/test split (stratified, random_state=42) and logistic regression.
    4. Evaluate on the hold-out set and with cross-validation on the train set.
    5. Interpret odds ratios (statsmodels) and save charts and metrics.

All data is SYNTHETIC. Every number printed or saved comes from the run.

Run from the project root (after generating the data):
    python src/data_generator.py
    python src/analysis.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score, train_test_split

# Make `import utils` work when running `python src/analysis.py` from the root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import utils  # noqa: E402

PROJECT_DIR = utils.PROJECT_DIR
FIGURES_DIR = PROJECT_DIR / "figures"
REPORTS_DIR = PROJECT_DIR / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.25


def print_section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def univariate_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Test each feature against attrition on its own.

    Numeric features: Mann-Whitney U (left vs stayed).
    Categorical features: chi-square test of independence.
    These are exploratory; p-values are not adjusted for multiple testing.
    """
    rows = []
    for col in utils.NUMERIC_FEATURES:
        left = df.loc[df[utils.TARGET] == 1, col].dropna()
        stayed = df.loc[df[utils.TARGET] == 0, col].dropna()
        _, p = stats.mannwhitneyu(left, stayed, alternative="two-sided")
        rows.append(
            {
                "feature": col,
                "test": "Mann-Whitney U",
                "mean_stayed": stayed.mean(),
                "mean_left": left.mean(),
                "p_value": p,
            }
        )
    for col in utils.CATEGORICAL_FEATURES:
        table = pd.crosstab(df[col], df[utils.TARGET])
        _, p, _, _ = stats.chi2_contingency(table)
        rows.append(
            {
                "feature": col,
                "test": "Chi-square",
                "mean_stayed": np.nan,
                "mean_left": np.nan,
                "p_value": p,
            }
        )
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


def plot_roc(y_test, y_prob, auc: float) -> None:
    """ROC curve on the hold-out set."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color="steelblue", label=f"Logistic regression (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random guess")
    ax.set_title("ROC curve (hold-out test set)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curve.png", dpi=120)
    plt.close(fig)


def plot_odds_ratios(table: pd.DataFrame) -> None:
    """Forest plot of odds ratios with 95% confidence intervals (log scale)."""
    data = table.sort_values("odds_ratio").reset_index(drop=True)
    significant = data["p_value"] < 0.05
    colors = np.where(significant, "indianred", "grey")
    y = np.arange(len(data))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.hlines(y, data["ci_low"], data["ci_high"], color=colors, linewidth=2)
    ax.scatter(data["odds_ratio"], y, color=colors, zorder=3)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(data["feature"])
    ax.set_xlabel("Odds ratio (log scale); red = p < 0.05")
    ax.set_title("Logistic regression odds ratios with 95% CI\n"
                 "(numeric features: per +1 standard deviation)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "odds_ratios.png", dpi=120)
    plt.close(fig)


def main() -> None:
    """Run the full analysis and save charts and metrics."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Load data
    employees, surveys = utils.load_data()
    df = utils.merge_data(employees, surveys)
    print_section("Data")
    print(f"Rows: {len(df)}; employees without survey: "
          f"{int(df['job_satisfaction'].isna().sum())}")
    print(f"Overall attrition rate: {df[utils.TARGET].mean():.3f}")

    # 2. Univariate tests
    print_section("Univariate tests vs attrition (sorted by p-value)")
    tests = univariate_tests(df)
    print(tests.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # 3. Split and fit
    X, y = utils.split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    model = utils.build_model()
    model.fit(X_train, y_train)

    print_section("Train / test split")
    print(f"Train rows: {len(X_train)}, test rows: {len(X_test)}")
    print(f"Attrition rate train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")

    # 4. Evaluation
    cv_auc = cross_val_score(
        utils.build_model(), X_train, y_train, cv=5, scoring="roc_auc"
    )
    prob_train = model.predict_proba(X_train)[:, 1]
    prob_test = model.predict_proba(X_test)[:, 1]

    # F1-optimal threshold is chosen on the TRAIN set only, then applied to test
    tuned_threshold = utils.best_f1_threshold(y_train, prob_train)
    metrics_default = utils.compute_metrics(y_test, prob_test, threshold=0.5)
    metrics_tuned = utils.compute_metrics(y_test, prob_test, threshold=tuned_threshold)

    print_section("Model evaluation")
    print(f"Cross-validated ROC AUC (5-fold, train): "
          f"{cv_auc.mean():.3f} +/- {cv_auc.std():.3f}")
    print(f"Majority-class baseline accuracy (test): {1 - y_test.mean():.3f}")
    for label, m in [("threshold 0.5", metrics_default),
                     (f"tuned threshold {tuned_threshold:.3f}", metrics_tuned)]:
        print(f"\nHold-out test metrics at {label}:")
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]:
            print(f"  {key:<18}{m[key]:.3f}")
        print(f"  confusion matrix  TN={m['true_negatives']} FP={m['false_positives']} "
              f"FN={m['false_negatives']} TP={m['true_positives']}")

    # 5. Interpretation
    odds = utils.odds_ratio_table(model, X_train, y_train)
    print_section("Odds ratios (statsmodels Logit on train set)")
    print("Numeric features: per +1 standard deviation; categorical: vs reference category.")
    shown = odds.assign(abs_coef=odds["coef"].abs()).sort_values("abs_coef", ascending=False)
    print(shown.drop(columns="abs_coef").to_string(
        index=False, float_format=lambda v: f"{v:.4g}"))

    plot_roc(y_test, prob_test, metrics_default["roc_auc"])
    plot_odds_ratios(odds)

    # 6. Save metrics summary
    summary = {
        "data": {
            "n_rows": int(len(df)),
            "attrition_rate": float(df[utils.TARGET].mean()),
            "synthetic": True,
        },
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        },
        "cross_validation_roc_auc": {
            "mean": float(cv_auc.mean()),
            "std": float(cv_auc.std()),
        },
        "baseline_majority_accuracy_test": float(1 - y_test.mean()),
        "test_metrics_threshold_0.5": metrics_default,
        "test_metrics_tuned_threshold": metrics_tuned,
        "odds_ratios": [
            {
                "feature": r.feature,
                "odds_ratio": float(r.odds_ratio),
                "ci_low": float(r.ci_low),
                "ci_high": float(r.ci_high),
                "p_value": float(r.p_value),
            }
            for r in shown.itertuples()
        ],
    }
    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print_section("Outputs saved")
    print(f"Charts: {FIGURES_DIR / 'roc_curve.png'}, {FIGURES_DIR / 'odds_ratios.png'}")
    print(f"Metrics: {REPORTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
