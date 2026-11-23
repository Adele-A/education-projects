"""Main analysis: Isolation Forest vs a supervised baseline on SYNTHETIC data.

Run from the project root:

    python src/analysis.py

Steps:
    1. Load data and build features (src/utils.py).
    2. Stratified train/test split (random_state=42).
    3. Isolation Forest (unsupervised: fitted without labels).
    4. Logistic regression (supervised, class_weight="balanced").
    5. Compare on the test set: PR-AUC, ROC-AUC, precision/recall at a fixed alert budget.

Output:
    figures/pr_curves.png
    figures/logistic_coefficients.png
    reports/metrics.json
"""

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import build_features, evaluate_scores, load_data  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_DIR / "figures"
REPORT_DIR = PROJECT_DIR / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.3
ALERT_BUDGET = 0.02  # review team can check 2% of transactions


def print_metrics(name, m):
    """Print one model's metrics with clear labels."""
    print("\n%s" % name)
    print("  PR-AUC (average precision): %.4f" % m["pr_auc"])
    print("  ROC-AUC:                    %.4f" % m["roc_auc"])
    print("  At alert budget of %.0f%% (%d flagged transactions):"
          % (m["alert_budget"] * 100, m["flagged"]))
    print("    true positives: %d" % m["true_positives"])
    print("    precision:      %.4f" % m["precision"])
    print("    recall:         %.4f" % m["recall"])
    print("    F1:             %.4f" % m["f1"])


def plot_pr_curves(y_test, scores_by_model):
    """Chart 1: precision-recall curves, with the base rate as a reference."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (scores, pr_auc) in scores_by_model.items():
        prec, rec, _ = precision_recall_curve(y_test, scores)
        ax.plot(rec, prec, label="%s (PR-AUC = %.3f)" % (name, pr_auc))
    ax.axhline(y_test.mean(), color="grey", linestyle="--", label="Base rate (random)")
    ax.set_title("Precision-recall curves on the test set (synthetic data)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pr_curves.png", dpi=120)
    plt.close(fig)


def plot_coefficients(model, feature_names, top_n=12):
    """Chart 2: largest standardised logistic regression coefficients."""
    coefs = pd.Series(model[-1].coef_[0], index=feature_names)
    top = coefs.reindex(coefs.abs().sort_values(ascending=False).head(top_n).index)
    top = top.sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["indianred" if v > 0 else "steelblue" for v in top.values]
    ax.barh(top.index, top.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Logistic regression: top %d coefficients (standardised features)" % top_n)
    ax.set_xlabel("Coefficient (positive = higher fraud probability)")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "logistic_coefficients.png", dpi=120)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    customers, transactions = load_data()
    X, y = build_features(transactions, customers)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print("Train: %d rows (%d fraud) | Test: %d rows (%d fraud)"
          % (len(X_train), y_train.sum(), len(X_test), y_test.sum()))
    print("Number of features: %d" % X.shape[1])

    # Unsupervised: labels are NOT used for fitting. Higher score = more anomalous.
    iso = IsolationForest(n_estimators=200, contamination="auto",
                          random_state=RANDOM_STATE)
    iso.fit(X_train)
    iso_scores = -iso.score_samples(X_test)

    # Supervised baseline with class weights to handle the imbalance
    logit = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000,
                           random_state=RANDOM_STATE),
    )
    logit.fit(X_train, y_train)
    logit_scores = logit.predict_proba(X_test)[:, 1]

    iso_metrics = evaluate_scores(y_test, iso_scores, ALERT_BUDGET)
    logit_metrics = evaluate_scores(y_test, logit_scores, ALERT_BUDGET)

    print("\n" + "=" * 60)
    print("TEST SET RESULTS (synthetic data)")
    print("=" * 60)
    print("Base rate of fraud in test set: %.4f" % y_test.mean())
    print_metrics("Isolation Forest (unsupervised)", iso_metrics)
    print_metrics("Logistic regression (supervised)", logit_metrics)

    plot_pr_curves(y_test, {
        "Isolation Forest": (iso_scores, iso_metrics["pr_auc"]),
        "Logistic regression": (logit_scores, logit_metrics["pr_auc"]),
    })
    plot_coefficients(logit, X.columns)

    summary = {
        "data": "synthetic",
        "config": {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "alert_budget": ALERT_BUDGET,
            "n_features": int(X.shape[1]),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "fraud_in_test": int(y_test.sum()),
        },
        "isolation_forest": iso_metrics,
        "logistic_regression": logit_metrics,
    }
    with open(REPORT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nCharts saved to:", FIG_DIR)
    print("Metrics saved to:", REPORT_DIR / "metrics.json")


if __name__ == "__main__":
    main()
