"""Default prediction with class imbalance handling and ROC analysis (SYNTHETIC data).

Steps:
    1. Load and join the data, stratified 75/25 train/test split (random_state=42).
    2. Compare three models by 5-fold cross-validated ROC AUC on the training set:
       plain logistic regression, logistic regression with balanced class weights,
       and a random forest with balanced class weights.
    3. Pick the best model by CV AUC, choose a cost-based decision threshold from
       out-of-fold training predictions (no test data used), and evaluate on the test set.
    4. Save charts to figures/ and metrics to reports/metrics.json.

Run from the project root:  python src/analysis.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (best_threshold, build_pipeline, compute_metrics,  # noqa: E402
                   cost_curve, load_model_data, prepare_features)

PROJECT_DIR = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_DIR / "figures"
REPORT_DIR = PROJECT_DIR / "reports"

SEED = 42
# Business assumption (not estimated from data): a missed default costs 5 times
# as much as wrongly rejecting a good borrower.
COST_FN = 5.0
COST_FP = 1.0


def make_models() -> dict:
    """Candidate models, each wrapped in the shared preprocessing pipeline."""
    return {
        "Logistic regression (no weights)": build_pipeline(
            LogisticRegression(max_iter=1000)),
        "Logistic regression (balanced)": build_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced")),
        "Random forest (balanced)": build_pipeline(
            RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                   class_weight="balanced_subsample",
                                   random_state=SEED, n_jobs=1)),
    }


def plot_roc(roc_data: dict, path: Path) -> None:
    """Save ROC curves of all models on the test set."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, (fpr, tpr, auc) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on the test set (synthetic data)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_cost(curve, threshold: float, model_name: str, path: Path) -> None:
    """Save the total-cost-vs-threshold chart (out-of-fold training predictions)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(curve["threshold"], curve["cost"], color="steelblue")
    ax.axvline(threshold, color="red", linestyle="--",
               label=f"Chosen threshold = {threshold:.2f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel(f"Total cost (FN x {COST_FN:g} + FP x {COST_FP:g})")
    ax.set_title(f"Cost vs threshold, {model_name}\n(out-of-fold training data, synthetic)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    """Run the full modelling workflow."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    data = load_model_data()
    X, y = prepare_features(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED)

    print("=== Data split ===")
    print(f"Train: {len(X_train)} rows, default rate {y_train.mean():.4f}")
    print(f"Test:  {len(X_test)} rows, default rate {y_test.mean():.4f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models = make_models()

    # --- Model comparison: out-of-fold predictions on the training set ---
    print("\n=== Model comparison (5-fold CV on train) ===")
    oof_scores, cv_auc = {}, {}
    for name, model in models.items():
        oof = cross_val_predict(model, X_train, y_train, cv=cv,
                                method="predict_proba")[:, 1]
        oof_scores[name] = oof
        cv_auc[name] = compute_metrics(y_train, oof)["roc_auc"]
        print(f"{name}: CV ROC AUC = {cv_auc[name]:.4f}")

    best_name = max(cv_auc, key=cv_auc.get)
    print(f"\nSelected model (highest CV ROC AUC): {best_name}")

    # --- Fit all models on the full training set and score the test set ---
    test_scores, roc_data, test_metrics = {}, {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        test_scores[name] = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, test_scores[name])
        auc = compute_metrics(y_test, test_scores[name])["roc_auc"]
        roc_data[name] = (fpr, tpr, auc)

    # --- Cost-based threshold, chosen on out-of-fold TRAIN scores only ---
    # Note: balanced weights shift the score scale, so each model gets its own threshold.
    print(f"\n=== Test-set metrics (cost: FN = {COST_FN:g}, FP = {COST_FP:g}) ===")
    thresholds = {}
    for name in models:
        thr = best_threshold(y_train, oof_scores[name], COST_FN, COST_FP)
        thresholds[name] = thr
        default_m = compute_metrics(y_test, test_scores[name], 0.5, COST_FN, COST_FP)
        tuned_m = compute_metrics(y_test, test_scores[name], thr, COST_FN, COST_FP)
        test_metrics[name] = {"threshold_0.5": default_m, "tuned_threshold": tuned_m}
        print(f"\n{name}")
        print(f"  ROC AUC = {tuned_m['roc_auc']:.4f}, PR AUC = {tuned_m['pr_auc']:.4f}")
        for label, m in (("threshold 0.50", default_m),
                         (f"tuned threshold {thr:.2f}", tuned_m)):
            print(f"  [{label}] precision = {m['precision']:.3f}, "
                  f"recall = {m['recall']:.3f}, F1 = {m['f1']:.3f}, "
                  f"FN = {m['false_negatives']}, FP = {m['false_positives']}, "
                  f"total cost = {m['total_cost']:.1f}")

    # Reference: cost of approving every loan (predict no default for all)
    approve_all_cost = float(y_test.sum() * COST_FN)
    reject_all_cost = float((len(y_test) - y_test.sum()) * COST_FP)
    print(f"\nReference cost, approve all loans: {approve_all_cost:.1f}")
    print(f"Reference cost, reject all loans:  {reject_all_cost:.1f}")

    # --- Charts ---
    plot_roc(roc_data, FIG_DIR / "roc_curves.png")
    curve = cost_curve(y_train, oof_scores[best_name], COST_FN, COST_FP)
    plot_cost(curve, thresholds[best_name], best_name, FIG_DIR / "cost_vs_threshold.png")
    print(f"\nSaved 2 charts to {FIG_DIR.name}/")

    # --- Metrics summary ---
    summary = {
        "note": "All data is synthetic, generated by src/data_generator.py.",
        "random_state": SEED,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_default_rate": float(y_train.mean()),
        "test_default_rate": float(y_test.mean()),
        "cost_assumption": {"cost_false_negative": COST_FN, "cost_false_positive": COST_FP},
        "cv_roc_auc_train": cv_auc,
        "selected_model": best_name,
        "tuned_thresholds": thresholds,
        "test_metrics": test_metrics,
        "reference_cost_approve_all": approve_all_cost,
        "reference_cost_reject_all": reject_all_cost,
    }
    with open(REPORT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved metrics to {REPORT_DIR.name}/metrics.json")


if __name__ == "__main__":
    main()
