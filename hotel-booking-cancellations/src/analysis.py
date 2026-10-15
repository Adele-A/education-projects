"""Cancellation model for the SYNTHETIC hotel bookings dataset.

Trains a logistic regression baseline and a gradient boosting classifier,
evaluates both on a held-out test set, computes permutation importance and a
short error analysis, saves charts to figures/ and metrics to reports/metrics.json.

Run from the project root (after python src/data_generator.py):
    python src/analysis.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are only saved to disk
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

import utils  # noqa: E402

DATA_DIR = PROJECT_DIR / "data"
FIG_DIR = PROJECT_DIR / "figures"
REPORT_DIR = PROJECT_DIR / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read both CSV files, with a clear message if they are missing."""
    for name in ("guests.csv", "bookings.csv"):
        if not (DATA_DIR / name).exists():
            raise FileNotFoundError(
                f"{DATA_DIR / name} not found. Run 'python src/data_generator.py' first."
            )
    return pd.read_csv(DATA_DIR / "guests.csv"), pd.read_csv(DATA_DIR / "bookings.csv")


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_metrics(name: str, m: dict) -> None:
    print(f"\n{name} (threshold {m['threshold']:.2f})")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc",
                "average_precision", "brier_score"):
        print(f"  {key:<18}{m[key]:.3f}")
    c = m["confusion_matrix"]
    print(f"  confusion matrix  TN={c['tn']}  FP={c['fp']}  FN={c['fn']}  TP={c['tp']}")


def plot_roc(y_test, probas: dict) -> None:
    """ROC curves of both models on the test set."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, p in probas.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_title("ROC curves on the test set")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_curves.png", dpi=120)
    plt.close(fig)


def plot_importance(imp: pd.Series) -> None:
    """Horizontal bar chart of permutation importance (drop in ROC AUC)."""
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(imp.index, imp.values, color="steelblue")
    ax.set_title("Permutation importance (gradient boosting, test set)")
    ax.set_xlabel("Mean drop in ROC AUC when the feature is shuffled")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png", dpi=120)
    plt.close(fig)


def main() -> None:
    guests, bookings = load_data()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = utils.build_dataset(guests, bookings)
    X, y = utils.split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    section("SETUP")
    print(f"Train rows: {len(X_train)}, test rows: {len(X_test)}")
    print(f"Cancellation rate: train {y_train.mean():.3f}, test {y_test.mean():.3f}")
    print(f"Features: {len(utils.NUMERIC_FEATURES)} numeric, "
          f"{len(utils.CATEGORICAL_FEATURES)} categorical")

    models = {
        "Logistic regression": utils.make_logistic_pipeline(),
        "Gradient boosting": utils.make_gb_pipeline(),
    }

    # --- Cross-validation on the training set only ---------------------------
    section("5-FOLD CROSS-VALIDATION (train set, ROC AUC)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    for name, pipe in models.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        cv_results[name] = {"mean": float(scores.mean()), "std": float(scores.std())}
        print(f"{name:<22}mean {scores.mean():.3f}  std {scores.std():.3f}")

    # --- Fit and evaluate on the test set ---------------------------------------
    section("TEST SET METRICS")
    probas, test_metrics, lifts = {}, {}, {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        probas[name] = pipe.predict_proba(X_test)[:, 1]
        test_metrics[name] = utils.classification_metrics(y_test, probas[name])
        print_metrics(name, test_metrics[name])
        lifts[name] = utils.top_fraction_lift(y_test, probas[name], 0.1)

    section("LIFT: highest-risk 10% of test bookings")
    for name, l in lifts.items():
        print(f"{name:<22}cancel rate in top 10%: {l['top_cancel_rate']:.3f} "
              f"vs base {l['base_cancel_rate']:.3f} (lift {l['lift']:.2f})")

    # --- Feature importance (gradient boosting) --------------------------------
    section("PERMUTATION IMPORTANCE (gradient boosting, drop in ROC AUC)")
    gb = models["Gradient boosting"]
    perm = permutation_importance(
        gb, X_test, y_test, scoring="roc_auc", n_repeats=10,
        random_state=RANDOM_STATE, n_jobs=1,
    )
    importance = pd.Series(perm.importances_mean, index=X_test.columns)
    print(importance.sort_values(ascending=False).round(4).to_string())

    # --- Error analysis ------------------------------------------------------------
    section("ERROR ANALYSIS (gradient boosting, test set)")
    test_df = df.loc[X_test.index].copy()
    test_df["lead_bucket"] = pd.cut(
        test_df["lead_time_days"], bins=utils.LEAD_BINS, labels=utils.LEAD_LABELS
    )
    gb_proba = probas["Gradient boosting"]
    for by in ("deposit_type", "lead_bucket"):
        print(f"\nBy {by}:")
        print(utils.error_rate_by_group(test_df, y_test, gb_proba, by).to_string())

    # --- Charts and report ------------------------------------------------------------
    plot_roc(y_test, probas)
    plot_importance(importance)

    summary = {
        "note": "All numbers come from SYNTHETIC data generated by this project.",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "cv_roc_auc": cv_results,
        "test_metrics": test_metrics,
        "top_decile_lift": lifts,
        "permutation_importance_roc_auc_drop": {
            k: float(v) for k, v in importance.sort_values(ascending=False).items()
        },
    }
    with open(REPORT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    section("OUTPUT FILES")
    print(Path("reports") / "metrics.json")
    print(Path("figures") / "roc_curves.png")
    print(Path("figures") / "feature_importance.png")


if __name__ == "__main__":
    main()
