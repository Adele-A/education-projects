"""Main analysis: churn classification models and feature importance.

Trains a baseline logistic regression and a gradient boosting model on the
SYNTHETIC data in data/, evaluates them with cross-validation (training set)
and on a held-out test set, and computes permutation feature importance for
the model with the best cross-validated ROC AUC.

Outputs:
  * figures/roc_curves.png
  * figures/feature_importance.png
  * reports/metrics.json

Run from the project root:  python src/analysis.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

# Make `import utils` work when running `python src/analysis.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import build_features, build_preprocessor, compute_metrics  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "figures"
REPORTS_DIR = PROJECT_DIR / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

sns.set_theme(style="whitegrid")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both tables, failing with a clear message if data is missing."""
    cust_path = DATA_DIR / "customers.csv"
    tick_path = DATA_DIR / "support_tickets.csv"
    if not cust_path.exists() or not tick_path.exists():
        raise FileNotFoundError(
            "Data files not found. Run `python src/data_generator.py` first."
        )
    return pd.read_csv(cust_path), pd.read_csv(tick_path)


def make_models(X: pd.DataFrame) -> dict[str, Pipeline]:
    """Build the two model pipelines (preprocessing + classifier)."""
    return {
        "Logistic regression": Pipeline([
            ("prep", build_preprocessor(X)),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Gradient boosting": Pipeline([
            ("prep", build_preprocessor(X)),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=200, max_depth=3,
                random_state=RANDOM_STATE)),
        ]),
    }


def plot_roc(models: dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Save ROC curves of all fitted models on the test set."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random guess")
    ax.set_title("ROC curves on the test set (synthetic data)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=120)
    plt.close(fig)


def plot_importance(importance: pd.DataFrame, model_name: str, top_n: int = 12) -> None:
    """Save a bar chart of permutation importance (drop in ROC AUC)."""
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"],
            color="steelblue")
    ax.set_title(f"Permutation importance: {model_name}\n(synthetic data, test set)")
    ax.set_xlabel("Mean drop in ROC AUC when the feature is shuffled")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=120)
    plt.close(fig)


def main() -> None:
    """Run the full modelling workflow."""
    customers, tickets = load_data()
    X, y = build_features(customers, tickets)
    print(f"Feature matrix: {X.shape[0]} rows x {X.shape[1]} columns")
    print(f"Overall churn rate: {y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    print(f"Train rows: {len(X_train)}, test rows: {len(X_test)}")

    models = make_models(X)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results: dict[str, dict] = {}
    for name, model in models.items():
        # Cross-validated ROC AUC on the training set only
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        model.fit(X_train, y_train)
        test_metrics = compute_metrics(y_test, model.predict_proba(X_test)[:, 1])
        results[name] = {
            "cv_roc_auc_mean": float(scores.mean()),
            "cv_roc_auc_std": float(scores.std()),
            "test": test_metrics,
        }

        print(f"\n--- {name} ---")
        print(f"CV ROC AUC ({CV_FOLDS}-fold, train set): "
              f"{scores.mean():.3f} +/- {scores.std():.3f}")
        for key, value in test_metrics.items():
            print(f"Test {key}: {value:.3f}")

    # Select the model by cross-validated AUC (not by test performance)
    best_name = max(results, key=lambda k: results[k]["cv_roc_auc_mean"])
    print(f"\nBest model by CV ROC AUC: {best_name}")

    # Permutation importance on raw input features, computed on the test set
    perm = permutation_importance(
        models[best_name], X_test, y_test, scoring="roc_auc",
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=1)
    importance = (pd.DataFrame({
        "feature": X.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True))

    print(f"\nPermutation importance ({best_name}, drop in test ROC AUC):")
    print(importance.round(4).to_string(index=False))

    FIGURES_DIR.mkdir(exist_ok=True)
    plot_roc(models, X_test, y_test)
    plot_importance(importance, best_name)
    print(f"\nSaved 2 charts to {FIGURES_DIR}")

    # Metrics summary
    REPORTS_DIR.mkdir(exist_ok=True)
    summary = {
        "data_note": "All data is synthetic, generated by src/data_generator.py",
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "churn_rate": float(y.mean()),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "models": results,
        "best_model_by_cv": best_name,
        "permutation_importance": importance.round(5).to_dict(orient="records"),
    }
    out_path = REPORTS_DIR / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved metrics summary to {out_path}")


if __name__ == "__main__":
    main()
