"""Main analysis for the SYNTHETIC gym membership dataset.

Part 1: churn risk model (logistic regression) with a risk score per member.
Part 2: statistical analysis of the drivers of plan upgrades.

Outputs:
    figures/churn_model_performance.png
    figures/churn_model_coefficients.png
    reports/metrics.json
    reports/at_risk_members.csv

Run from the project root:  python src/analysis.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: figures are only saved to files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
FIG_DIR = PROJECT_DIR / "figures"
REPORT_DIR = PROJECT_DIR / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.25
TOP_N_AT_RISK = 50

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)


def section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Part 1: churn model
# ---------------------------------------------------------------------------
def churn_analysis(members):
    """Train and evaluate the churn model; return results for reporting and charts."""
    section("Part 1: churn risk model (logistic regression)")
    X, y = utils.prepare_churn_features(members)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train rows: {len(X_train)}, test rows: {len(X_test)} (stratified, random_state={RANDOM_STATE})")
    print(f"Churn rate in train: {y_train.mean():.3f}, in test: {y_test.mean():.3f}")

    model = utils.build_churn_model(RANDOM_STATE)

    # Cross-validation on the training set only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"\nCross-validated ROC AUC (5-fold, train): mean={cv_auc.mean():.3f}, std={cv_auc.std():.3f}")

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = utils.classification_metrics(y_test, proba)

    # Simple baseline: fewer visits per week -> higher churn risk
    baseline_auc = float(roc_auc_score(y_test, -X_test["avg_visits_per_week"]))

    print("\nTest-set metrics (threshold 0.5):")
    for key, value in metrics.items():
        print(f"  {key:>18}: {value:.3f}")
    print(f"  {'baseline_visits_auc':>18}: {baseline_auc:.3f}  (single feature: avg_visits_per_week)")

    coefs = utils.model_coefficients(model)
    print("\nModel coefficients (numeric features standardised; categoricals vs reference level):")
    print(coefs.round(3).to_string(index=False))

    deciles = utils.risk_decile_table(y_test, proba)
    print("\nTest set: observed churn rate by predicted risk decile (1 = lowest risk):")
    print(deciles.round(3).to_string(index=False))

    # Risk score for every ACTIVE member (the retention team's target group)
    active = members[members[utils.TARGET] == 0].copy()
    active["churn_risk"] = model.predict_proba(active[X.columns])[:, 1]
    active["risk_tier"] = utils.assign_risk_tier(active["churn_risk"])
    tier_counts = active["risk_tier"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).astype(int)
    print(f"\nRisk tiers for {len(active)} active members "
          f"(Medium >= {utils.MEDIUM_RISK_THRESHOLD}, High >= {utils.HIGH_RISK_THRESHOLD}):")
    print(tier_counts.to_string())

    top = (
        active.sort_values("churn_risk", ascending=False)
        .head(TOP_N_AT_RISK)[[
            "member_id", "initial_plan", "contract_type", "avg_visits_per_week",
            "days_since_last_visit", "distance_to_gym_km", "churn_risk", "risk_tier",
        ]]
        .round({"churn_risk": 4})
    )
    REPORT_DIR.mkdir(exist_ok=True)
    top.to_csv(REPORT_DIR / "at_risk_members.csv", index=False)
    print(f"\nSaved the {TOP_N_AT_RISK} highest-risk active members to reports/at_risk_members.csv")

    return {
        "y_test": y_test, "proba": proba, "coefs": coefs, "deciles": deciles,
        "metrics": metrics, "cv_auc": cv_auc, "baseline_auc": baseline_auc,
        "tier_counts": tier_counts, "n_train": len(X_train), "n_test": len(X_test),
        "n_active": len(active),
    }


def churn_charts(res):
    """Save the model performance chart and the coefficient chart."""
    FIG_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Chart 1: ROC curve and calibration by risk decile
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    fpr, tpr, _ = roc_curve(res["y_test"], res["proba"])
    axes[0].plot(fpr, tpr, color="tab:blue", label=f"Logistic regression (AUC = {res['metrics']['roc_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random guess")
    axes[0].set_title("ROC curve (test set)")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].legend(loc="lower right")

    d = res["deciles"]
    axes[1].plot(d["bin"], d["mean_predicted"], marker="o", color="tab:orange", label="Mean predicted risk")
    axes[1].plot(d["bin"], d["observed_rate"], marker="s", color="tab:red", label="Observed churn rate")
    axes[1].set_title("Predicted vs observed churn by risk decile (test set)")
    axes[1].set_xlabel("Risk decile (1 = lowest predicted risk)")
    axes[1].set_ylabel("Churn probability / rate")
    axes[1].set_xticks(d["bin"])
    axes[1].legend(loc="upper left")
    fig.suptitle("Churn risk model performance (synthetic data)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "churn_model_performance.png", dpi=120)
    plt.close(fig)

    # Chart 2: coefficients (log-odds); positive = higher churn risk
    coefs = res["coefs"].sort_values("coefficient")
    colors = np.where(coefs["coefficient"] > 0, "tab:red", "tab:green")
    # Taller figure when there are more features, so bars and labels do not overlap
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.5 * len(coefs) + 1.5)))
    bars = ax.barh(coefs["feature"], coefs["coefficient"], color=colors)
    ax.bar_label(bars, labels=[f"{v:+.2f}" for v in coefs["coefficient"]], padding=4, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.margins(x=0.15)  # room for the value labels
    ax.set_title("Churn model coefficients (synthetic data)\nred = higher churn risk, green = lower")
    ax.set_xlabel("Coefficient (log-odds; numeric features per 1 SD, categoricals vs reference level)")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "churn_model_coefficients.png", dpi=120)
    plt.close(fig)
    print(f"\nSaved 2 charts to {FIG_DIR}")


# ---------------------------------------------------------------------------
# Part 2: upgrade drivers
# ---------------------------------------------------------------------------
def upgrade_analysis(members, upgrades):
    """Statistical tests and a logistic regression on plan upgrades."""
    section("Part 2: drivers of plan upgrades (eligible members only)")
    df = utils.add_upgrade_flag(members, upgrades)
    elig = df[utils.upgrade_eligible(df)].copy()
    n_elig = len(elig)
    overall = float(elig["upgraded"].mean())
    print(f"Eligible members: {n_elig}; upgrade rate: {overall:.3f}")

    # Mann-Whitney U test: visits of upgraders vs non-upgraders
    up_v = elig.loc[elig["upgraded"] == 1, "avg_visits_per_week"]
    no_v = elig.loc[elig["upgraded"] == 0, "avg_visits_per_week"]
    mw = stats.mannwhitneyu(up_v, no_v, alternative="two-sided")
    print(f"\nMean visits/week: upgraded={up_v.mean():.2f}, not upgraded={no_v.mean():.2f}")
    print(f"Mann-Whitney U test on visits: U={mw.statistic:.0f}, p={mw.pvalue:.4g}")

    # Chi-square tests for categorical / binary drivers
    chi_results = {}
    print("\nChi-square tests of independence with the upgrade flag:")
    for col in ["has_personal_trainer", "uses_group_classes", "contract_type", "initial_plan"]:
        table = pd.crosstab(elig[col], elig["upgraded"])
        chi2, p, dof, _ = stats.chi2_contingency(table)
        rates = elig.groupby(col)["upgraded"].mean().round(3).to_dict()
        chi_results[col] = {"chi2": float(chi2), "p_value": float(p), "dof": int(dof)}
        print(f"  {col:>22}: chi2={chi2:.2f}, dof={dof}, p={p:.4g}; upgrade rates={rates}")

    # Multivariable logistic regression (statsmodels) for odds ratios
    formula = ("upgraded ~ avg_visits_per_week + C(initial_plan) + has_personal_trainer"
               " + uses_group_classes + C(contract_type)")
    fit = smf.logit(formula, data=elig).fit(disp=False)
    ci = fit.conf_int()
    table = pd.DataFrame({
        "odds_ratio": np.exp(fit.params),
        "ci_low": np.exp(ci[0]),
        "ci_high": np.exp(ci[1]),
        "p_value": fit.pvalues,
    }).drop(index="Intercept")
    print("\nLogistic regression for upgrade (odds ratios, 95% CI):")
    print(table.round(3).to_string())

    return {
        "eligible_members": int(n_elig),
        "upgrade_rate": overall,
        "mean_visits_upgraded": float(up_v.mean()),
        "mean_visits_not_upgraded": float(no_v.mean()),
        "mann_whitney_visits": {"u_statistic": float(mw.statistic), "p_value": float(mw.pvalue)},
        "chi_square": chi_results,
        "logit_odds_ratios": {
            name: {k: float(v) for k, v in row.items()} for name, row in table.iterrows()
        },
    }


def save_metrics(churn_res, upgrade_res):
    """Write the metrics summary to reports/metrics.json."""
    REPORT_DIR.mkdir(exist_ok=True)
    summary = {
        "data_note": "Synthetic data generated by src/data_generator.py (seed 42)",
        "random_state": RANDOM_STATE,
        "churn_model": {
            "model": "LogisticRegression with standardised numeric and one-hot categorical features",
            "n_train": churn_res["n_train"],
            "n_test": churn_res["n_test"],
            "cv_roc_auc_mean": float(churn_res["cv_auc"].mean()),
            "cv_roc_auc_std": float(churn_res["cv_auc"].std()),
            "test_metrics": churn_res["metrics"],
            "baseline_visits_only_roc_auc": churn_res["baseline_auc"],
            "active_members_scored": churn_res["n_active"],
            "risk_tier_counts": {k: int(v) for k, v in churn_res["tier_counts"].items()},
            "coefficients": {
                r.feature: {"coefficient": float(r.coefficient), "odds_ratio": float(r.odds_ratio)}
                for r in churn_res["coefs"].itertuples()
            },
        },
        "upgrade_analysis": upgrade_res,
    }
    with open(REPORT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metrics summary to {REPORT_DIR / 'metrics.json'}")


def main():
    members, upgrades = utils.load_data(DATA_DIR)
    churn_res = churn_analysis(members)
    churn_charts(churn_res)
    upgrade_res = upgrade_analysis(members, upgrades)
    save_metrics(churn_res, upgrade_res)


if __name__ == "__main__":
    main()
