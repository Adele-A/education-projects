"""Exploratory data analysis for the SYNTHETIC gym membership dataset.

Loads data/members.csv and data/upgrades.csv, prints an overview
(shape, dtypes, missing values, descriptive statistics), several
aggregations, and saves charts to figures/.

Run from the project root:  python src/eda.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: figures are only saved to files
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
FIG_DIR = PROJECT_DIR / "figures"

PLAN_ORDER = ["Basic", "Standard", "Premium"]

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)


def section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_data():
    """Load both tables and parse date columns."""
    members = pd.read_csv(DATA_DIR / "members.csv", parse_dates=["signup_date", "churn_date"])
    upgrades = pd.read_csv(DATA_DIR / "upgrades.csv", parse_dates=["upgrade_date"])
    return members, upgrades


def overview(members, upgrades):
    """Print shape, dtypes, missing values and descriptive statistics."""
    for name, df in [("members", members), ("upgrades", upgrades)]:
        section(f"Overview: {name}")
        print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        print("\nDtypes:")
        print(df.dtypes.to_string())
        missing = df.isna().sum()
        print("\nMissing values (columns with at least one):")
        print(missing[missing > 0].to_string() if (missing > 0).any() else "none")

    section("Descriptive statistics: numeric columns of members")
    num_cols = [
        "age", "monthly_fee", "distance_to_gym_km", "avg_visits_per_week",
        "days_since_last_visit", "tenure_months",
    ]
    print(members[num_cols].describe().round(2).to_string())

    section("Descriptive statistics: categorical columns of members")
    for col in ["gender", "city", "initial_plan", "current_plan", "contract_type", "payment_method"]:
        print(f"\n{col}:")
        print(members[col].value_counts(normalize=True).round(3).to_string())

    # Churn date is empty by design for active members; check that explicitly
    n_active = int((members["is_churned"] == 0).sum())
    print(f"\nActive members: {n_active}; missing churn_date: {int(members['churn_date'].isna().sum())}")


def aggregations(members, upgrades):
    """Print the key aggregations and return the ones used for charts."""
    members = members.copy()
    members["upgraded"] = members["member_id"].isin(upgrades["member_id"]).astype(int)

    # 1. Visit frequency by current plan
    section("Aggregation 1: visit frequency by current plan")
    by_plan = (
        members.groupby("current_plan")
        .agg(members=("member_id", "count"),
             mean_visits=("avg_visits_per_week", "mean"),
             median_visits=("avg_visits_per_week", "median"),
             churn_rate=("is_churned", "mean"))
        .reindex(PLAN_ORDER)
        .round(3)
    )
    print(by_plan.to_string())

    # 2. Churned vs active behaviour
    section("Aggregation 2: churned vs active members")
    by_status = (
        members.groupby("is_churned")
        .agg(members=("member_id", "count"),
             mean_visits=("avg_visits_per_week", "mean"),
             mean_days_since_last=("days_since_last_visit", "mean"),
             mean_distance_km=("distance_to_gym_km", "mean"))
        .round(2)
    )
    print(by_status.to_string())

    # 3. Churn rate by contract type and initial plan
    section("Aggregation 3: churn rate by contract type and initial plan")
    churn_seg = (
        members.pivot_table(index="initial_plan", columns="contract_type",
                            values="is_churned", aggfunc="mean")
        .reindex(PLAN_ORDER)
        .round(3)
    )
    print(churn_seg.to_string())

    # 4. Churn rate by visit-frequency band
    section("Aggregation 4: churn rate by visit-frequency band")
    members["visit_band"] = pd.cut(
        members["avg_visits_per_week"], bins=[-0.01, 1, 2, 3, 4, 7],
        labels=["0-1", "1-2", "2-3", "3-4", "4-7"],
    )
    by_band = (
        members.groupby("visit_band", observed=True)
        .agg(members=("member_id", "count"), churn_rate=("is_churned", "mean"))
        .round(3)
    )
    print(by_band.to_string())

    # 5. Upgrade rate among eligible plans (Premium at sign-up cannot upgrade)
    section("Aggregation 5: upgrade rate by initial plan and visit band")
    eligible = members[members["initial_plan"] != "Premium"]
    upg_plan = eligible.groupby("initial_plan")["upgraded"].agg(["count", "mean"]).round(3)
    upg_plan.columns = ["eligible_members", "upgrade_rate"]
    print(upg_plan.to_string())
    upg_band = (
        eligible.groupby("visit_band", observed=True)["upgraded"].mean().round(3)
        .rename("upgrade_rate")
    )
    print("\nUpgrade rate by visit band (Basic and Standard starters):")
    print(upg_band.to_string())

    return members, by_band, upg_band


def make_charts(members, by_band, upg_band):
    """Save charts to figures/."""
    FIG_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Chart 1: distribution of visits per week by churn status
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = members.assign(status=members["is_churned"].map({0: "Active", 1: "Churned"}))
    sns.histplot(data=plot_df, x="avg_visits_per_week", hue="status", bins=30,
                 stat="density", common_norm=False, element="step", ax=ax)
    ax.set_title("Average visits per week: active vs churned members (synthetic data)")
    ax.set_xlabel("Average visits per week (last 90 days)")
    ax.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "visits_by_churn_status.png", dpi=120)
    plt.close(fig)

    # Chart 2: visits per week by current plan
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=members, x="current_plan", y="avg_visits_per_week",
                order=PLAN_ORDER, ax=ax)
    ax.set_title("Visit frequency by current plan (synthetic data)")
    ax.set_xlabel("Current plan")
    ax.set_ylabel("Average visits per week")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "visits_by_plan.png", dpi=120)
    plt.close(fig)

    # Chart 3: churn rate and upgrade rate by visit band
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(by_band.index.astype(str), by_band["churn_rate"], color="tab:red")
    axes[0].set_title("Churn rate by visit band")
    axes[0].set_xlabel("Average visits per week")
    axes[0].set_ylabel("Churn rate")
    axes[1].bar(upg_band.index.astype(str), upg_band.values, color="tab:green")
    axes[1].set_title("Upgrade rate by visit band (Basic/Standard starters)")
    axes[1].set_xlabel("Average visits per week")
    axes[1].set_ylabel("Upgrade rate")
    fig.suptitle("Synthetic data")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "churn_upgrade_by_visit_band.png", dpi=120)
    plt.close(fig)

    # Chart 4: churn rate by contract type and initial plan
    fig, ax = plt.subplots(figsize=(8, 5))
    seg = members.groupby(["initial_plan", "contract_type"])["is_churned"].mean().reset_index()
    sns.barplot(data=seg, x="initial_plan", y="is_churned", hue="contract_type",
                order=PLAN_ORDER, ax=ax)
    ax.set_title("Churn rate by initial plan and contract type (synthetic data)")
    ax.set_xlabel("Initial plan")
    ax.set_ylabel("Churn rate")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "churn_by_plan_contract.png", dpi=120)
    plt.close(fig)

    print(f"\nSaved 4 charts to {FIG_DIR}")


def main():
    members, upgrades = load_data()
    overview(members, upgrades)
    members, by_band, upg_band = aggregations(members, upgrades)
    make_charts(members, by_band, upg_band)


if __name__ == "__main__":
    main()
