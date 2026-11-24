"""Exploratory data analysis for the SYNTHETIC customer survey NPS dataset.

Loads data/customers.csv and data/survey_responses.csv, prints a data
quality overview (shape, dtypes, missing values, duplicates), descriptive
statistics and a few aggregations, and saves charts to figures/.

Run from the project root:  python src/eda.py
(Run  python src/data_generator.py  first to create the CSV files.)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: no plot windows
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "figures"

CATEGORY_ORDER = ["Detractor", "Passive", "Promoter"]
CATEGORY_COLORS = {"Detractor": "#d9534f", "Passive": "#f0ad4e", "Promoter": "#5cb85c"}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both CSV files, failing with a clear message if they are missing."""
    customers_path = DATA_DIR / "customers.csv"
    responses_path = DATA_DIR / "survey_responses.csv"
    for path in (customers_path, responses_path):
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run: python src/data_generator.py")

    customers = pd.read_csv(customers_path, parse_dates=["signup_date"])
    responses = pd.read_csv(responses_path, parse_dates=["survey_date"])
    return customers, responses


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def overview(name: str, df: pd.DataFrame) -> None:
    """Print shape, dtypes, missing values and duplicates for one table."""
    section(f"Overview: {name}")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nData types:")
    print(df.dtypes.to_string())
    print("\nMissing values:")
    missing = pd.DataFrame(
        {"missing": df.isna().sum(), "share": (df.isna().mean() * 100).round(1)}
    )
    print(missing.to_string())
    print(f"\nFully duplicated rows: {df.duplicated().sum()}")


def descriptive_stats(customers: pd.DataFrame, responses: pd.DataFrame) -> None:
    """Print descriptive statistics for numeric and categorical columns."""
    section("Descriptive statistics")
    print("Customers (numeric):")
    print(customers[["tenure_months"]].describe().round(2).to_string())
    print("\nSurvey responses (numeric):")
    print(responses[["nps_score"]].describe().round(2).to_string())
    print("\nCustomers (categorical, share of customers in %):")
    for col in ["segment", "plan", "region", "acquisition_channel"]:
        shares = (customers[col].value_counts(normalize=True) * 100).round(1)
        print(f"\n{col}:\n{shares.to_string()}")


def integrity_checks(customers: pd.DataFrame, responses: pd.DataFrame) -> None:
    """Check keys, ranges and consistency between the two tables."""
    section("Data quality checks")
    print(f"customer_id unique in customers: {customers['customer_id'].is_unique}")
    print(f"response_id unique in responses: {responses['response_id'].is_unique}")
    print(f"customer_id unique in responses: {responses['customer_id'].is_unique}")
    orphans = (~responses["customer_id"].isin(customers["customer_id"])).sum()
    print(f"Responses with unknown customer_id: {orphans}")
    print(
        f"nps_score range: {responses['nps_score'].min()} to {responses['nps_score'].max()}"
    )
    print(
        f"survey_date range: {responses['survey_date'].min().date()} "
        f"to {responses['survey_date'].max().date()}"
    )

    # The category must match the score buckets used in the generator
    expected = pd.cut(
        responses["nps_score"], bins=[-1, 6, 8, 10], labels=CATEGORY_ORDER
    ).astype(str)
    mismatches = (expected != responses["nps_category"]).sum()
    print(f"Rows where nps_category does not match nps_score: {mismatches}")


def aggregations(customers: pd.DataFrame, responses: pd.DataFrame) -> pd.DataFrame:
    """Print the key aggregations and return the merged table."""
    merged = responses.merge(customers, on="customer_id", how="left")
    merged["has_comment"] = merged["comment"].notna()

    section("Aggregation 1: NPS category distribution (%)")
    dist = (merged["nps_category"].value_counts(normalize=True) * 100).round(1)
    print(dist.reindex(CATEGORY_ORDER).to_string())

    section("Aggregation 2: survey coverage and response rate by segment")
    surveyed = customers["segment"].value_counts().rename("customers")
    answered = merged["segment"].value_counts().rename("responses")
    coverage = pd.concat([surveyed, answered], axis=1)
    coverage["response_rate_%"] = (coverage["responses"] / coverage["customers"] * 100).round(1)
    print(coverage.to_string())

    section("Aggregation 3: mean score and category shares by segment and plan")
    for col in ["segment", "plan"]:
        table = (
            merged.groupby(col)
            .agg(
                responses=("nps_score", "size"),
                mean_score=("nps_score", "mean"),
                promoter_pct=("nps_category", lambda s: (s == "Promoter").mean() * 100),
                detractor_pct=("nps_category", lambda s: (s == "Detractor").mean() * 100),
            )
            .round(2)
        )
        print(f"\nBy {col}:")
        print(table.to_string())

    section("Aggregation 4: mean score by region and acquisition channel")
    for col in ["region", "acquisition_channel"]:
        table = merged.groupby(col)["nps_score"].agg(["size", "mean"]).round(2)
        print(f"\nBy {col}:")
        print(table.to_string())

    section("Aggregation 5: comment coverage by NPS category")
    comment_cov = (
        merged.groupby("nps_category")["has_comment"].mean().mul(100).round(1)
    )
    print((comment_cov.reindex(CATEGORY_ORDER)).rename("comment_share_%").to_string())
    print(f"\nOverall share of responses with a comment: {merged['has_comment'].mean() * 100:.1f}%")
    return merged


def plot_score_distribution(merged: pd.DataFrame) -> None:
    """Bar chart of the 0-10 score counts, colored by NPS category."""
    counts = merged["nps_score"].value_counts().reindex(range(11), fill_value=0)
    colors = [CATEGORY_COLORS[c] for c in pd.cut(
        counts.index, bins=[-1, 6, 8, 10], labels=CATEGORY_ORDER
    ).astype(str)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_xticks(range(11))
    ax.set_xlabel("Recommendation score (0-10)")
    ax.set_ylabel("Number of responses")
    ax.set_title("Distribution of NPS scores (synthetic data)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLORS[c]) for c in CATEGORY_ORDER]
    ax.legend(handles, CATEGORY_ORDER, title="Category")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "score_distribution.png", dpi=120)
    plt.close(fig)


def plot_category_by_group(merged: pd.DataFrame) -> None:
    """Stacked bars: NPS category shares by segment and by plan."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, col in zip(axes, ["segment", "plan"]):
        shares = (
            pd.crosstab(merged[col], merged["nps_category"], normalize="index")
            .reindex(columns=CATEGORY_ORDER)
            * 100
        )
        shares.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=[CATEGORY_COLORS[c] for c in CATEGORY_ORDER],
            width=0.7,
        )
        ax.set_xlabel(col.capitalize())
        ax.set_title(f"NPS category shares by {col}")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Category", loc="upper right", fontsize=8)
    axes[0].set_ylabel("Share of responses (%)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "category_share_by_group.png", dpi=120)
    plt.close(fig)


def plot_monthly_responses_and_scores(merged: pd.DataFrame) -> None:
    """Monthly number of responses (bars) and mean score (line)."""
    monthly = (
        merged.assign(month=merged["survey_date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month")["nps_score"]
        .agg(responses="size", mean_score="mean")
    )
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.bar(monthly.index, monthly["responses"], width=20, color="#9ecae1")
    ax1.set_xlabel("Survey month")
    ax1.set_ylabel("Number of responses")
    ax2 = ax1.twinx()
    ax2.plot(monthly.index, monthly["mean_score"], color="#08519c", marker="o")
    ax2.set_ylabel("Mean score")
    ax2.set_ylim(0, 10)
    ax1.set_title("Responses and mean score by month (synthetic data)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "monthly_responses_and_score.png", dpi=120)
    plt.close(fig)


def plot_score_by_tenure(merged: pd.DataFrame) -> None:
    """Boxplot of scores by tenure band."""
    bands = pd.cut(
        merged["tenure_months"],
        bins=[0, 12, 24, 48, 72],
        labels=["1-12", "13-24", "25-48", "49-72"],
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(x=bands, y=merged["nps_score"], color="#c7e9c0", ax=ax)
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Recommendation score (0-10)")
    ax.set_title("Score by tenure band (synthetic data)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "score_by_tenure.png", dpi=120)
    plt.close(fig)


def main() -> None:
    sns.set_theme(style="whitegrid")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    customers, responses = load_data()
    print("NOTE: all data is SYNTHETIC, generated by src/data_generator.py")

    overview("customers", customers)
    overview("survey_responses", responses)
    descriptive_stats(customers, responses)
    integrity_checks(customers, responses)
    merged = aggregations(customers, responses)

    plot_score_distribution(merged)
    plot_category_by_group(merged)
    plot_monthly_responses_and_scores(merged)
    plot_score_by_tenure(merged)

    section("Charts")
    print(f"Saved 4 charts to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
