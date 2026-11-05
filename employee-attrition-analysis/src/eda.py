"""Exploratory data analysis for the synthetic employee attrition data.

Loads the CSV files produced by src/data_generator.py, prints data quality
checks, descriptive statistics and a few aggregations, and saves charts
to the figures/ folder.

All data is SYNTHETIC. No numbers are hard-coded here: everything printed
comes from the data itself.

Run from the project root (after generating the data):
    python src/data_generator.py
    python src/eda.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: no display needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Paths are relative to the project root, not to the current working directory
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "figures"

SURVEY_COLUMNS = ["job_satisfaction", "work_life_balance", "manager_support"]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the employees and survey tables from data/."""
    employees = pd.read_csv(DATA_DIR / "employees.csv")
    surveys = pd.read_csv(DATA_DIR / "satisfaction_surveys.csv")
    return employees, surveys


def print_section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def describe_tables(employees: pd.DataFrame, surveys: pd.DataFrame) -> None:
    """Print shape, dtypes, missing values and duplicate checks."""
    for name, df in [("employees", employees), ("satisfaction_surveys", surveys)]:
        print_section(f"Table: {name}")
        print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        print("\nData types:")
        print(df.dtypes.to_string())
        print("\nMissing values per column:")
        print(df.isna().sum().to_string())
        print(f"\nDuplicate employee_id values: {df['employee_id'].duplicated().sum()}")

    # Survey coverage: employees without a survey row are missing after a join
    n_missing_survey = (~employees["employee_id"].isin(surveys["employee_id"])).sum()
    print_section("Survey coverage")
    print(f"Employees without a survey response: {n_missing_survey} "
          f"({n_missing_survey / len(employees):.1%})")


def descriptive_statistics(df: pd.DataFrame) -> None:
    """Print descriptive statistics for numeric and categorical columns."""
    print_section("Descriptive statistics (numeric columns)")
    print(df.drop(columns=["employee_id"]).describe().T.round(2).to_string())

    print_section("Categorical columns: value shares")
    for col in ["department", "gender", "overtime"]:
        print(f"\n{col}:")
        print(df[col].value_counts(normalize=True).round(3).to_string())

    print_section("Overall attrition")
    print(f"Attrition rate: {df['attrition'].mean():.3f} "
          f"({df['attrition'].sum()} of {len(df)} employees left)")


def aggregations(df: pd.DataFrame) -> None:
    """Print attrition rates across meaningful groups."""
    print_section("Attrition rate by department")
    by_dept = df.groupby("department")["attrition"].agg(["count", "mean"])
    print(by_dept.sort_values("mean", ascending=False).round(3).to_string())

    print_section("Attrition rate by overtime")
    print(df.groupby("overtime")["attrition"].agg(["count", "mean"]).round(3).to_string())

    print_section("Attrition rate by job level")
    print(df.groupby("job_level")["attrition"].agg(["count", "mean"]).round(3).to_string())

    # Tenure bands to see how attrition changes with time at the company
    bands = pd.cut(
        df["years_at_company"],
        bins=[-0.01, 2, 5, 10, 40],
        labels=["0-2", "2-5", "5-10", "10+"],
    )
    print_section("Attrition rate by tenure band (years)")
    print(df.groupby(bands, observed=True)["attrition"].agg(["count", "mean"]).round(3).to_string())

    # Survey scores: left join keeps non-respondents, who are excluded per column
    print_section("Mean numeric features: stayed (0) vs left (1)")
    cols = ["age", "years_at_company", "monthly_income", "distance_from_home_km",
            "years_since_last_promotion", "num_projects"] + SURVEY_COLUMNS
    print(df.groupby("attrition")[cols].mean().round(2).T.to_string())


def plot_attrition_by_group(df: pd.DataFrame) -> None:
    """Bar charts of attrition rate by department and overtime."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    dept = (df.groupby("department")["attrition"].mean()
            .sort_values(ascending=False).reset_index())
    sns.barplot(data=dept, x="department", y="attrition", color="steelblue", ax=axes[0])
    axes[0].set_title("Attrition rate by department")
    axes[0].set_xlabel("Department")
    axes[0].set_ylabel("Attrition rate")
    axes[0].tick_params(axis="x", rotation=30)

    ot = df.groupby("overtime")["attrition"].mean().reset_index()
    sns.barplot(data=ot, x="overtime", y="attrition", color="indianred", ax=axes[1])
    axes[1].set_title("Attrition rate by overtime")
    axes[1].set_xlabel("Works overtime")
    axes[1].set_ylabel("Attrition rate")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "attrition_by_group.png", dpi=120)
    plt.close(fig)


def plot_income_distribution(df: pd.DataFrame) -> None:
    """Income distribution split by attrition status."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_df = df.assign(status=df["attrition"].map({0: "Stayed", 1: "Left"}))
    sns.histplot(data=plot_df, x="monthly_income", hue="status", bins=40,
                 stat="density", common_norm=False, element="step", ax=ax)
    ax.set_title("Monthly income distribution by attrition status")
    ax.set_xlabel("Monthly income (synthetic currency units)")
    ax.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "income_distribution.png", dpi=120)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Correlation heatmap of numeric features, survey scores and the target."""
    numeric = df.select_dtypes("number").drop(columns=["employee_id"])
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation matrix (numeric features and attrition)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=120)
    plt.close(fig)


def main() -> None:
    """Run the full EDA."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    employees, surveys = load_data()
    describe_tables(employees, surveys)

    # Left join: employees without a survey get NaN in survey columns
    df = employees.merge(surveys, on="employee_id", how="left")
    print_section("Missing values after joining surveys")
    print(df[SURVEY_COLUMNS].isna().sum().to_string())

    descriptive_statistics(df)
    aggregations(df)

    plot_attrition_by_group(df)
    plot_income_distribution(df)
    plot_correlation_heatmap(df)
    print_section("Charts saved")
    print(f"Figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
