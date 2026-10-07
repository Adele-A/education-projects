"""Exploratory data analysis for the SYNTHETIC credit default dataset.

Loads data/customers.csv and data/loans.csv (created by src/data_generator.py),
prints data quality checks, descriptive statistics and key aggregations, and
saves charts to figures/.

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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two CSV tables."""
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    loans = pd.read_csv(DATA_DIR / "loans.csv", parse_dates=["application_date"])
    return customers, loans


def data_quality(name: str, df: pd.DataFrame) -> None:
    """Print shape, dtypes, missing values and duplicates for one table."""
    print(f"\n=== {name}: data quality ===")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nData types:")
    print(df.dtypes.to_string())
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    print(f"\nDuplicated rows: {df.duplicated().sum()}")


def describe_tables(customers: pd.DataFrame, loans: pd.DataFrame) -> None:
    """Print descriptive statistics for numeric columns."""
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 30)
    print("\n=== Customers: descriptive statistics ===")
    print(customers.drop(columns="customer_id").describe().round(2).T.to_string())
    print("\n=== Loans: descriptive statistics ===")
    print(loans.drop(columns="loan_id").describe().round(3).T.to_string())


def default_rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return loan count and default rate for each value of a column."""
    out = df.groupby(col, observed=True)["default"].agg(loans="count", default_rate="mean")
    return out.sort_values("default_rate", ascending=False)


def aggregations(data: pd.DataFrame) -> None:
    """Print the key aggregations of default behaviour."""
    print("\n=== Target balance ===")
    counts = data["default"].value_counts().sort_index()
    print(counts.to_string())
    print(f"Overall default rate: {data['default'].mean():.4f}")

    print("\n=== Default rate by loan purpose ===")
    print(default_rate_by(data, "loan_purpose").round(4).to_string())

    print("\n=== Default rate by credit score band ===")
    print(default_rate_by(data, "score_band").sort_index().round(4).to_string())

    print("\n=== Default rate by number of delinquencies (2y) ===")
    delinq = data.assign(delinq_group=data["num_delinquencies_2y"].clip(upper=3))
    print(default_rate_by(delinq, "delinq_group").sort_index().round(4).to_string())

    print("\n=== Default rate by region and home ownership ===")
    pivot = data.pivot_table(index="region", columns="home_ownership",
                             values="default", aggfunc="mean")
    print(pivot.round(4).to_string())

    print("\n=== Default rate by term ===")
    print(default_rate_by(data, "term_months").round(4).to_string())


def plot_figures(data: pd.DataFrame) -> None:
    """Save the EDA charts to figures/."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Default rate by credit score band
    by_band = default_rate_by(data, "score_band").sort_index().reset_index()
    by_band["score_band"] = by_band["score_band"].astype(str)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=by_band, x="score_band", y="default_rate", color="steelblue", ax=ax)
    ax.axhline(data["default"].mean(), color="red", linestyle="--", label="Overall default rate")
    ax.set_title("Default rate by credit score band (synthetic data)")
    ax.set_xlabel("Credit score band")
    ax.set_ylabel("Default rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "default_rate_by_score_band.png", dpi=120)
    plt.close(fig)

    # 2. Default rate by loan purpose
    by_purpose = default_rate_by(data, "loan_purpose").reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=by_purpose, x="default_rate", y="loan_purpose", color="darkorange", ax=ax)
    ax.set_title("Default rate by loan purpose (synthetic data)")
    ax.set_xlabel("Default rate")
    ax.set_ylabel("Loan purpose")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "default_rate_by_purpose.png", dpi=120)
    plt.close(fig)

    # 3. Distribution of debt-to-income by default status
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=data, x="default", y="debt_to_income", ax=ax)
    ax.set_title("Debt-to-income by default status (synthetic data)")
    ax.set_xlabel("Default (0 = repaid, 1 = defaulted)")
    ax.set_ylabel("Debt-to-income")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dti_by_default.png", dpi=120)
    plt.close(fig)

    # 4. Correlation heatmap of numeric features and the target
    num_cols = ["age", "annual_income", "employment_years", "credit_score",
                "existing_debt", "num_credit_lines", "loan_amount", "term_months",
                "interest_rate", "debt_to_income", "num_delinquencies_2y", "default"]
    corr = data[num_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation matrix (synthetic data)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correlation_heatmap.png", dpi=120)
    plt.close(fig)

    print(f"\nSaved 4 charts to {FIG_DIR.name}/")


def main() -> None:
    """Run the full EDA."""
    customers, loans = load_data()

    data_quality("customers", customers)
    data_quality("loans", loans)
    describe_tables(customers, loans)

    # Referential integrity: every loan must point to an existing customer
    orphans = (~loans["customer_id"].isin(customers["customer_id"])).sum()
    print(f"\nLoans without a matching customer: {orphans}")

    # Join the tables: one row per loan with borrower attributes
    data = loans.merge(customers, on="customer_id", how="left")
    data["score_band"] = pd.cut(
        data["credit_score"], bins=[299, 579, 669, 739, 799, 850],
        labels=["300-579", "580-669", "670-739", "740-799", "800-850"],
    )

    aggregations(data)
    plot_figures(data)


if __name__ == "__main__":
    main()
