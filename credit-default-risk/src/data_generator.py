"""Generate the SYNTHETIC credit default dataset.

Creates two linked tables and saves them to data/:
    - customers.csv (4000 rows): borrower profile
    - loans.csv     (5000 rows): loan applications with the target `default`

All data is artificial. The random seed is fixed (42) so the output is reproducible.
Run from the project root:  python src/data_generator.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

SEED = 42
N_CUSTOMERS = 4000
N_LOANS = 5000


def generate_customers(rng: np.random.Generator, n: int = N_CUSTOMERS) -> pd.DataFrame:
    """Create the customers table."""
    age = np.clip(rng.normal(41, 12, n).round(), 18, 75).astype(int)

    # Log-normal income is right-skewed, like real income distributions
    annual_income = np.clip(rng.lognormal(mean=10.9, sigma=0.5, size=n), 15000, 250000)
    annual_income = annual_income.round(2)

    # Employment length cannot exceed the years since age 18
    employment_years = np.minimum(age - 18, rng.gamma(shape=2.0, scale=4.0, size=n))
    employment_years = np.clip(employment_years, 0, 45).round(1)

    home_ownership = rng.choice(["RENT", "MORTGAGE", "OWN"], size=n, p=[0.45, 0.35, 0.20])
    region = rng.choice(["North", "South", "East", "West"], size=n, p=[0.25, 0.30, 0.20, 0.25])

    # Credit score is mildly correlated with income
    income_z = (np.log(annual_income) - 10.9) / 0.5
    credit_score = np.clip(670 + 20 * income_z + rng.normal(0, 70, n), 300, 850)
    credit_score = credit_score.round().astype(int)

    # Existing debt is a share of annual income
    existing_debt = (annual_income * rng.beta(2, 5, n)).round(2)
    num_credit_lines = np.clip(rng.poisson(6, n), 1, 20).astype(int)

    return pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "age": age,
        "annual_income": annual_income,
        "employment_years": employment_years,
        "home_ownership": home_ownership,
        "region": region,
        "credit_score": credit_score,
        "existing_debt": existing_debt,
        "num_credit_lines": num_credit_lines,
    })


def generate_loans(rng: np.random.Generator, customers: pd.DataFrame,
                   n: int = N_LOANS) -> pd.DataFrame:
    """Create the loans table; each loan belongs to one customer.

    Customers can have several loans, and some customers have none.
    """
    customer_ids = rng.choice(customers["customer_id"].to_numpy(), size=n, replace=True)
    cust = customers.set_index("customer_id").loc[customer_ids]

    loan_amount = np.clip(rng.lognormal(mean=9.0, sigma=0.7, size=n), 500, 40000).round(2)
    term_months = rng.choice([36, 60], size=n, p=[0.7, 0.3])
    loan_purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "car",
         "small_business", "medical", "other"],
        size=n, p=[0.40, 0.20, 0.12, 0.10, 0.08, 0.06, 0.04],
    )

    score = cust["credit_score"].to_numpy()
    income = cust["annual_income"].to_numpy()
    debt = cust["existing_debt"].to_numpy()
    employment = cust["employment_years"].to_numpy()
    rent = (cust["home_ownership"].to_numpy() == "RENT").astype(float)

    # Interest rate (%) is higher for weaker credit scores and longer terms
    interest_rate = 22 - 0.02 * (score - 300) + 1.5 * (term_months == 60) + rng.normal(0, 1.5, n)
    interest_rate = np.clip(interest_rate, 4, 30).round(2)

    debt_to_income = np.clip((debt + loan_amount) / income, 0, 3).round(3)

    # Past delinquencies are more frequent for low credit scores
    lam = 0.1 + np.maximum(0, 650 - score) / 200
    num_delinquencies_2y = np.clip(rng.poisson(lam), 0, 10).astype(int)

    # Default probability: logistic function of the risk drivers (defaults are the minority)
    logit = (
        -3.4
        - 0.012 * (score - 650)
        + 0.9 * debt_to_income
        + 0.35 * num_delinquencies_2y
        + 0.08 * (interest_rate - 15)
        + 0.3 * (loan_purpose == "small_business")
        + 0.2 * (term_months == 60)
        + 0.25 * rent
        - 0.03 * employment
    )
    default_prob = 1 / (1 + np.exp(-logit))
    default = (rng.random(n) < default_prob).astype(int)

    # Application dates spread over two years
    offsets = rng.integers(0, 730, n)
    application_date = pd.Timestamp("2022-01-01") + pd.to_timedelta(offsets, unit="D")

    loans = pd.DataFrame({
        "customer_id": customer_ids,
        "loan_amount": loan_amount,
        "term_months": term_months,
        "interest_rate": interest_rate,
        "loan_purpose": loan_purpose,
        "application_date": application_date.strftime("%Y-%m-%d"),
        "debt_to_income": debt_to_income,
        "num_delinquencies_2y": num_delinquencies_2y,
        "default": default,
    })
    # Sort chronologically and assign loan ids in that order
    loans = loans.sort_values("application_date", kind="stable").reset_index(drop=True)
    loans.insert(0, "loan_id", np.arange(1, n + 1))
    return loans


def main() -> None:
    """Generate both tables and save them to data/."""
    rng = np.random.default_rng(SEED)
    os.makedirs(DATA_DIR, exist_ok=True)

    customers = generate_customers(rng)
    loans = generate_loans(rng, customers)

    customers.to_csv(DATA_DIR / "customers.csv", index=False)
    loans.to_csv(DATA_DIR / "loans.csv", index=False)

    print(f"Saved customers.csv: {customers.shape[0]} rows, {customers.shape[1]} columns")
    print(f"Saved loans.csv: {loans.shape[0]} rows, {loans.shape[1]} columns")
    print(f"Default rate in generated loans: {loans['default'].mean():.4f}")


if __name__ == "__main__":
    main()
