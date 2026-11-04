"""Generate synthetic HR data for the employee attrition project.

All data is SYNTHETIC. It is produced with numpy using a fixed seed (42),
so every run creates identical CSV files in the data/ folder.

Tables created:
    employees.csv             - one row per employee, includes the target `attrition`
    satisfaction_surveys.csv  - one row per employee who answered the survey

Run from the project root:
    python src/data_generator.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Paths are relative to the project root, not to the current working directory
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

SEED = 42
N_EMPLOYEES = 3000          # must stay <= 5000 rows per table
SURVEY_RESPONSE_RATE = 0.92  # share of employees who answered the survey

DEPARTMENTS = ["Sales", "Engineering", "Support", "HR", "Finance", "Marketing"]
DEPARTMENT_PROBS = [0.26, 0.28, 0.20, 0.05, 0.09, 0.12]

# Salary multiplier per department (relative to the base salary of a job level)
DEPARTMENT_PAY_FACTOR = {
    "Sales": 1.00,
    "Engineering": 1.20,
    "Support": 0.85,
    "HR": 0.95,
    "Finance": 1.15,
    "Marketing": 1.00,
}

# Probability of regular overtime per department
DEPARTMENT_OVERTIME_PROB = {
    "Sales": 0.35,
    "Engineering": 0.30,
    "Support": 0.38,
    "HR": 0.15,
    "Finance": 0.25,
    "Marketing": 0.22,
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic function."""
    return 1.0 / (1.0 + np.exp(-x))


def to_ordinal(latent: np.ndarray) -> np.ndarray:
    """Convert a continuous latent score into a 1-4 ordinal rating."""
    return np.digitize(latent, [-1.0, 0.0, 1.0]) + 1


def generate_employees(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Generate the base employee attributes (without attrition)."""
    employee_id = np.arange(1, n + 1)

    department = rng.choice(DEPARTMENTS, size=n, p=DEPARTMENT_PROBS)
    gender = rng.choice(["Female", "Male", "Non-binary"], size=n, p=[0.46, 0.51, 0.03])

    # Age and tenure: tenure cannot exceed the working years since age 21
    age = np.clip(np.round(rng.normal(37, 9, n)), 22, 60).astype(int)
    tenure = rng.gamma(shape=2.0, scale=3.0, size=n)
    tenure = np.clip(np.minimum(tenure, age - 21), 0, 35)
    years_at_company = np.round(tenure, 1)

    # Job level grows with age and tenure, with some noise
    raw_level = 0.05 * (age - 22) + 0.06 * years_at_company + rng.normal(0, 0.6, n)
    job_level = np.clip(np.round(1 + raw_level), 1, 5).astype(int)

    # Monthly income depends on job level and department
    pay_factor = np.array([DEPARTMENT_PAY_FACTOR[d] for d in department])
    base_income = 2800 + 1500 * (job_level - 1)
    income = base_income * pay_factor * rng.lognormal(mean=0.0, sigma=0.12, size=n)
    monthly_income = np.clip(np.round(income), 2500, 20000).astype(int)

    # Overtime flag
    overtime_prob = np.array([DEPARTMENT_OVERTIME_PROB[d] for d in department])
    overtime = np.where(rng.random(n) < overtime_prob, "Yes", "No")

    distance_from_home_km = np.clip(np.round(rng.exponential(12, n) + 1), 1, 50).astype(int)
    num_projects = np.clip(rng.poisson(3.5, n), 1, 7).astype(int)

    # Years since last promotion cannot exceed tenure
    promo = np.minimum(rng.exponential(2.0, n), tenure)
    years_since_last_promotion = np.round(promo, 1)

    return pd.DataFrame(
        {
            "employee_id": employee_id,
            "department": department,
            "gender": gender,
            "age": age,
            "job_level": job_level,
            "years_at_company": years_at_company,
            "monthly_income": monthly_income,
            "overtime": overtime,
            "distance_from_home_km": distance_from_home_km,
            "num_projects": num_projects,
            "years_since_last_promotion": years_since_last_promotion,
        }
    )


def generate_surveys(rng: np.random.Generator, employees: pd.DataFrame) -> pd.DataFrame:
    """Generate satisfaction survey scores for every employee (1-4 scale).

    Scores are lower for people working overtime and higher for better-paid
    and more senior people. Non-respondents are dropped later.
    """
    n = len(employees)
    overtime = (employees["overtime"] == "Yes").to_numpy().astype(float)
    income = employees["monthly_income"].to_numpy()
    income_z = (income - income.mean()) / income.std()
    level = employees["job_level"].to_numpy()

    job_latent = rng.normal(0, 1, n) - 0.5 * overtime + 0.25 * income_z + 0.1 * (level - 3)
    wlb_latent = rng.normal(0, 1, n) - 0.8 * overtime
    support_latent = rng.normal(0.1, 1, n)

    return pd.DataFrame(
        {
            "employee_id": employees["employee_id"].to_numpy(),
            "job_satisfaction": to_ordinal(job_latent),
            "work_life_balance": to_ordinal(wlb_latent),
            "manager_support": to_ordinal(support_latent),
        }
    )


def add_attrition(
    rng: np.random.Generator, employees: pd.DataFrame, surveys: pd.DataFrame
) -> pd.DataFrame:
    """Add the binary target `attrition` (1 = left the company).

    The probability follows a logistic model of the employee attributes
    and survey scores. These coefficients are the "ground truth" that the
    analysis should later be able to recover approximately.
    """
    df = employees.merge(surveys, on="employee_id", how="left")

    income = df["monthly_income"].to_numpy()
    income_z = (income - income.mean()) / income.std()
    promo_z = (
        df["years_since_last_promotion"] - df["years_since_last_promotion"].mean()
    ) / df["years_since_last_promotion"].std()

    logit = (
        -1.9
        + 0.85 * (df["overtime"] == "Yes").to_numpy()
        - 0.45 * (df["job_satisfaction"].to_numpy() - 2.5)
        - 0.35 * (df["work_life_balance"].to_numpy() - 2.5)
        - 0.50 * income_z
        + 0.25 * promo_z.to_numpy()
        + 0.02 * (df["distance_from_home_km"].to_numpy() - 12)
        - 0.03 * (df["age"].to_numpy() - 37)
        - 0.04 * df["years_at_company"].to_numpy()
        + 0.30 * (df["department"] == "Sales").to_numpy()
    )
    probability = sigmoid(logit)
    attrition = (rng.random(len(df)) < probability).astype(int)

    result = employees.copy()
    result["attrition"] = attrition
    return result


def main() -> None:
    """Generate all tables and save them to data/."""
    rng = np.random.default_rng(SEED)
    os.makedirs(DATA_DIR, exist_ok=True)

    employees = generate_employees(rng, N_EMPLOYEES)
    surveys_all = generate_surveys(rng, employees)
    employees = add_attrition(rng, employees, surveys_all)

    # Some employees did not answer the survey
    responded = rng.random(len(surveys_all)) < SURVEY_RESPONSE_RATE
    surveys = surveys_all[responded].reset_index(drop=True)

    employees.to_csv(DATA_DIR / "employees.csv", index=False)
    surveys.to_csv(DATA_DIR / "satisfaction_surveys.csv", index=False)

    print(f"Saved {len(employees)} rows to {DATA_DIR / 'employees.csv'}")
    print(f"Saved {len(surveys)} rows to {DATA_DIR / 'satisfaction_surveys.csv'}")
    print(f"Attrition rate in generated data: {employees['attrition'].mean():.3f}")


if __name__ == "__main__":
    main()
