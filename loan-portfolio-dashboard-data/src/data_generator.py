"""Generate a synthetic loan portfolio dataset.

Creates three tables in the data/ folder:
    customers.csv       - one row per customer
    loans.csv           - one row per loan
    loan_snapshots.csv  - one row per loan per month-end (performance history)

All data is SYNTHETIC. The random seed is fixed so runs are reproducible.

Run from the project root:
    python src/data_generator.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths and global settings
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

SEED = 42
N_CUSTOMERS = 500
N_LOANS = 400
MAX_ROWS_PER_TABLE = 5000

AS_OF_DATE = pd.Timestamp("2025-12-31")   # last snapshot date
FIRST_ORIG_MONTH = pd.Period("2024-09", "M")
N_ORIG_MONTHS = 10                        # originations: 2024-09 .. 2025-06

REGIONS = ["North", "South", "East", "West", "Central"]
EMPLOYMENT = ["Employed", "Self-employed", "Retired", "Unemployed"]

# Product parameters: principal range and allowed terms (months)
PRODUCTS = {
    "Personal Loan":    {"principal": (2000, 25000),  "terms": [12, 24, 36, 48]},
    "Auto Loan":        {"principal": (8000, 45000),  "terms": [36, 48, 60]},
    "Home Improvement": {"principal": (3000, 40000),  "terms": [24, 36, 48, 60]},
    "Small Business":   {"principal": (10000, 80000), "terms": [24, 36, 48, 60]},
}

# Risk grade drives the interest rate and the delinquency probabilities
BASE_RATE = {"A": 7.0, "B": 10.0, "C": 14.0, "D": 19.0, "E": 25.0}
RISK_MULTIPLIER = {"A": 0.5, "B": 0.8, "C": 1.0, "D": 1.5, "E": 2.0}

# Delinquency states: 0..4, labelled by days past due
DPD_LABELS = ["Current", "1-29", "30-59", "60-89", "90+"]

MONTHLY_PREPAY_PROB = 0.01     # chance a current loan is fully prepaid
CHARGE_OFF_PROB = 0.25         # monthly chance a 90+ loan is charged off


# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------
def credit_segment(score: int) -> str:
    """Map a credit score to a credit segment."""
    if score >= 720:
        return "Prime"
    if score >= 640:
        return "Near-Prime"
    return "Subprime"


def generate_customers(rng: np.random.Generator) -> pd.DataFrame:
    """Create the customers table."""
    credit_score = np.clip(rng.normal(670, 70, N_CUSTOMERS), 450, 850).round().astype(int)
    income = np.clip(rng.lognormal(mean=10.9, sigma=0.5, size=N_CUSTOMERS), 18000, 250000)
    age = np.clip(rng.normal(41, 12, N_CUSTOMERS), 21, 70).round().astype(int)

    customers = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(1, N_CUSTOMERS + 1)],
        "age": age,
        "region": rng.choice(REGIONS, N_CUSTOMERS, p=[0.2, 0.25, 0.2, 0.2, 0.15]),
        "employment_status": rng.choice(EMPLOYMENT, N_CUSTOMERS, p=[0.68, 0.14, 0.12, 0.06]),
        "annual_income": income.round(0).astype(int),
        "credit_score": credit_score,
    })
    customers["credit_segment"] = customers["credit_score"].apply(credit_segment)
    return customers


# ---------------------------------------------------------------------------
# Loans
# ---------------------------------------------------------------------------
def risk_grade(score: int) -> str:
    """Map a credit score to a risk grade (A = best, E = worst)."""
    if score >= 760:
        return "A"
    if score >= 720:
        return "B"
    if score >= 680:
        return "C"
    if score >= 620:
        return "D"
    return "E"


def monthly_payment(principal: float, annual_rate_pct: float, term: int) -> float:
    """Level annuity payment."""
    r = annual_rate_pct / 100 / 12
    return principal * r / (1 - (1 + r) ** -term)


def generate_loans(rng: np.random.Generator, customers: pd.DataFrame) -> pd.DataFrame:
    """Create the loans table; each loan belongs to a distinct customer."""
    chosen = customers.sample(n=N_LOANS, random_state=SEED).reset_index(drop=True)
    product_names = list(PRODUCTS)
    products = rng.choice(product_names, N_LOANS, p=[0.35, 0.30, 0.20, 0.15])

    rows = []
    for i, cust in chosen.iterrows():
        prod = products[i]
        lo, hi = PRODUCTS[prod]["principal"]
        principal = int(rng.integers(lo // 100, hi // 100 + 1)) * 100
        term = int(rng.choice(PRODUCTS[prod]["terms"]))
        grade = risk_grade(cust["credit_score"])
        rate = float(np.clip(BASE_RATE[grade] + rng.normal(0, 1.0), 5.0, 30.0))
        rate = round(rate, 2)

        month = FIRST_ORIG_MONTH + int(rng.integers(0, N_ORIG_MONTHS))
        orig_date = pd.Timestamp(year=month.year, month=month.month,
                                 day=int(rng.integers(1, 29)))

        rows.append({
            "loan_id": f"L{i + 1:05d}",
            "customer_id": cust["customer_id"],
            "origination_date": orig_date.strftime("%Y-%m-%d"),
            "product_type": prod,
            "principal": principal,
            "interest_rate": rate,
            "term_months": term,
            "risk_grade": grade,
            "monthly_payment": round(monthly_payment(principal, rate, term), 2),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Monthly performance snapshots
# ---------------------------------------------------------------------------
def next_state(state: int, mult: float, rng: np.random.Generator) -> int:
    """Draw the next delinquency state; returns -1 for charge-off.

    Roll probabilities are scaled by the loan's risk multiplier.
    """
    u = rng.random()

    if state == 0:
        return 1 if u < min(0.5, 0.03 * mult) else 0
    if state == 1:
        if u < 0.55:
            return 0
        if u < 0.55 + min(0.4, 0.20 * mult):
            return 2
        return 1
    if state == 2:
        if u < 0.25:
            return 0
        if u < 0.25 + 0.15:
            return 1
        if u < 0.40 + min(0.5, 0.35 * mult):
            return 3
        return 2
    if state == 3:
        if u < 0.15:
            return 0
        if u < 0.15 + min(0.6, 0.45 * mult):
            return 4
        return 3
    # state == 4 (90+)
    if u < 0.05:
        return 0
    if u < 0.05 + CHARGE_OFF_PROB:
        return -1
    return 4


def generate_snapshots(rng: np.random.Generator, loans: pd.DataFrame) -> pd.DataFrame:
    """Simulate month-end snapshots for every loan up to the as-of date."""
    rows = []
    for loan in loans.itertuples(index=False):
        orig_period = pd.Period(loan.origination_date, "M")
        monthly_rate = loan.interest_rate / 100 / 12
        mult = RISK_MULTIPLIER[loan.risk_grade]
        balance = float(loan.principal)
        state = 0
        m = 0

        while True:
            m += 1
            snap_date = (orig_period + m).to_timestamp(how="end").normalize()
            if snap_date > AS_OF_DATE:
                break

            # Prepayment is only possible for loans that are current
            if state == 0 and rng.random() < MONTHLY_PREPAY_PROB:
                rows.append((loan.loan_id, snap_date, m, 0.0, "Current", "paid_off"))
                break

            new_state = next_state(state, mult, rng)

            if new_state == -1:
                # Charged off: balance at charge-off stays on the row, then the loan stops
                rows.append((loan.loan_id, snap_date, m, round(balance, 2), "90+", "charged_off"))
                break

            state = new_state
            if state == 0:
                # Payment made: interest accrues, payment reduces the balance
                balance = max(0.0, balance * (1 + monthly_rate) - loan.monthly_payment)
                if m >= loan.term_months or balance < 1.0:
                    rows.append((loan.loan_id, snap_date, m, 0.0, "Current", "paid_off"))
                    break
            # Delinquent loans: balance is held flat (no payment received)

            rows.append((loan.loan_id, snap_date, m, round(balance, 2),
                         DPD_LABELS[state], "active"))

    snapshots = pd.DataFrame(rows, columns=[
        "loan_id", "snapshot_date", "months_on_book",
        "balance", "dpd_bucket", "status",
    ])
    snapshots["snapshot_date"] = snapshots["snapshot_date"].dt.strftime("%Y-%m-%d")
    return snapshots


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(SEED)
    os.makedirs(DATA_DIR, exist_ok=True)

    customers = generate_customers(rng)
    loans = generate_loans(rng, customers)
    snapshots = generate_snapshots(rng, loans)

    tables = {
        "customers": customers,
        "loans": loans,
        "loan_snapshots": snapshots,
    }
    for name, df in tables.items():
        assert len(df) <= MAX_ROWS_PER_TABLE, f"{name} has too many rows: {len(df)}"
        path = DATA_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"Saved {name}: {len(df)} rows, {df.shape[1]} columns -> {path}")


if __name__ == "__main__":
    main()
