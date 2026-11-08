"""Tests for the synthetic data generator (src/data_generator.py).

The tests build small tables in memory with a fixed seed, so they are fast
and do not need the CSV files, figures or any other project output.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

import data_generator as dg  # noqa: E402

N = 500
SEED = 123


def make_tables(seed: int = SEED, n: int = N):
    """Generate employees, survey scores and the attrition-labelled table."""
    rng = np.random.default_rng(seed)
    employees = dg.generate_employees(rng, n)
    surveys = dg.generate_surveys(rng, employees)
    labelled = dg.add_attrition(rng, employees, surveys)
    return employees, surveys, labelled


def test_employees_columns_and_row_count():
    employees, _, _ = make_tables()
    expected = [
        "employee_id", "department", "gender", "age", "job_level",
        "years_at_company", "monthly_income", "overtime",
        "distance_from_home_km", "num_projects", "years_since_last_promotion",
    ]
    assert list(employees.columns) == expected
    assert len(employees) == N
    assert employees["employee_id"].is_unique
    assert not employees.isna().any().any()


def test_employees_value_ranges():
    employees, _, _ = make_tables()
    assert employees["age"].between(22, 60).all()
    assert employees["job_level"].between(1, 5).all()
    assert employees["years_at_company"].between(0, 35).all()
    assert employees["monthly_income"].between(2500, 20000).all()
    assert employees["distance_from_home_km"].between(1, 50).all()
    assert employees["num_projects"].between(1, 7).all()
    # Promotion gap can never exceed tenure
    assert (employees["years_since_last_promotion"] <= employees["years_at_company"]).all()
    assert set(employees["department"]) <= set(dg.DEPARTMENTS)
    assert set(employees["overtime"]) <= {"Yes", "No"}


def test_survey_scores_are_ordinal_1_to_4():
    employees, surveys, _ = make_tables()
    assert list(surveys.columns) == [
        "employee_id", "job_satisfaction", "work_life_balance", "manager_support"
    ]
    assert len(surveys) == len(employees)
    for col in ["job_satisfaction", "work_life_balance", "manager_support"]:
        assert surveys[col].between(1, 4).all()


def test_attrition_is_binary_and_keeps_rows():
    employees, _, labelled = make_tables()
    assert len(labelled) == len(employees)
    assert set(labelled["attrition"].unique()) <= {0, 1}
    # Both classes should appear in a table of this size
    assert labelled["attrition"].nunique() == 2


def test_generation_is_reproducible_with_same_seed():
    _, _, first = make_tables(seed=7)
    _, _, second = make_tables(seed=7)
    assert first.equals(second)
