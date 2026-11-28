"""Generate SYNTHETIC manufacturing data for the defect analysis project.

Run from the project root:
    python src/data_generator.py

Outputs (in data/):
    production_batches.csv - one row per inspected batch
    defects.csv            - one row per defective unit found

All data is synthetic and produced with numpy using a fixed seed (42).
"""

from pathlib import Path
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

SEED = 42
N_BATCHES = 2000
BATCHES_PER_DAY = 8
START_DATE = pd.Timestamp("2025-01-01")
MAX_ROWS = 5000  # hard limit per table

LINES = ["Line_A", "Line_B", "Line_C"]
MACHINES_BY_LINE = {
    "Line_A": ["M01", "M02", "M03"],
    "Line_B": ["M04", "M05", "M06"],
    "Line_C": ["M07", "M08", "M09"],
}
SHIFTS = ["Morning", "Afternoon", "Night"]
SHIFT_PROBS = [0.375, 0.375, 0.25]
SUPPLIERS = ["Supplier_X", "Supplier_Y", "Supplier_Z"]
SUPPLIER_PROBS = [0.5, 0.3, 0.2]
PRODUCT_TYPES = ["Type_A", "Type_B", "Type_C"]
OPERATORS = [f"OP{i:02d}" for i in range(1, 21)]

# Built-in root causes (what a later analysis should be able to discover)
DRIFT_MACHINE = "M05"        # temperature drifts upward after DRIFT_START_DAY
DRIFT_START_DAY = 150
DRIFT_MAX_C = 4.0
SHIFT_MACHINE = "M07"        # diameter shifts upward after SHIFT_START_DAY
SHIFT_START_DAY = 160
DIAMETER_SHIFT_MM = 0.03
BAD_OPERATOR = "OP13"

BASE_DEFECT_PROB = 0.02
SHIFT_MULT = {"Morning": 1.0, "Afternoon": 1.0, "Night": 1.4}
SUPPLIER_MULT = {"Supplier_X": 1.0, "Supplier_Y": 1.0, "Supplier_Z": 1.6}

DEFECT_TYPES = ["Scratch", "Dimensional", "Crack", "Porosity", "Discoloration"]
BASE_TYPE_WEIGHTS = np.array([0.30, 0.20, 0.15, 0.15, 0.20])

SEVERITIES = ["Minor", "Major", "Critical"]
SEVERITY_PROBS = {
    "Scratch": [0.80, 0.18, 0.02],
    "Dimensional": [0.40, 0.45, 0.15],
    "Crack": [0.10, 0.45, 0.45],
    "Porosity": [0.25, 0.50, 0.25],
    "Discoloration": [0.85, 0.14, 0.01],
}
COST_MEDIAN_USD = {"Minor": 5.0, "Major": 25.0, "Critical": 120.0}


# ---------------------------------------------------------------------------
# Table 1: production batches
# ---------------------------------------------------------------------------
def generate_batches(rng: np.random.Generator) -> pd.DataFrame:
    """Create the batch table, including defect counts driven by hidden causes."""
    idx = np.arange(N_BATCHES)
    day_index = idx // BATCHES_PER_DAY  # 0-based day number, ordered in time

    line = rng.choice(LINES, size=N_BATCHES)
    machine = np.array([rng.choice(MACHINES_BY_LINE[ln]) for ln in line])
    shift = rng.choice(SHIFTS, size=N_BATCHES, p=SHIFT_PROBS)
    supplier = rng.choice(SUPPLIERS, size=N_BATCHES, p=SUPPLIER_PROBS)
    product = rng.choice(PRODUCT_TYPES, size=N_BATCHES)
    operator = rng.choice(OPERATORS, size=N_BATCHES)
    units_inspected = rng.integers(40, 61, size=N_BATCHES)  # 40..60 inclusive

    # Process settings: in control by default, with two injected problems
    temperature = rng.normal(200.0, 2.5, size=N_BATCHES)
    drift = np.clip((day_index - DRIFT_START_DAY) / 100.0 * DRIFT_MAX_C, 0, DRIFT_MAX_C)
    temperature = temperature + np.where(machine == DRIFT_MACHINE, drift, 0.0)
    temperature = np.clip(temperature, 185.0, 215.0)

    pressure = np.clip(rng.normal(5.0, 0.15, size=N_BATCHES), 4.4, 5.6)

    diameter_shifted = (machine == SHIFT_MACHINE) & (day_index >= SHIFT_START_DAY)
    diameter = rng.normal(25.0, 0.015, size=N_BATCHES)
    diameter = diameter + np.where(diameter_shifted, DIAMETER_SHIFT_MM, 0.0)
    diameter = np.clip(diameter, 24.9, 25.1)

    # Defect probability = base rate x multipliers for each hidden cause
    temp_excess = np.clip(temperature - 200.0, 0, None)
    prob = (
        BASE_DEFECT_PROB
        * np.vectorize(SHIFT_MULT.get)(shift)
        * np.vectorize(SUPPLIER_MULT.get)(supplier)
        * np.where(operator == BAD_OPERATOR, 1.3, 1.0)
        * np.exp(0.10 * temp_excess)
        * np.where(diameter_shifted, 1.4, 1.0)
    )
    prob = np.clip(prob, 0.002, 0.25)
    defects_count = rng.binomial(units_inspected, prob)

    batches = pd.DataFrame(
        {
            "batch_id": [f"B{i + 1:05d}" for i in idx],
            "production_date": (START_DATE + pd.to_timedelta(day_index, unit="D")).strftime("%Y-%m-%d"),
            "shift": shift,
            "line": line,
            "machine_id": machine,
            "operator_id": operator,
            "supplier": supplier,
            "product_type": product,
            "units_inspected": units_inspected,
            "defects_count": defects_count,
            "defect_rate": np.round(defects_count / units_inspected, 4),
            "temperature_c": np.round(temperature, 2),
            "pressure_bar": np.round(pressure, 3),
            "avg_diameter_mm": np.round(diameter, 4),
        }
    )
    return batches


# ---------------------------------------------------------------------------
# Table 2: individual defects
# ---------------------------------------------------------------------------
def type_probabilities(row: pd.Series) -> np.ndarray:
    """Defect type mix for a batch: hidden causes favour specific defect types."""
    w = BASE_TYPE_WEIGHTS.copy()
    names = {name: i for i, name in enumerate(DEFECT_TYPES)}

    if row["temperature_c"] - 200.0 > 2.0:  # overheating -> porosity and cracks
        w[names["Porosity"]] *= 2.0
        w[names["Crack"]] *= 1.5
    if row["supplier"] == "Supplier_Z":  # poor raw material -> cracks
        w[names["Crack"]] *= 2.0
    if row["machine_id"] == SHIFT_MACHINE and row["avg_diameter_mm"] > 25.0 + DIAMETER_SHIFT_MM / 2:
        w[names["Dimensional"]] *= 3.0  # shifted diameter -> dimensional defects
    return w / w.sum()


def generate_defects(batches: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Expand each batch's defect count into one row per defective unit."""
    records = []
    for _, row in batches.iterrows():
        n = int(row["defects_count"])
        if n == 0:
            continue
        counts = rng.multinomial(n, type_probabilities(row))
        for dtype, k in zip(DEFECT_TYPES, counts):
            for _ in range(k):
                severity = rng.choice(SEVERITIES, p=SEVERITY_PROBS[dtype])
                cost = rng.lognormal(np.log(COST_MEDIAN_USD[severity]), 0.4)
                records.append((row["batch_id"], dtype, severity, float(np.clip(cost, 1.0, 500.0))))

    defects = pd.DataFrame(records, columns=["batch_id", "defect_type", "severity", "repair_cost_usd"])
    defects.insert(0, "defect_id", [f"D{i + 1:05d}" for i in range(len(defects))])
    defects["repair_cost_usd"] = defects["repair_cost_usd"].round(2)
    return defects


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(SEED)

    batches = generate_batches(rng)
    defects = generate_defects(batches, rng)

    for name, table in [("production_batches", batches), ("defects", defects)]:
        if len(table) > MAX_ROWS:
            raise ValueError(f"Table {name} has {len(table)} rows, limit is {MAX_ROWS}")

    os.makedirs(DATA_DIR, exist_ok=True)
    batches.to_csv(DATA_DIR / "production_batches.csv", index=False)
    defects.to_csv(DATA_DIR / "defects.csv", index=False)

    print("Synthetic data generated (seed 42).")
    print(f"production_batches.csv: {len(batches)} rows, {batches.shape[1]} columns")
    print(f"defects.csv:            {len(defects)} rows, {defects.shape[1]} columns")
    print(f"Saved to: {DATA_DIR}")


if __name__ == "__main__":
    main()
