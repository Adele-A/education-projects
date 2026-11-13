"""Generate SYNTHETIC warehouse order picking data.

Creates two CSV tables in the data/ folder at the project root:
  - pickers.csv: one row per picker
  - orders.csv:  one row per picked order

All data is artificial and produced with numpy using a fixed seed (42).
Run from the project root:  python src/data_generator.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Paths are computed relative to the project root, not the working directory
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

SEED = 42
N_PICKERS = 60
N_ORDERS = 5000
START_DATE = "2025-01-01"
N_DAYS = 90

# Latent order profiles (not saved to the output tables).
# Each profile defines the typical shape of an order.
PROFILES = {
    # name: (share, base_lines, extra_lines_lambda, extra_units_per_line,
    #        mean_unit_weight_kg, zone_prob)
    "small":      (0.45, 1, 1.0, 0.3, 0.6, 0.10),
    "medium":     (0.30, 4, 3.0, 0.8, 1.0, 0.30),
    "bulk":       (0.15, 2, 2.0, 12.0, 1.8, 0.15),
    "multi_zone": (0.10, 8, 5.0, 1.5, 0.8, 0.60),
}


def generate_pickers(rng: np.random.Generator) -> pd.DataFrame:
    """Create the pickers table with experience and contract type."""
    picker_ids = [f"P{i:03d}" for i in range(1, N_PICKERS + 1)]
    experience = rng.integers(1, 121, size=N_PICKERS)  # months
    contract = rng.choice(
        ["full_time", "part_time", "temp"], size=N_PICKERS, p=[0.55, 0.25, 0.20]
    )
    # Temp workers tend to be less experienced
    experience = np.where(contract == "temp", np.minimum(experience, 24), experience)

    return pd.DataFrame(
        {
            "picker_id": picker_ids,
            "experience_months": experience.astype(int),
            "contract_type": contract,
        }
    )


def generate_orders(rng: np.random.Generator, pickers: pd.DataFrame) -> pd.DataFrame:
    """Create the orders table; picking time depends on order and picker traits."""
    names = list(PROFILES.keys())
    shares = [PROFILES[n][0] for n in names]
    profile_idx = rng.choice(len(names), size=N_ORDERS, p=shares)

    # Per-order profile parameters
    base_lines = np.array([PROFILES[n][1] for n in names])[profile_idx]
    line_lam = np.array([PROFILES[n][2] for n in names])[profile_idx]
    unit_lam = np.array([PROFILES[n][3] for n in names])[profile_idx]
    unit_weight = np.array([PROFILES[n][4] for n in names])[profile_idx]
    zone_prob = np.array([PROFILES[n][5] for n in names])[profile_idx]

    # Order size: lines, units (at least one unit per line), weight
    n_lines = np.clip(base_lines + rng.poisson(line_lam), 1, 40)
    n_units = n_lines + rng.poisson(unit_lam * n_lines)
    n_units = np.clip(n_units, n_lines, 600)
    weight_per_unit = unit_weight * rng.lognormal(mean=0.0, sigma=0.35, size=N_ORDERS)
    total_weight = np.clip(n_units * weight_per_unit, 0.1, 500.0)

    # Zones visited cannot exceed the number of lines
    n_zones = 1 + rng.binomial(5, zone_prob)
    n_zones = np.clip(np.minimum(n_zones, n_lines), 1, 8)

    # Walking distance grows with zones and lines
    distance = 20 + 35 * n_zones + 4 * n_lines + rng.normal(0, 15, size=N_ORDERS)
    distance = np.clip(distance, 10, 600)

    # Assign pickers and a hidden speed factor (experience helps, plus personal noise)
    picker_pos = rng.integers(0, N_PICKERS, size=N_ORDERS)
    exp_months = pickers["experience_months"].to_numpy()
    personal = rng.lognormal(mean=0.0, sigma=0.10, size=N_PICKERS)
    speed_factor = np.clip(
        (0.85 + 0.15 * np.minimum(exp_months, 36) / 36) * personal, 0.6, 1.5
    )
    order_speed = speed_factor[picker_pos]

    # Calendar and shift
    day_offset = rng.integers(0, N_DAYS, size=N_ORDERS)
    order_date = pd.to_datetime(START_DATE) + pd.to_timedelta(day_offset, unit="D")
    shift = rng.choice(
        ["morning", "afternoon", "night"], size=N_ORDERS, p=[0.45, 0.35, 0.20]
    )
    shift_mult = np.where(shift == "night", 1.08, 1.0)
    priority = rng.choice(["standard", "express"], size=N_ORDERS, p=[0.8, 0.2])

    # Picking time (minutes): setup + walking + per-line, per-unit, weight and zone effects
    base_time = (
        1.0
        + distance / 70.0
        + 0.35 * n_lines
        + 0.05 * n_units
        + 0.01 * total_weight
        + 0.5 * n_zones
    )
    noise = rng.lognormal(mean=0.0, sigma=0.15, size=N_ORDERS)
    pick_time = np.clip(base_time / order_speed * shift_mult * noise, 0.5, 240.0)

    orders = pd.DataFrame(
        {
            "order_id": [f"O{i:05d}" for i in range(1, N_ORDERS + 1)],
            "picker_id": pickers["picker_id"].to_numpy()[picker_pos],
            "order_date": order_date.strftime("%Y-%m-%d"),
            "shift": shift,
            "priority": priority,
            "n_lines": n_lines.astype(int),
            "n_units": n_units.astype(int),
            "total_weight_kg": np.round(total_weight, 2),
            "n_zones": n_zones.astype(int),
            "distance_m": np.round(distance, 1),
            "pick_time_min": np.round(pick_time, 2),
        }
    )
    return orders


def main() -> None:
    """Generate both tables and save them to data/."""
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    pickers = generate_pickers(rng)
    orders = generate_orders(rng, pickers)

    pickers.to_csv(DATA_DIR / "pickers.csv", index=False)
    orders.to_csv(DATA_DIR / "orders.csv", index=False)

    print(f"Saved {len(pickers)} rows to {DATA_DIR / 'pickers.csv'}")
    print(f"Saved {len(orders)} rows to {DATA_DIR / 'orders.csv'}")


if __name__ == "__main__":
    main()
