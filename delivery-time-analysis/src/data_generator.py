"""Generate SYNTHETIC data for the Delivery Time Analysis project.

Creates two tables and saves them to <project root>/data/:
    - couriers.csv  (300 rows)
    - orders.csv    (5000 rows)

All data is artificial. The relationships (distance, weather, courier load,
vehicle, experience) are assumptions coded below, not real observations.

Run from the project root:
    python src/data_generator.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

SEED = 42
N_COURIERS = 300
N_ORDERS = 5000
START_DATE = "2025-01-01"
N_DAYS = 90

# Assumed average speed (km/h) per vehicle type
VEHICLE_SPEED = {"bike": 15.0, "scooter": 25.0, "car": 22.0}
# Assumed travel time multiplier per weather condition
WEATHER_FACTOR = {"clear": 1.00, "cloudy": 1.05, "rain": 1.25, "snow": 1.50}
WEATHER_PROBS = {"clear": 0.55, "cloudy": 0.20, "rain": 0.17, "snow": 0.08}
PEAK_HOURS = {12, 13, 18, 19, 20}


def generate_couriers(rng: np.random.Generator) -> pd.DataFrame:
    """Create the couriers table."""
    vehicle = rng.choice(["bike", "scooter", "car"], size=N_COURIERS, p=[0.40, 0.40, 0.20])
    experience = np.clip(rng.gamma(shape=2.0, scale=14.0, size=N_COURIERS), 1, 120)
    # Slightly better ratings for more experienced couriers
    rating = 4.2 + 0.003 * np.minimum(experience, 60) + rng.normal(0, 0.25, N_COURIERS)
    return pd.DataFrame({
        "courier_id": np.arange(1, N_COURIERS + 1),
        "vehicle_type": vehicle,
        "experience_months": np.round(experience).astype(int),
        "avg_rating": np.round(np.clip(rating, 3.0, 5.0), 2),
    })


def generate_orders(rng: np.random.Generator, couriers: pd.DataFrame) -> pd.DataFrame:
    """Create the orders table, including the target delivery_time_min."""
    n = N_ORDERS

    # Order timestamps: random day, hour drawn with lunch and dinner peaks
    hour_weights = np.array([
        0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.5, 1.0, 1.5, 1.5, 2.0, 4.0,
        7.0, 6.0, 3.0, 2.5, 3.0, 5.0, 8.0, 8.0, 6.0, 3.5, 1.5, 0.6,
    ])
    hour_weights = hour_weights / hour_weights.sum()
    day = rng.integers(0, N_DAYS, size=n)
    hour = rng.choice(24, size=n, p=hour_weights)
    minute = rng.integers(0, 60, size=n)
    order_datetime = (
        pd.Timestamp(START_DATE)
        + pd.to_timedelta(day, unit="D")
        + pd.to_timedelta(hour, unit="h")
        + pd.to_timedelta(minute, unit="min")
    )
    is_peak = np.isin(hour, list(PEAK_HOURS)).astype(int)

    # Courier assignment and courier attributes
    courier_id = rng.choice(couriers["courier_id"].to_numpy(), size=n)
    info = couriers.set_index("courier_id").loc[courier_id]
    speed = info["vehicle_type"].map(VEHICLE_SPEED).to_numpy()
    experience = info["experience_months"].to_numpy()

    # Order features
    distance = np.clip(rng.lognormal(mean=1.0, sigma=0.55, size=n), 0.3, 15.0)
    weather = rng.choice(list(WEATHER_PROBS), size=n, p=list(WEATHER_PROBS.values()))
    # Active orders held by the courier, including this one (busier at peak hours)
    load = np.clip(1 + rng.poisson(1.0 + 1.5 * is_peak), 1, 6)
    prep_time = np.clip(rng.normal(12.0, 4.0, n), 3.0, 35.0)

    # Delivery time = preparation + travel + extra waiting due to load + noise
    weather_factor = pd.Series(weather).map(WEATHER_FACTOR).to_numpy()
    experience_factor = 1.10 - 0.002 * np.minimum(experience, 60)
    travel = distance / speed * 60.0 * weather_factor * experience_factor
    load_penalty = 3.0 * (load - 1)
    noise = rng.normal(0, 3.0, n) + rng.exponential(1.5, n)
    delivery_time = np.clip(prep_time + travel + load_penalty + noise, 8.0, 120.0)

    return pd.DataFrame({
        "order_id": np.arange(1, n + 1),
        "courier_id": courier_id,
        "order_datetime": order_datetime,
        "is_peak_hour": is_peak,
        "distance_km": np.round(distance, 2),
        "weather": weather,
        "courier_active_orders": load.astype(int),
        "prep_time_min": np.round(prep_time, 1),
        "delivery_time_min": np.round(delivery_time, 1),
    }).sort_values("order_datetime").reset_index(drop=True).assign(
        order_id=lambda d: np.arange(1, len(d) + 1)
    )


def main() -> None:
    rng = np.random.default_rng(SEED)
    os.makedirs(DATA_DIR, exist_ok=True)

    couriers = generate_couriers(rng)
    orders = generate_orders(rng, couriers)

    couriers.to_csv(DATA_DIR / "couriers.csv", index=False)
    orders.to_csv(DATA_DIR / "orders.csv", index=False)

    print(f"Saved {len(couriers)} rows to {DATA_DIR / 'couriers.csv'}")
    print(f"Saved {len(orders)} rows to {DATA_DIR / 'orders.csv'}")


if __name__ == "__main__":
    main()
