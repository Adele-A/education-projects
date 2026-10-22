"""Generate SYNTHETIC hourly household energy data.

Run from the project root:
    python src/data_generator.py

Outputs (written to data/):
    households.csv      - household attributes
    weather_hourly.csv  - hourly outdoor temperature
    energy_hourly.csv   - hourly consumption per household (kWh)

All data is artificial and produced with numpy using a fixed seed.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

SEED = 42
N_HOUSEHOLDS = 6
N_DAYS = 30
START = "2025-01-01 00:00:00"

# Relative hourly load shape: morning peak (7-9) and larger evening peak (18-22)
HOURLY_PROFILE = np.array([
    0.55, 0.50, 0.48, 0.47, 0.50, 0.65,   # 00-05
    0.95, 1.30, 1.25, 0.95, 0.85, 0.85,   # 06-11
    0.95, 0.90, 0.85, 0.90, 1.05, 1.35,   # 12-17
    1.70, 1.85, 1.80, 1.55, 1.10, 0.75,   # 18-23
])


def make_households(rng: np.random.Generator) -> pd.DataFrame:
    """Create the household attribute table."""
    ids = [f"H{i:03d}" for i in range(1, N_HOUSEHOLDS + 1)]
    dwelling = rng.choice(["apartment", "house"], size=N_HOUSEHOLDS, p=[0.5, 0.5])
    residents = rng.integers(1, 6, size=N_HOUSEHOLDS)
    area = np.where(
        dwelling == "house",
        rng.uniform(90, 180, size=N_HOUSEHOLDS),
        rng.uniform(35, 90, size=N_HOUSEHOLDS),
    ).round(0)
    # Half of the households heat electrically (shuffled so it is not tied to order)
    heating = rng.permutation([1, 1, 1, 0, 0, 0][:N_HOUSEHOLDS])
    return pd.DataFrame({
        "household_id": ids,
        "dwelling_type": dwelling,
        "n_residents": residents,
        "floor_area_m2": area.astype(int),
        "has_electric_heating": heating,
    })


def make_weather(rng: np.random.Generator, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Hourly temperature: winter level + daily cycle + slowly drifting weather."""
    n_hours = len(timestamps)
    hour = timestamps.hour.to_numpy()

    # Day-to-day drift as an AR(1) process, one value per day
    daily = np.zeros(N_DAYS)
    for d in range(1, N_DAYS):
        daily[d] = 0.8 * daily[d - 1] + rng.normal(0, 1.8)
    drift = np.repeat(daily, 24)[:n_hours]

    # Daily cycle: coldest early morning, warmest mid-afternoon
    cycle = 3.5 * np.sin(2 * np.pi * (hour - 9) / 24)
    noise = rng.normal(0, 0.6, size=n_hours)
    temp = 2.0 + drift + cycle + noise
    return pd.DataFrame({
        "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
        "temperature_c": temp.round(1),
    })


def make_energy(rng: np.random.Generator, households: pd.DataFrame,
                weather: pd.DataFrame, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Hourly kWh per household."""
    n_hours = len(timestamps)
    hour = timestamps.hour.to_numpy()
    is_weekend = (timestamps.dayofweek >= 5).to_numpy()
    temp = weather["temperature_c"].to_numpy()
    cold = np.clip(16.0 - temp, 0, None)  # degrees below the heating threshold

    # Weekend: more usage during the day, slightly later evening peak
    weekend_mult = np.where(is_weekend, 1.12, 1.0)
    heating_shape = 0.6 + 0.4 * HOURLY_PROFILE[hour] / HOURLY_PROFILE.max()

    frames = []
    for _, hh in households.iterrows():
        base = 0.15 + 0.12 * hh["n_residents"] + 0.002 * hh["floor_area_m2"]
        load = base * HOURLY_PROFILE[hour] * weekend_mult

        if hh["has_electric_heating"] == 1:
            load = load + 0.045 * cold * (hh["floor_area_m2"] / 100) * heating_shape

        # Household-specific day-to-day level change plus hourly noise
        day_factor = np.repeat(rng.normal(1.0, 0.08, size=N_DAYS), 24)[:n_hours]
        noise = rng.lognormal(mean=0.0, sigma=0.15, size=n_hours)
        kwh = np.clip(load * day_factor * noise, 0.05, 8.0)

        frames.append(pd.DataFrame({
            "household_id": hh["household_id"],
            "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
            "kwh": kwh.round(3),
        }))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    rng = np.random.default_rng(SEED)
    timestamps = pd.date_range(START, periods=N_DAYS * 24, freq="h")

    households = make_households(rng)
    weather = make_weather(rng, timestamps)
    energy = make_energy(rng, households, weather, timestamps)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    import os
    os.makedirs(DATA_DIR, exist_ok=True)
    households.to_csv(DATA_DIR / "households.csv", index=False)
    weather.to_csv(DATA_DIR / "weather_hourly.csv", index=False)
    energy.to_csv(DATA_DIR / "energy_hourly.csv", index=False)

    for name, df in [("households", households), ("weather_hourly", weather),
                     ("energy_hourly", energy)]:
        print(f"{name}: {len(df)} rows, {df.shape[1]} columns -> data/{name}.csv")


if __name__ == "__main__":
    main()
