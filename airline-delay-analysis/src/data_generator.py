"""Generate a SYNTHETIC airline delay dataset.

All data is produced by this script with a fixed random seed (42).
It does not describe real flights or real airlines.

Output tables (CSV, saved to <project root>/data/):
    airports.csv  - airport reference table
    carriers.csv  - carrier reference table
    flights.csv   - one row per flight

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
N_FLIGHTS = 5000
DELAY_THRESHOLD_MIN = 15  # arrival delay >= 15 min counts as "delayed"

# code, city, region, latitude, longitude, congestion_factor, weather_exposure
AIRPORTS = [
    ("ATL", "Atlanta", "South", 33.64, -84.43, 1.30, 0.9),
    ("ORD", "Chicago", "Midwest", 41.98, -87.90, 1.40, 1.4),
    ("DFW", "Dallas", "South", 32.90, -97.04, 1.15, 1.0),
    ("DEN", "Denver", "West", 39.86, -104.67, 1.10, 1.3),
    ("LAX", "Los Angeles", "West", 33.94, -118.41, 1.25, 0.5),
    ("SFO", "San Francisco", "West", 37.62, -122.38, 1.30, 0.8),
    ("SEA", "Seattle", "West", 47.45, -122.31, 1.00, 1.1),
    ("JFK", "New York", "Northeast", 40.64, -73.78, 1.40, 1.2),
    ("BOS", "Boston", "Northeast", 42.36, -71.01, 1.15, 1.3),
    ("MIA", "Miami", "South", 25.79, -80.29, 1.05, 1.1),
    ("PHX", "Phoenix", "West", 33.43, -112.01, 0.85, 0.4),
    ("MSP", "Minneapolis", "Midwest", 44.88, -93.22, 0.90, 1.3),
]

# code, name (fictional), type, delay propensity (log-odds shift, hidden driver)
CARRIERS = [
    ("NA", "Northern Air", "legacy", -0.15),
    ("SJ", "SkyJet", "low-cost", 0.30),
    ("PW", "Pacific Wings", "legacy", -0.30),
    ("EC", "EconoFly", "low-cost", 0.45),
    ("RL", "Regional Link", "regional", 0.20),
    ("AV", "Avanti Airways", "legacy", 0.00),
]

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]


def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles between two points (degrees)."""
    earth_radius_miles = 3958.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * earth_radius_miles * np.arcsin(np.sqrt(a))


def build_airports():
    """Airport reference table."""
    return pd.DataFrame(
        [a[:6] for a in AIRPORTS],
        columns=["airport_code", "city", "region", "latitude",
                 "longitude", "congestion_factor"],
    )


def build_carriers():
    """Carrier reference table."""
    return pd.DataFrame(
        [c[:3] for c in CARRIERS],
        columns=["carrier_code", "carrier_name", "carrier_type"],
    )


def build_flights(rng):
    """Generate the flights table with realistic delay structure."""
    n = N_FLIGHTS
    codes = [a[0] for a in AIRPORTS]
    coords = {a[0]: (a[3], a[4]) for a in AIRPORTS}
    congestion = {a[0]: a[5] for a in AIRPORTS}
    exposure = {a[0]: a[6] for a in AIRPORTS}
    carrier_codes = [c[0] for c in CARRIERS]
    carrier_shift = {c[0]: c[3] for c in CARRIERS}

    # Dates: random days across 2025, sorted so flight_id follows time
    day_offsets = np.sort(rng.integers(0, 365, size=n))
    dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(day_offsets, unit="D")

    # Carriers and airports (origin != destination)
    carrier = rng.choice(carrier_codes, size=n,
                         p=[0.24, 0.20, 0.16, 0.14, 0.12, 0.14])
    origin = rng.choice(codes, size=n)
    destination = np.empty(n, dtype=object)
    for i, o in enumerate(origin):
        destination[i] = rng.choice([c for c in codes if c != o])

    # Scheduled departure hour: morning and evening peaks, 5..22
    hours = np.arange(5, 23)
    hour_w = np.array([3, 6, 8, 7, 6, 5, 5, 5, 5, 6, 7, 8, 8, 7, 6, 4, 3, 2],
                      dtype=float)
    hour = rng.choice(hours, size=n, p=hour_w / hour_w.sum())

    # Distance from airport coordinates
    distance = np.array([
        haversine_miles(*coords[o], *coords[d])
        for o, d in zip(origin, destination)
    ]).round().astype(int)

    # Weather severity 0-10: worse in winter, scaled by airport exposure
    month = dates.month.to_numpy()
    winter = np.isin(month, [12, 1, 2])
    summer_storm = np.isin(month, [6, 7, 8])
    season = np.where(winter, 1.8, np.where(summer_storm, 1.2, 1.0))
    expo = np.array([exposure[o] for o in origin])
    weather = np.clip(rng.exponential(1.0, size=n) * season * expo, 0, 10)
    weather = weather.round(1)

    # Probability that the flight suffers a disruption (logistic model)
    cong = np.array([congestion[o] for o in origin])
    c_shift = np.array([carrier_shift[c] for c in carrier])
    weekday = dates.dayofweek.to_numpy()
    weekend = (weekday >= 5).astype(float)
    logit = (-2.0
             + c_shift
             + 1.6 * (cong - 1.0)
             + 0.09 * (hour - 5)          # delays accumulate during the day
             + 0.28 * weather
             + 0.0002 * distance
             - 0.15 * weekend)
    p_event = 1.0 / (1.0 + np.exp(-logit))
    event = rng.random(n) < p_event

    # Departure delay: small noise normally, long exponential tail on events
    normal_delay = rng.normal(-3.0, 6.0, size=n)
    event_delay = DELAY_THRESHOLD_MIN + rng.exponential(30 + 5 * weather)
    dep_delay = np.where(event, event_delay, normal_delay)
    dep_delay = np.clip(np.round(dep_delay), -15, 300).astype(int)

    # Arrival delay: departure delay plus en-route noise (can recover time)
    arr_delay = dep_delay + rng.normal(-2.0, 8.0, size=n)
    arr_delay = np.clip(np.round(arr_delay), -30, 320).astype(int)

    flights = pd.DataFrame({
        "flight_id": [f"F{i + 1:05d}" for i in range(n)],
        "flight_date": dates.strftime("%Y-%m-%d"),
        "month": month.astype(int),
        "day_of_week": [DAY_NAMES[d] for d in weekday],
        "carrier_code": carrier,
        "origin": origin,
        "destination": destination,
        "scheduled_dep_hour": hour.astype(int),
        "distance_miles": distance,
        "weather_severity": weather,
        "dep_delay_min": dep_delay,
        "arr_delay_min": arr_delay,
    })
    flights["is_delayed"] = (
        flights["arr_delay_min"] >= DELAY_THRESHOLD_MIN
    ).astype(int)
    return flights


def main():
    """Generate all tables and save them to data/."""
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    airports = build_airports()
    carriers = build_carriers()
    flights = build_flights(rng)

    airports.to_csv(DATA_DIR / "airports.csv", index=False)
    carriers.to_csv(DATA_DIR / "carriers.csv", index=False)
    flights.to_csv(DATA_DIR / "flights.csv", index=False)

    print("Synthetic data saved to:", DATA_DIR)
    print(f"airports.csv: {len(airports)} rows")
    print(f"carriers.csv: {len(carriers)} rows")
    print(f"flights.csv:  {len(flights)} rows")


if __name__ == "__main__":
    main()
