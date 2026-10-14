"""Generate SYNTHETIC hotel booking data.

Creates two tables in <project root>/data/:
    guests.csv   - one row per guest
    bookings.csv - one row per booking (target: is_canceled)

All data is artificial. Cancellations are drawn from a hand-designed logistic
model so that the dataset contains plausible, discoverable drivers.

Run from the project root:  python src/data_generator.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

SEED = 42
N_GUESTS = 3500
N_BOOKINGS = 5000

# Columns that generate_bookings reads from the guests table.
REQUIRED_GUEST_COLUMNS = ("guest_id", "previous_stays", "previous_cancellations")


def generate_guests(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Create the guests table with travel history."""
    # Validate the size up front: a non-positive n would produce an empty or
    # invalid table and fail later with a less helpful message.
    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n}")
    guest_id = [f"G{i:05d}" for i in range(1, n + 1)]
    country = rng.choice(
        ["USA", "UK", "Germany", "France", "Spain", "Italy", "Netherlands", "Other"],
        size=n,
        p=[0.22, 0.14, 0.12, 0.12, 0.10, 0.08, 0.06, 0.16],
    )
    age_group = rng.choice(
        ["18-29", "30-44", "45-59", "60+"], size=n, p=[0.20, 0.40, 0.28, 0.12]
    )
    previous_stays = np.clip(rng.poisson(0.5, size=n), 0, 8)
    # Each earlier stay had a small chance of having been a canceled booking.
    previous_cancellations = rng.binomial(previous_stays, 0.15)
    return pd.DataFrame(
        {
            "guest_id": guest_id,
            "country": country,
            "age_group": age_group,
            "previous_stays": previous_stays,
            "previous_cancellations": previous_cancellations,
        }
    )


def generate_bookings(
    rng: np.random.Generator, n: int, guests: pd.DataFrame
) -> pd.DataFrame:
    """Create the bookings table, including the cancellation target."""
    # Bookings are linked to guests by random index, which fails with an
    # obscure error when there are no guests, so check the inputs first.
    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n}")
    if len(guests) == 0:
        raise ValueError("guests table is empty; cannot link bookings to guests")
    # A guests table without the expected columns would otherwise raise a bare
    # KeyError later, so report exactly which columns are missing.
    missing = [c for c in REQUIRED_GUEST_COLUMNS if c not in guests.columns]
    if missing:
        raise ValueError(f"guests table is missing required columns: {missing}")
    # Link each booking to a guest.
    g_idx = rng.integers(0, len(guests), size=n)
    guest_id = guests["guest_id"].to_numpy()[g_idx]
    prev_stays = guests["previous_stays"].to_numpy()[g_idx]
    prev_canc = guests["previous_cancellations"].to_numpy()[g_idx]
    is_repeat = (prev_stays > 0).astype(int)

    hotel_type = rng.choice(["City Hotel", "Resort Hotel"], size=n, p=[0.62, 0.38])
    is_city = hotel_type == "City Hotel"

    # Arrival dates over two years, with a summer peak.
    days = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    month_weight = {1: 0.7, 2: 0.7, 3: 0.9, 4: 1.0, 5: 1.1, 6: 1.3,
                    7: 1.5, 8: 1.5, 9: 1.1, 10: 1.0, 11: 0.8, 12: 0.9}
    w = np.array([month_weight[m] for m in days.month], dtype=float)
    arrival = days[rng.choice(len(days), size=n, p=w / w.sum())]
    arrival_month = arrival.month.to_numpy()

    lead_time = np.clip(rng.gamma(1.5, 60.0, size=n), 0, 450).astype(int)

    # Stay length: resorts have slightly longer stays.
    weekend = np.clip(rng.poisson(np.where(is_city, 0.8, 1.1)), 0, 6)
    week = np.clip(rng.poisson(np.where(is_city, 2.2, 3.0)), 0, 14)
    week = np.where(weekend + week == 0, 1, week)  # at least one night

    adults = rng.choice([1, 2, 3], size=n, p=[0.20, 0.72, 0.08])
    children = rng.choice([0, 1, 2], size=n, p=[0.92, 0.06, 0.02])
    meal_plan = rng.choice(
        ["Bed & Breakfast", "Half Board", "Full Board", "Self Catering"],
        size=n,
        p=[0.68, 0.14, 0.03, 0.15],
    )
    market_segment = rng.choice(
        ["Online TA", "Offline TA", "Direct", "Corporate", "Groups"],
        size=n,
        p=[0.50, 0.20, 0.12, 0.08, 0.10],
    )
    deposit_type = rng.choice(
        ["No Deposit", "Non Refundable", "Refundable"], size=n, p=[0.88, 0.10, 0.02]
    )
    room_type = rng.choice(["A", "B", "C", "D"], size=n, p=[0.55, 0.20, 0.15, 0.10])

    # Average daily rate: depends on hotel, season and room type, plus noise.
    room_uplift = np.array([{"A": 1.0, "B": 1.1, "C": 1.25, "D": 1.5}[r] for r in room_type])
    season = 1.0 + 0.25 * np.isin(arrival_month, [6, 7, 8])
    adr = np.where(is_city, 105.0, 95.0) * season * room_uplift * rng.lognormal(0, 0.2, n)
    adr = np.round(np.clip(adr, 30, 400), 2)

    special_requests = np.clip(rng.poisson(0.6 + 0.3 * is_repeat), 0, 4)
    parking_spaces = rng.binomial(1, np.where(is_city, 0.05, 0.12))
    booking_changes = np.clip(rng.poisson(0.3, size=n), 0, 5)

    # Hand-designed logistic model for the cancellation probability.
    # Note: non-refundable bookings are given a high cancellation effect on purpose
    # (a known quirk of some real booking data); this is a design choice of the simulator.
    z = (
        -1.5
        + 0.012 * (lead_time - 90)
        + 0.35 * is_city
        + 0.60 * (market_segment == "Online TA")
        + 0.10 * (market_segment == "Offline TA")
        - 0.80 * (market_segment == "Direct")
        - 0.90 * (market_segment == "Corporate")
        + 1.50 * (deposit_type == "Non Refundable")
        - 0.20 * (deposit_type == "Refundable")
        - 0.45 * special_requests
        - 0.90 * parking_spaces
        - 0.35 * np.minimum(booking_changes, 3)
        + 0.80 * np.minimum(prev_canc, 3)
        - 0.50 * is_repeat
        + 0.004 * (adr - 100)
        + 0.25 * (children > 0)
    )
    p_cancel = 1.0 / (1.0 + np.exp(-z))
    is_canceled = rng.binomial(1, p_cancel)

    return pd.DataFrame(
        {
            "booking_id": [f"B{i:05d}" for i in range(1, n + 1)],
            "guest_id": guest_id,
            "hotel_type": hotel_type,
            "arrival_date": arrival.strftime("%Y-%m-%d"),
            "arrival_month": arrival_month,
            "lead_time_days": lead_time,
            "stay_nights_weekend": weekend,
            "stay_nights_week": week,
            "adults": adults,
            "children": children,
            "meal_plan": meal_plan,
            "market_segment": market_segment,
            "deposit_type": deposit_type,
            "room_type": room_type,
            "adr": adr,
            "special_requests": special_requests,
            "parking_spaces": parking_spaces,
            "booking_changes": booking_changes,
            "is_canceled": is_canceled,
        }
    )


def main() -> None:
    rng = np.random.default_rng(SEED)
    guests = generate_guests(rng, N_GUESTS)
    bookings = generate_bookings(rng, N_BOOKINGS, guests)

    os.makedirs(DATA_DIR, exist_ok=True)
    guests.to_csv(DATA_DIR / "guests.csv", index=False)
    bookings.to_csv(DATA_DIR / "bookings.csv", index=False)

    print(f"Saved {len(guests)} rows to {DATA_DIR / 'guests.csv'}")
    print(f"Saved {len(bookings)} rows to {DATA_DIR / 'bookings.csv'}")
    print(f"Overall cancellation rate: {bookings['is_canceled'].mean():.3f}")


if __name__ == "__main__":
    main()
