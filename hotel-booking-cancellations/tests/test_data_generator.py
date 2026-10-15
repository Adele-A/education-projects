"""Tests for the SYNTHETIC data generator (src/data_generator.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

import data_generator as dg

N_GUESTS = 200
N_BOOKINGS = 400


@pytest.fixture(scope="module")
def tables():
    """Small guests and bookings tables generated with the project seed."""
    rng = np.random.default_rng(dg.SEED)
    guests = dg.generate_guests(rng, N_GUESTS)
    bookings = dg.generate_bookings(rng, N_BOOKINGS, guests)
    return guests, bookings


def test_guests_columns_and_rows(tables):
    guests, _ = tables
    assert list(guests.columns) == [
        "guest_id", "country", "age_group", "previous_stays", "previous_cancellations",
    ]
    assert len(guests) == N_GUESTS
    assert guests["guest_id"].is_unique


def test_guest_history_ranges(tables):
    guests, _ = tables
    assert guests["previous_stays"].between(0, 8).all()
    assert (guests["previous_cancellations"] >= 0).all()
    # A guest cannot have canceled more bookings than earlier stays.
    assert (guests["previous_cancellations"] <= guests["previous_stays"]).all()


def test_bookings_columns_and_rows(tables):
    _, bookings = tables
    expected = [
        "booking_id", "guest_id", "hotel_type", "arrival_date", "arrival_month",
        "lead_time_days", "stay_nights_weekend", "stay_nights_week", "adults",
        "children", "meal_plan", "market_segment", "deposit_type", "room_type",
        "adr", "special_requests", "parking_spaces", "booking_changes", "is_canceled",
    ]
    assert list(bookings.columns) == expected
    assert len(bookings) == N_BOOKINGS
    assert bookings["booking_id"].is_unique
    assert not bookings.isna().any().any()


def test_bookings_value_ranges(tables):
    _, bookings = tables
    assert bookings["lead_time_days"].between(0, 450).all()
    assert bookings["adr"].between(30, 400).all()
    assert bookings["adults"].between(1, 3).all()
    assert bookings["children"].between(0, 2).all()
    assert bookings["special_requests"].between(0, 4).all()
    assert bookings["booking_changes"].between(0, 5).all()
    assert set(bookings["parking_spaces"]).issubset({0, 1})
    assert set(bookings["is_canceled"]).issubset({0, 1})
    assert set(bookings["deposit_type"]).issubset(
        {"No Deposit", "Non Refundable", "Refundable"}
    )
    # At least one night per booking.
    total_nights = bookings["stay_nights_weekend"] + bookings["stay_nights_week"]
    assert (total_nights >= 1).all()


def test_dates_and_foreign_keys(tables):
    guests, bookings = tables
    dates = pd.to_datetime(bookings["arrival_date"])
    assert dates.min() >= pd.Timestamp("2024-01-01")
    assert dates.max() <= pd.Timestamp("2025-12-31")
    assert (dates.dt.month == bookings["arrival_month"]).all()
    assert bookings["guest_id"].isin(guests["guest_id"]).all()


def test_generation_is_reproducible():
    def make():
        rng = np.random.default_rng(dg.SEED)
        g = dg.generate_guests(rng, 50)
        return g, dg.generate_bookings(rng, 80, g)

    g1, b1 = make()
    g2, b2 = make()
    pd.testing.assert_frame_equal(g1, g2)
    pd.testing.assert_frame_equal(b1, b2)


def test_invalid_inputs_raise():
    rng = np.random.default_rng(dg.SEED)
    with pytest.raises(ValueError):
        dg.generate_guests(rng, 0)
    guests = dg.generate_guests(rng, 10)
    with pytest.raises(ValueError):
        dg.generate_bookings(rng, 0, guests)
    with pytest.raises(ValueError):
        dg.generate_bookings(rng, 5, guests.iloc[0:0])
    with pytest.raises(ValueError):
        dg.generate_bookings(rng, 5, guests.drop(columns=["previous_stays"]))
