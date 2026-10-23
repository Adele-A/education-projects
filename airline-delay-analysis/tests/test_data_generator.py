"""Tests for the SYNTHETIC data generator (src/data_generator.py).

The tests build the tables in memory with the project's fixed seed, so they
need neither the data/ folder nor any figures.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import data_generator as gen  # noqa: E402

EXPECTED_FLIGHT_COLUMNS = [
    "flight_id", "flight_date", "month", "day_of_week", "carrier_code",
    "origin", "destination", "scheduled_dep_hour", "distance_miles",
    "weather_severity", "dep_delay_min", "arr_delay_min", "is_delayed",
]


@pytest.fixture(scope="module")
def flights():
    """Flights table generated with the project seed."""
    return gen.build_flights(np.random.default_rng(gen.SEED))


def test_flights_columns_and_row_count(flights):
    assert list(flights.columns) == EXPECTED_FLIGHT_COLUMNS
    assert len(flights) == gen.N_FLIGHTS
    assert flights["flight_id"].is_unique


def test_flights_value_ranges(flights):
    assert flights["scheduled_dep_hour"].between(5, 22).all()
    assert flights["weather_severity"].between(0, 10).all()
    assert flights["dep_delay_min"].between(-15, 300).all()
    assert flights["arr_delay_min"].between(-30, 320).all()
    assert flights["month"].between(1, 12).all()
    assert (flights["distance_miles"] > 0).all()
    dates = pd.to_datetime(flights["flight_date"])
    assert dates.min() >= pd.Timestamp("2025-01-01")
    assert dates.max() <= pd.Timestamp("2025-12-31")


def test_flights_logic_and_keys(flights):
    # Destination differs from origin, target follows the 15-minute rule
    assert (flights["origin"] != flights["destination"]).all()
    expected = (flights["arr_delay_min"] >= gen.DELAY_THRESHOLD_MIN).astype(int)
    assert (flights["is_delayed"] == expected).all()
    assert set(flights["is_delayed"].unique()) <= {0, 1}
    # Foreign keys point to the reference tables
    assert set(flights["origin"]) <= set(gen.build_airports()["airport_code"])
    assert set(flights["carrier_code"]) <= set(gen.build_carriers()["carrier_code"])


def test_generation_is_reproducible(flights):
    again = gen.build_flights(np.random.default_rng(gen.SEED))
    pd.testing.assert_frame_equal(flights, again)


def test_reference_tables():
    airports = gen.build_airports()
    carriers = gen.build_carriers()
    assert len(airports) == 12
    assert len(carriers) == 6
    assert airports["airport_code"].is_unique
    assert carriers["carrier_code"].is_unique
    assert "NA" in set(carriers["carrier_code"])  # valid code, not missing
    assert airports["congestion_factor"].between(0.85, 1.40).all()


def test_haversine_miles():
    assert gen.haversine_miles(40.0, -75.0, 40.0, -75.0) == pytest.approx(0.0)
    d_ab = gen.haversine_miles(40.64, -73.78, 33.94, -118.41)
    d_ba = gen.haversine_miles(33.94, -118.41, 40.64, -73.78)
    assert d_ab == pytest.approx(d_ba)
    assert 2300 < d_ab < 2600  # roughly New York to Los Angeles


def test_main_writes_csv_files(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "DATA_DIR", tmp_path)
    gen.main()
    for name, rows in [("airports.csv", 12), ("carriers.csv", 6),
                       ("flights.csv", gen.N_FLIGHTS)]:
        saved = pd.read_csv(tmp_path / name, keep_default_na=False)
        assert len(saved) == rows
