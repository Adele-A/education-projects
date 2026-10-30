"""Tests for the SYNTHETIC data generator (src/data_generator.py).

The tables are built in memory with the generator's own functions, so the
tests are fast and do not depend on files in data/ or on figures.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import data_generator as dg  # noqa: E402


def generate_tables():
    """Build the three tables exactly as dg.main() does, without writing files."""
    rng = np.random.default_rng(dg.SEED)
    timestamps = pd.date_range(dg.START, periods=dg.N_DAYS * 24, freq="h")
    households = dg.make_households(rng)
    weather = dg.make_weather(rng, timestamps)
    energy = dg.make_energy(rng, households, weather, timestamps)
    return households, weather, energy


@pytest.fixture(scope="module")
def tables():
    return generate_tables()


def test_columns_and_row_counts(tables):
    households, weather, energy = tables
    n_hours = dg.N_DAYS * 24
    assert list(households.columns) == [
        "household_id", "dwelling_type", "n_residents",
        "floor_area_m2", "has_electric_heating"]
    assert list(weather.columns) == ["timestamp", "temperature_c"]
    assert list(energy.columns) == ["household_id", "timestamp", "kwh"]
    assert len(households) == dg.N_HOUSEHOLDS
    assert len(weather) == n_hours
    assert len(energy) == dg.N_HOUSEHOLDS * n_hours


def test_household_value_ranges(tables):
    households, _, _ = tables
    assert households["household_id"].is_unique
    assert households["n_residents"].between(1, 5).all()
    assert set(households["dwelling_type"]) <= {"apartment", "house"}
    assert set(households["has_electric_heating"]) <= {0, 1}
    # Three households heat electrically, three do not
    assert households["has_electric_heating"].sum() == 3

    apartments = households[households["dwelling_type"] == "apartment"]
    houses = households[households["dwelling_type"] == "house"]
    assert apartments["floor_area_m2"].between(35, 90).all()
    assert houses["floor_area_m2"].between(90, 180).all()


def test_energy_values_and_keys(tables):
    households, weather, energy = tables
    assert energy["kwh"].notna().all()
    assert energy["kwh"].between(0.05, 8.0).all()
    # Every (household, timestamp) pair is unique and has a weather row
    assert not energy.duplicated(["household_id", "timestamp"]).any()
    assert energy["timestamp"].isin(weather["timestamp"]).all()
    assert energy["household_id"].isin(households["household_id"]).all()
    # Every household has a complete hourly series
    assert (energy.groupby("household_id").size() == dg.N_DAYS * 24).all()


def test_weather_is_complete_and_plausible(tables):
    _, weather, _ = tables
    assert weather["temperature_c"].notna().all()
    assert weather["timestamp"].is_unique
    assert weather["timestamp"].iloc[0] == "2025-01-01 00:00:00"
    assert weather["timestamp"].iloc[-1] == "2025-01-30 23:00:00"
    # Winter data: values should stay in a broad, sensible window
    assert weather["temperature_c"].between(-30, 30).all()


def test_generation_is_reproducible(tables):
    again = generate_tables()
    for first, second in zip(tables, again):
        pd.testing.assert_frame_equal(first, second)


def test_main_writes_csv_files(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dg, "DATA_DIR", tmp_path / "data")
    dg.main()
    out = capsys.readouterr().out
    for name, rows in [("households", dg.N_HOUSEHOLDS),
                       ("weather_hourly", dg.N_DAYS * 24),
                       ("energy_hourly", dg.N_HOUSEHOLDS * dg.N_DAYS * 24)]:
        path = tmp_path / "data" / f"{name}.csv"
        assert path.exists()
        assert len(pd.read_csv(path)) == rows
        assert f"{name}: {rows} rows" in out
