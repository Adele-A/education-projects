"""Tests for the SYNTHETIC data generator (src/data_generator.py).

The data is generated in memory with the fixed seed; no files in data/ are
read or written (except a temporary folder for the notes test).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import data_generator as dg  # noqa: E402

EXPECTED_COLUMNS = [
    "date", "station_id", "pm25", "pm10", "no2", "o3", "temperature_c",
    "humidity_pct", "wind_speed_ms", "precipitation_mm", "pressure_hpa",
]


@pytest.fixture(scope="module")
def measurements():
    """Generate the full synthetic table once for all tests in this file."""
    return dg.generate_measurements(np.random.default_rng(dg.SEED))


def test_columns_and_row_count(measurements):
    assert list(measurements.columns) == EXPECTED_COLUMNS
    n_days = len(np.arange(np.datetime64(dg.START_DATE),
                           np.datetime64(dg.END_DATE) + 1))
    assert n_days == 1096
    assert len(measurements) == len(dg.STATIONS) * n_days


def test_keys_are_unique_and_valid(measurements):
    assert not measurements.duplicated(subset=["date", "station_id"]).any()
    assert set(measurements["station_id"]) == set(dg.STATIONS["station_id"])
    assert measurements["date"].min() == dg.START_DATE
    assert measurements["date"].max() == dg.END_DATE


def test_value_ranges_respect_clipping(measurements):
    bounds = {
        "pm25": (1.0, 250.0), "pm10": (2.0, 400.0), "no2": (2.0, 200.0),
        "o3": (2.0, 180.0), "humidity_pct": (25.0, 100.0),
        "wind_speed_ms": (0.2, 15.0), "precipitation_mm": (0.0, 80.0),
        "pressure_hpa": (980.0, 1045.0),
    }
    for col, (low, high) in bounds.items():
        values = measurements[col].dropna()
        assert values.min() >= low, col
        assert values.max() <= high, col


def test_pm10_not_below_pm25_and_missing_share_is_small(measurements):
    both = measurements[["pm25", "pm10"]].dropna()
    assert (both["pm10"] > both["pm25"]).all()
    # Missing values exist by design but only for a small share of rows.
    share = measurements["pm25"].isna().mean()
    assert 0.0 < share < 0.05
    assert measurements[["date", "station_id", "temperature_c"]].notna().all().all()


def test_generation_is_reproducible(measurements):
    again = dg.generate_measurements(np.random.default_rng(dg.SEED))
    assert measurements.equals(again)


def test_write_notes_is_english_and_replaces_old_file(tmp_path):
    path = tmp_path / "notes.csv"
    path.write_text("note_id,note\n1,old text\n", encoding="utf-8")
    dg.write_notes(path)
    text = path.read_text(encoding="utf-8")
    assert text.isascii()
    assert "old text" not in text
    assert text.splitlines()[0] == "note_id,note"
