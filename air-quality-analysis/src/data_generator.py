"""Generate SYNTHETIC air quality data for the air-quality-analysis project.

All values are simulated with numpy (fixed seed 42); nothing here is real
measurement data.

Outputs (relative to the project root):
    data/stations.csv
    data/daily_measurements.csv
    data/notes.csv
    data/comments.csv

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
START_DATE = "2022-01-01"
END_DATE = "2024-12-31"

# Station metadata: type drives the baseline pollution level.
STATIONS = pd.DataFrame(
    {
        "station_id": ["S01", "S02", "S03", "S04"],
        "station_name": ["Central Avenue", "Old Town", "Riverside Industrial", "Green Hills"],
        "station_type": ["traffic", "urban_background", "industrial", "suburban"],
        "latitude": [50.062, 50.071, 50.041, 50.095],
        "longitude": [19.938, 19.925, 19.981, 19.902],
        "elevation_m": [212, 205, 198, 265],
    }
)

# Free-text project notes (English only).
NOTES = pd.DataFrame(
    {
        "note_id": [1],
        "note": ["Test note for checking the data pipeline"],
    }
)

# Free-text project comments (English only).
COMMENTS = pd.DataFrame(
    {
        "comment_id": [1],
        "comment": ["Test comment for checking the data pipeline"],
    }
)

# Baseline levels (ug/m3) by station type.
BASE_PM25 = {"traffic": 24.0, "urban_background": 18.0, "industrial": 22.0, "suburban": 12.0}
BASE_NO2 = {"traffic": 45.0, "urban_background": 32.0, "industrial": 36.0, "suburban": 18.0}

MISSING_RATE_PM25 = 0.02
MISSING_RATE_OTHER = 0.015


def ar1(rng, n, phi, sigma):
    """Autocorrelated noise: x[t] = phi * x[t-1] + N(0, sigma)."""
    eps = rng.normal(0.0, sigma, n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


def generate_city_weather(rng, dates):
    """City-wide daily weather shared by all stations."""
    n = len(dates)
    doy = dates.dayofyear.values

    # Annual temperature cycle, peak in mid-July, plus autocorrelated noise.
    temp = 11.0 + 10.5 * np.sin(2 * np.pi * (doy - 105) / 365.25) + ar1(rng, n, 0.7, 2.5)

    wind = np.clip(rng.gamma(shape=2.5, scale=1.4, size=n), 0.2, 15.0)

    rain_day = rng.random(n) < 0.28
    precip = np.where(rain_day, rng.exponential(5.0, n), 0.0)
    precip = np.clip(precip, 0.0, 80.0)

    humidity = 72.0 - 0.8 * (temp - 11.0) + 12.0 * rain_day + rng.normal(0, 6, n)
    humidity = np.clip(humidity, 25.0, 100.0)

    pressure = np.clip(1013.0 + ar1(rng, n, 0.8, 4.0), 980.0, 1045.0)

    return {
        "temp": temp,
        "wind": wind,
        "precip": precip,
        "humidity": humidity,
        "pressure": pressure,
    }


def generate_station_series(rng, dates, weather, station_type, temp_offset):
    """Daily pollutant and weather values for one station."""
    n = len(dates)
    temp = weather["temp"] + temp_offset + rng.normal(0, 0.5, n)
    wind = np.clip(weather["wind"] * rng.normal(1.0, 0.08, n), 0.2, 15.0)
    humidity = np.clip(weather["humidity"] + rng.normal(0, 2.0, n), 25.0, 100.0)
    precip = weather["precip"]
    pressure = weather["pressure"]

    # Heating season: colder days -> more emissions. Wind dilutes, rain washes out.
    heating = 1.0 + 0.04 * np.clip(15.0 - temp, 0, None)
    washout = np.where(precip > 1.0, 0.7, 1.0)
    city_noise = np.exp(ar1(rng, n, 0.6, 0.25))   # multi-day pollution episodes
    station_noise = np.exp(rng.normal(0, 0.12, n))

    pm25 = (BASE_PM25[station_type] * heating * np.exp(-0.15 * (wind - 3.5))
            * washout * city_noise * station_noise)
    pm25 = np.clip(pm25, 1.0, 250.0)

    # PM10 contains PM2.5 plus a coarse fraction, so it is always above PM2.5.
    pm10 = pm25 * (1.4 + rng.normal(0, 0.1, n)) + rng.gamma(2.0, 3.0, n)
    pm10 = np.clip(np.maximum(pm10, pm25 + 1.0), 2.0, 400.0)

    no2 = (BASE_NO2[station_type] * (1.0 + 0.02 * np.clip(15.0 - temp, 0, None))
           * np.exp(-0.10 * (wind - 3.5)) * np.exp(rng.normal(0, 0.2, n)))
    no2 = np.clip(no2, 2.0, 200.0)

    # Ozone rises with temperature and is reduced by NO2 titration.
    o3 = 30.0 + 2.0 * (temp - 11.0) + 2.0 * (wind - 3.5) - 0.25 * (no2 - 30.0) + rng.normal(0, 8, n)
    o3 = np.clip(o3, 2.0, 180.0)

    return pd.DataFrame(
        {
            "date": dates,
            "pm25": pm25,
            "pm10": pm10,
            "no2": no2,
            "o3": o3,
            "temperature_c": temp,
            "humidity_pct": humidity,
            "wind_speed_ms": wind,
            "precipitation_mm": precip,
            "pressure_hpa": pressure,
        }
    )


def generate_measurements(rng):
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    weather = generate_city_weather(rng, dates)

    frames = []
    for i, row in STATIONS.iterrows():
        # Suburban/higher stations are slightly cooler.
        temp_offset = -0.8 if row["station_type"] == "suburban" else 0.0
        df = generate_station_series(rng, dates, weather, row["station_type"], temp_offset)
        df.insert(1, "station_id", row["station_id"])
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    # Missing values (sensor outages), missing completely at random.
    for col in ["pm25"]:
        data.loc[rng.random(len(data)) < MISSING_RATE_PM25, col] = np.nan
    for col in ["pm10", "no2", "o3"]:
        data.loc[rng.random(len(data)) < MISSING_RATE_OTHER, col] = np.nan

    data = data.round(
        {"pm25": 1, "pm10": 1, "no2": 1, "o3": 1, "temperature_c": 1,
         "humidity_pct": 1, "wind_speed_ms": 1, "precipitation_mm": 1, "pressure_hpa": 1}
    )
    data["date"] = data["date"].dt.strftime("%Y-%m-%d")
    return data


def write_english_table(table, text_column, path):
    """Write a small free-text table, replacing any older version of the file.

    If a previous file contains non-English (non-ASCII) text, it is reported
    and then overwritten.
    """
    # Safety check: all project text must be English (ASCII).
    assert all(str(t).isascii() for t in table[text_column]), \
        f"{path.name}: text must be English only"

    if path.exists():
        old_text = path.read_text(encoding="utf-8", errors="replace")
        if not old_text.isascii():
            print(f"Found non-English text in old {path.name}; replacing it with English text.")
        path.unlink()  # always start from a clean file

    table.to_csv(path, index=False, encoding="utf-8")


def write_notes(path):
    """Write the English-only notes file, replacing any older version."""
    write_english_table(NOTES, "note", path)


def write_comments(path):
    """Write the English-only comments file, replacing any older version."""
    write_english_table(COMMENTS, "comment", path)


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(DATA_DIR, exist_ok=True)

    measurements = generate_measurements(rng)

    STATIONS.to_csv(DATA_DIR / "stations.csv", index=False)
    measurements.to_csv(DATA_DIR / "daily_measurements.csv", index=False)
    write_notes(DATA_DIR / "notes.csv")
    write_comments(DATA_DIR / "comments.csv")

    print(f"Saved {len(STATIONS)} rows to {DATA_DIR / 'stations.csv'}")
    print(f"Saved {len(measurements)} rows to {DATA_DIR / 'daily_measurements.csv'}")
    print(f"Saved {len(NOTES)} rows to {DATA_DIR / 'notes.csv'}")
    print(f"Saved {len(COMMENTS)} rows to {DATA_DIR / 'comments.csv'}")


if __name__ == "__main__":
    main()
