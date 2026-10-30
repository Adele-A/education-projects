"""Tests for the helper functions in src/utils.py (SYNTHETIC energy data).

A small dataset is generated in memory with the project's own generator,
so the tests need no files in data/ and no figures.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import data_generator as dg  # noqa: E402
import utils  # noqa: E402


@pytest.fixture(scope="module")
def featured():
    """Full feature table (before dropping incomplete rows)."""
    rng = np.random.default_rng(dg.SEED)
    timestamps = pd.date_range(dg.START, periods=dg.N_DAYS * 24, freq="h")
    households = dg.make_households(rng)
    weather = dg.make_weather(rng, timestamps)
    energy = dg.make_energy(rng, households, weather, timestamps)
    df = utils.build_dataset(households, weather, energy)
    return utils.add_features(df)


def test_metrics_on_known_values():
    y_true, y_pred = [2.0, 4.0], [1.0, 5.0]
    assert utils.mae(y_true, y_pred) == pytest.approx(1.0)
    assert utils.rmse(y_true, y_pred) == pytest.approx(1.0)
    assert utils.wape(y_true, y_pred) == pytest.approx(2.0 / 6.0)
    # ss_res = 2 and ss_tot = 2, so R2 = 0
    assert utils.r2(y_true, y_pred) == pytest.approx(0.0)

    perfect = utils.evaluate(y_true, y_true)
    assert perfect == {"mae": 0.0, "rmse": 0.0, "wape": 0.0, "r2": 1.0}


def test_undefined_metrics_raise():
    with pytest.raises(ValueError):
        utils.wape([0.0, 0.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        utils.r2([3.0, 3.0, 3.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        utils.improvement_pct(0.0, 0.5)


def test_improvement_pct():
    assert utils.improvement_pct(2.0, 1.0) == pytest.approx(50.0)
    assert utils.improvement_pct(1.0, 1.0) == pytest.approx(0.0)
    # A worse model gives a negative reduction
    assert utils.improvement_pct(1.0, 1.5) == pytest.approx(-50.0)


def test_add_features_columns_and_lags(featured):
    assert set(utils.FEATURE_COLUMNS + [utils.TARGET]) <= set(featured.columns)

    hh = featured[featured["household_id"] == "H001"].reset_index(drop=True)
    # lag_24 / lag_168 are the same hour one day / one week earlier
    assert hh.loc[30, "lag_24"] == hh.loc[6, "kwh"]
    assert hh.loc[200, "lag_168"] == hh.loc[32, "kwh"]
    # The first rows have no lag available
    assert hh.loc[:23, "lag_24"].isna().all()
    assert hh.loc[:167, "lag_168"].isna().all()
    # prev_day_mean of day 2 equals the mean of day 1
    day1_mean = hh.loc[24:47, "kwh"].mean()
    assert hh.loc[50, "prev_day_mean"] == pytest.approx(day1_mean)


def test_calendar_and_temperature_features(featured):
    # 2025-01-01 was a Wednesday (day_of_week = 2) and 2025-01-04 a Saturday
    first = featured[featured["timestamp"] == "2025-01-01 00:00:00"]
    assert (first["day_of_week"] == 2).all()
    assert (first["is_weekend"] == 0).all()
    saturday = featured[featured["timestamp"] == "2025-01-04 12:00:00"]
    assert (saturday["is_weekend"] == 1).all()

    assert np.allclose(featured["hour_sin"] ** 2 + featured["hour_cos"] ** 2, 1.0)
    assert (featured["cold_degrees"] >= 0).all()
    expected = (utils.HEATING_THRESHOLD_C - featured["temperature_c"]).clip(lower=0)
    assert np.allclose(featured["cold_degrees"], expected)
    # Heating interaction is zero for households without electric heating
    no_heat = featured[featured["has_electric_heating"] == 0]
    assert (no_heat["heating_x_cold"] == 0).all()


def test_drop_incomplete_removes_first_week(featured):
    data = utils.drop_incomplete(featured)
    n_households = featured["household_id"].nunique()
    assert len(data) == n_households * (dg.N_DAYS * 24 - utils.LAG_WEEK)
    assert data[["lag_24", "lag_168", "prev_day_mean"]].notna().all().all()
    assert data["date"].min() == pd.Timestamp("2025-01-08")


def test_chronological_split_has_no_leakage(featured):
    data = utils.drop_incomplete(featured)
    test_days = 6
    train, test = utils.chronological_split(data, test_days)

    assert len(train) + len(test) == len(data)
    assert train["date"].max() < test["date"].min()
    assert test["date"].nunique() == test_days
    assert test["date"].max() == data["date"].max()
    # Each household appears in both sets with whole days
    n_households = data["household_id"].nunique()
    assert len(test) == n_households * test_days * 24
