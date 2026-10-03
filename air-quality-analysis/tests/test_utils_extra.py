"""Additional unit tests for src/utils.py (small hand-made frames)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from utils import load_data, prepare_model_data, smearing_factor  # noqa: E402


def test_smearing_factor_zero_residuals_is_one():
    assert smearing_factor(np.zeros(5)) == pytest.approx(1.0)
    # Non-zero residuals give a factor above 1 (Jensen's inequality).
    assert smearing_factor([-0.5, 0.5]) > 1.0


def test_prepare_model_data_drops_missing_target():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
        "station_type": ["traffic", "suburban", "industrial"],
        "pm25": [20.0, np.nan, 15.0],
        "temperature_c": [1.0, 2.0, 3.0],
        "humidity_pct": [80.0, 70.0, 60.0],
        "wind_speed_ms": [2.0, 3.0, 4.0],
        "precipitation_mm": [0.0, 0.0, 2.0],
        "pressure_hpa": [1010.0, 1012.0, 1015.0],
    })
    rows, X, y = prepare_model_data(df, target="pm25")
    assert len(rows) == len(X) == len(y) == 2
    assert y.notna().all()


def test_load_data_missing_files_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path)
