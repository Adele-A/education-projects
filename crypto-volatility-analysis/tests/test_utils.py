"""Tests for the analysis helpers in src/utils.py.

Small hand-made inputs with known answers are used where possible, plus the
synthetic tables built in memory by the generator. No files or figures are used.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

import data_generator as dg
import utils


def test_returns_wide_and_split_by_regime():
    """The wide matrix is date x asset and the regime split covers all days."""
    tables = dg.build_tables()
    prices = tables["daily_prices"].copy()
    regimes = tables["market_regimes"].copy()
    prices["date"] = pd.to_datetime(prices["date"])
    regimes["date"] = pd.to_datetime(regimes["date"])

    wide = utils.returns_wide(prices)
    assert wide.shape == (dg.N_DAYS, len(dg.ASSETS))
    assert wide.index.is_monotonic_increasing

    parts = utils.split_by_regime(wide, utils.regime_series(regimes))
    assert set(parts) <= {"calm", "turbulent"}
    assert sum(len(p) for p in parts.values()) == len(wide)


def test_historical_var_and_expected_shortfall_known_values():
    """VaR and ES on a simple series with a hand-computed answer."""
    returns = np.arange(-100, 0, dtype=float)  # -100, ..., -1
    # 5% quantile sits at position 4.95 -> -95.05
    assert utils.historical_var(returns, 0.95) == pytest.approx(95.05)
    # Values at or below the cutoff are -100..-96, whose mean is -98
    assert utils.expected_shortfall(returns, 0.95) == pytest.approx(98.0)
    assert utils.expected_shortfall(returns, 0.95) >= utils.historical_var(returns, 0.95)
    with pytest.raises(ValueError):
        utils.historical_var(returns, 1.5)


def test_annualized_vol_scaling():
    """Annualized volatility equals daily std times sqrt(365)."""
    rng = np.random.default_rng(0)
    series = pd.Series(rng.normal(0, 0.02, 500))
    assert utils.annualized_vol(series) == pytest.approx(
        series.std() * np.sqrt(365)
    )
    rolling = utils.rolling_annualized_vol(series, 30)
    assert rolling.iloc[:29].isna().all()
    assert rolling.iloc[29:].notna().all()


def test_correlation_helpers():
    """Mean pairwise correlation, its rolling version and the implied value."""
    corr = np.array([[1.0, 0.5, 0.1],
                     [0.5, 1.0, 0.3],
                     [0.1, 0.3, 1.0]])
    assert utils.mean_pairwise_corr(corr) == pytest.approx((0.5 + 0.1 + 0.3) / 3)

    rng = np.random.default_rng(1)
    data = pd.DataFrame(rng.normal(size=(50, 3)), columns=list("abc"))
    rolling = utils.rolling_mean_pairwise_corr(data, 20)
    assert rolling.iloc[:19].isna().all()
    assert rolling.iloc[19:].between(-1, 1).all()

    # Equal loadings of 0.5 -> correlation 0.25 for every pair
    assert utils.expected_mean_corr([0.5, 0.5, 0.5]) == pytest.approx(0.25)
    # Boost is capped: 0.9 + 0.1 -> 0.95, so correlation is 0.95 * 0.95
    assert utils.expected_mean_corr([0.9, 0.9], boost=0.1, max_loading=0.95) == (
        pytest.approx(0.9025)
    )


def test_expected_annual_vol_scales_settings_by_regime():
    """Expected volatility is the base annual volatility times the regime multiplier."""
    calm = utils.expected_annual_vol(dg.ASSETS, dg.VOL_MULTIPLIER[0])
    turbulent = utils.expected_annual_vol(dg.ASSETS, dg.VOL_MULTIPLIER[1])
    base = dg.ASSETS.set_index("asset")["annual_volatility"]

    assert list(calm.index) == list(dg.ASSETS["asset"])
    pd.testing.assert_series_equal(calm, base * dg.VOL_MULTIPLIER[0], check_names=False)
    # Turbulent / calm ratio equals the ratio of the two multipliers
    np.testing.assert_allclose(turbulent / calm, dg.VOL_MULTIPLIER[1] / dg.VOL_MULTIPLIER[0])


def test_to_jsonable_makes_output_json_serialisable():
    """NaN becomes None and numpy scalars become plain Python before json.dumps."""
    obj = {
        "vol": np.float64(0.5),
        "count": np.int64(3),
        "missing": np.nan,
        "nested": [np.float32(1.5), {"flag": "ok"}],
    }
    result = utils.to_jsonable(obj)
    assert result == {"vol": 0.5, "count": 3, "missing": None,
                      "nested": [1.5, {"flag": "ok"}]}
    assert isinstance(result["count"], int)
    assert json.loads(json.dumps(result))["missing"] is None
