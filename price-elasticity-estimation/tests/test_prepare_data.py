"""Edge-case tests for prepare_model_data and extract_price_effect in src/utils.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from utils import extract_price_effect, fit_log_log_model, prepare_model_data  # noqa: E402


def small_tables(sales_product_ids):
    products = pd.DataFrame(
        {"product_id": [1], "category": ["Dairy"], "brand_tier": ["Budget"], "base_price": [2.0]}
    )
    n = len(sales_product_ids)
    sales = pd.DataFrame(
        {
            "product_id": sales_product_ids,
            "week_of_year": list(range(1, n + 1)),
            "price": [2.0, 1.8, 2.2][:n],
            "competitor_price": [2.1, 2.0, 2.0][:n],
            "units_sold": [10, 0, 14][:n],
        }
    )
    return products, sales


def test_prepare_model_data_drops_zero_units_and_computes_ratios():
    products, sales = small_tables([1, 1, 1])
    df = prepare_model_data(products, sales)
    assert len(df) == 2  # the zero-units row is removed
    assert df["log_price_ratio"].iloc[0] == pytest.approx(0.0)  # price equals base price
    assert df["log_units"].iloc[0] == pytest.approx(np.log(10))


def test_prepare_model_data_rejects_unmatched_products():
    products, sales = small_tables([1, 2, 1])
    with pytest.raises(ValueError):
        prepare_model_data(products, sales)


def test_extract_price_effect_returns_expected_keys():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(0, 0.2, n)
    df = pd.DataFrame(
        {
            "product_id": np.repeat([1, 2], n // 2),
            "log_price_ratio": x,
            "log_competitor_ratio": rng.normal(0, 0.1, n),
            "is_promo": rng.integers(0, 2, n),
            "is_holiday_week": rng.integers(0, 2, n),
            "season_sin": rng.normal(0, 1, n),
            "season_cos": rng.normal(0, 1, n),
        }
    )
    df["log_units"] = 5 - 1.5 * x + rng.normal(0, 0.05, n)
    effect = extract_price_effect(fit_log_log_model(df, controls=True))
    assert set(effect) == {"elasticity", "std_err", "ci_low", "ci_high", "p_value", "n_obs"}
    assert effect["n_obs"] == n
    assert effect["elasticity"] == pytest.approx(-1.5, abs=0.2)
