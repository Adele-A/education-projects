"""Tests for the modelling helpers in src/utils.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import data_generator as dg  # noqa: E402
from utils import (  # noqa: E402
    build_formula,
    classify_elasticity,
    extract_price_effect,
    fit_log_log_model,
    implied_revenue_change_pct,
    prepare_model_data,
    regression_metrics,
)


@pytest.fixture(scope="module")
def model_df():
    rng = np.random.default_rng(dg.SEED)
    products = dg.generate_products(rng)
    sales = dg.generate_weekly_sales(products, rng)
    return prepare_model_data(products, sales)


def test_prepare_drops_zero_units_and_adds_features(model_df):
    assert (model_df["units_sold"] > 0).all()
    assert model_df["log_units"].notna().all()
    for col in ["log_price_ratio", "log_competitor_ratio", "season_sin", "season_cos"]:
        assert col in model_df.columns


def test_formula_controls_toggle():
    assert "is_promo" in build_formula(controls=True)
    assert "is_promo" not in build_formula(controls=False)
    assert "C(product_id)" in build_formula(controls=False)


def test_pooled_elasticity_is_negative(model_df):
    effect = extract_price_effect(fit_log_log_model(model_df, controls=True))
    assert effect["elasticity"] < 0
    assert effect["ci_low"] < effect["elasticity"] < effect["ci_high"]


def test_regression_metrics_perfect_and_imperfect():
    y = np.array([1.0, 2.0, 3.0])
    perfect = regression_metrics(y, y)
    assert perfect["rmse"] == 0 and perfect["r2"] == pytest.approx(1.0)
    off = regression_metrics(y, y + 1)
    assert off["rmse"] == pytest.approx(1.0)
    assert off["mae"] == pytest.approx(1.0)


def test_classify_elasticity():
    assert classify_elasticity(-1.6) == "elastic"
    assert classify_elasticity(-0.7) == "inelastic"


def test_implied_revenue_change():
    # Unit elasticity: revenue does not change.
    assert implied_revenue_change_pct(-1.0, -10) == pytest.approx(0.0)
    # Elastic demand: a price cut raises revenue.
    assert implied_revenue_change_pct(-2.0, -10) > 0
    # Inelastic demand: a price cut lowers revenue.
    assert implied_revenue_change_pct(-0.5, -10) < 0
