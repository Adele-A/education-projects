"""Reusable helpers for the price elasticity analysis.

Contains feature preparation, model fitting and metric functions so they can
be unit-tested separately from the reporting script (src/analysis.py).
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Column holding the price effect; its coefficient is the price elasticity
# because both units and price are on the log scale.
PRICE_TERM = "log_price_ratio"

# Control variables added in the "controlled" specification.
CONTROL_TERMS = [
    "log_competitor_ratio",
    "is_promo",
    "is_holiday_week",
    "season_sin",
    "season_cos",
]


def prepare_model_data(products: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    """Join sales with products and build the regression features.

    Rows with zero units (log undefined) or non-positive prices are removed.
    Prices are expressed relative to each product's regular base price, so the
    coefficient on the log price ratio is the elasticity.
    """
    df = sales.merge(products, on="product_id", how="left", validate="many_to_one")
    if df["category"].isna().any():
        raise ValueError("Some sales rows have no matching product.")

    valid = (df["units_sold"] > 0) & (df["price"] > 0) & (df["competitor_price"] > 0)
    df = df.loc[valid].copy()

    df["log_units"] = np.log(df["units_sold"])
    df["log_price_ratio"] = np.log(df["price"] / df["base_price"])
    df["log_competitor_ratio"] = np.log(df["competitor_price"] / df["base_price"])

    # Smooth yearly seasonality as a pair of Fourier terms.
    angle = 2 * np.pi * df["week_of_year"] / 52
    df["season_sin"] = np.sin(angle)
    df["season_cos"] = np.cos(angle)
    return df.reset_index(drop=True)


def build_formula(controls: bool = True) -> str:
    """Return the statsmodels formula (product fixed effects always included)."""
    terms = [PRICE_TERM] + (CONTROL_TERMS if controls else []) + ["C(product_id)"]
    return "log_units ~ " + " + ".join(terms)


def fit_log_log_model(df: pd.DataFrame, controls: bool = True):
    """Fit the log-log OLS model with heteroskedasticity-robust (HC1) errors."""
    return smf.ols(build_formula(controls), data=df).fit(cov_type="HC1")


def extract_price_effect(result, term: str = PRICE_TERM) -> dict:
    """Pull the elasticity estimate, standard error and 95% CI from a fit."""
    ci = result.conf_int().loc[term]
    return {
        "elasticity": float(result.params[term]),
        "std_err": float(result.bse[term]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "p_value": float(result.pvalues[term]),
        "n_obs": int(result.nobs),
    }


def regression_metrics(y_true, y_pred) -> dict:
    """RMSE, MAE and R^2 for predictions on the log-units scale."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mae": float(np.mean(np.abs(resid))),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def classify_elasticity(elasticity: float) -> str:
    """Label demand as elastic (|e| > 1) or inelastic (|e| <= 1)."""
    return "elastic" if abs(elasticity) > 1 else "inelastic"


def implied_revenue_change_pct(elasticity: float, price_change_pct: float) -> float:
    """Revenue change (in %) implied by a constant-elasticity demand curve.

    units scale by (1 + dp) ** e, revenue by (1 + dp) ** (1 + e).
    """
    factor = 1 + price_change_pct / 100
    return float((factor ** (1 + elasticity) - 1) * 100)
