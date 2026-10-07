"""Reusable helpers for the forecasting analysis.

Contains data preparation (weekly series per category), a time-based
train/test split, forecast functions, error metrics and decomposition
strength measures. Kept free of file I/O so it is easy to test.

All data in this project is synthetic.
"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing

SEASON_LENGTH = 52  # weeks per year


def weekly_category_series(sales: pd.DataFrame) -> pd.DataFrame:
    """Total weekly units per category (all stores) as a wide table.

    Index: week_start (weekly frequency, Mondays). Columns: categories.
    """
    wide = (sales.groupby(["week_start", "category"])["units_sold"]
            .sum().unstack("category").sort_index())
    return wide.asfreq("W-MON")


def time_split(series: pd.DataFrame | pd.Series, horizon: int):
    """Split by time: the last `horizon` weeks form the test set."""
    if horizon <= 0 or horizon >= len(series):
        raise ValueError("horizon must be between 1 and len(series) - 1")
    return series.iloc[:-horizon], series.iloc[-horizon:]


def naive_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    """Repeat the last observed value for every forecast week."""
    return np.repeat(float(train.iloc[-1]), horizon)


def seasonal_naive_forecast(train: pd.Series, horizon: int,
                            season: int = SEASON_LENGTH) -> np.ndarray:
    """Forecast each week with the value from the same week one season ago."""
    if len(train) < season:
        raise ValueError("train must contain at least one full season")
    values = train.to_numpy(dtype=float)
    last_season = values[-season:]
    return np.array([last_season[h % season] for h in range(horizon)])


def holt_winters_forecast(train: pd.Series, horizon: int,
                          season: int = SEASON_LENGTH) -> np.ndarray:
    """Holt-Winters forecast: damped additive trend, multiplicative seasonality."""
    if len(train) < 2 * season:
        raise ValueError("Holt-Winters needs at least two full seasons of data")
    model = ExponentialSmoothing(
        train.astype(float),
        trend="add",
        damped_trend=True,
        seasonal="mul",
        seasonal_periods=season,
        initialization_method="estimated",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        fit = model.fit(optimized=True)
    return np.asarray(fit.forecast(horizon), dtype=float)


def mae(actual, forecast) -> float:
    """Mean absolute error."""
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    return float(np.mean(np.abs(a - f)))


def rmse(actual, forecast) -> float:
    """Root mean squared error."""
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    return float(np.sqrt(np.mean((a - f) ** 2)))


def mape(actual, forecast) -> float:
    """Mean absolute percentage error in percent (zero actuals are skipped)."""
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    mask = a != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((a[mask] - f[mask]) / a[mask])) * 100)


def evaluate_forecast(actual, forecast) -> dict:
    """Return MAE, RMSE and MAPE (percent) as a dictionary."""
    return {
        "mae": mae(actual, forecast),
        "rmse": rmse(actual, forecast),
        "mape_pct": mape(actual, forecast),
    }


def component_strength(component: pd.Series, residual: pd.Series) -> float:
    """Strength of a decomposition component between 0 and 1.

    Defined as max(0, 1 - Var(residual) / Var(component + residual)),
    computed on log values for a multiplicative decomposition.
    """
    both = pd.concat([component, residual], axis=1).dropna()
    comp = np.log(both.iloc[:, 0])
    res = np.log(both.iloc[:, 1])
    total_var = np.var(comp + res)
    if total_var == 0:
        return 0.0
    return float(max(0.0, 1 - np.var(res) / total_var))
