"""Reusable helpers for the pharmacy sales analysis (SYNTHETIC data).

Contains feature preparation (Fourier terms, exogenous matrix, temperature
climatology), a time-based split, a seasonal naive baseline, forecast metrics
and a seasonal-strength helper. Kept free of plotting and file I/O so it can
be unit-tested.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

SEASON_WEEKS = 52  # weekly data, one yearly cycle


# ---------------------------------------------------------------------------
# Splitting and baseline
# ---------------------------------------------------------------------------
def time_split(df: pd.DataFrame, test_weeks: int, date_col: str = "week_start"):
    """Split a table into train / test by time (the last `test_weeks` dates are the test)."""
    dates = np.sort(df[date_col].unique())
    if test_weeks <= 0 or test_weeks >= len(dates):
        raise ValueError("test_weeks must be between 1 and the number of dates - 1")
    cutoff = dates[-test_weeks]
    return df[df[date_col] < cutoff].copy(), df[df[date_col] >= cutoff].copy()


def seasonal_naive_forecast(train_values, horizon: int, season: int = SEASON_WEEKS) -> np.ndarray:
    """Repeat the last `season` observed values to cover `horizon` steps."""
    train_values = np.asarray(train_values, dtype=float)
    if len(train_values) < season:
        raise ValueError("Training series is shorter than one season")
    last_cycle = train_values[-season:]
    reps = int(np.ceil(horizon / season))
    return np.tile(last_cycle, reps)[:horizon]


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------
def fourier_terms(dates: pd.DatetimeIndex, k: int = 3) -> pd.DataFrame:
    """Sin/cos pairs for the first `k` yearly harmonics, based on day of year."""
    doy = np.asarray(dates.dayofyear, dtype=float)
    cols = {}
    for h in range(1, k + 1):
        angle = 2 * np.pi * h * doy / 365.25
        cols[f"sin{h}"] = np.sin(angle)
        cols[f"cos{h}"] = np.cos(angle)
    return pd.DataFrame(cols, index=dates)


def temperature_climatology(train: pd.DataFrame, target_dates: pd.DatetimeIndex) -> np.ndarray:
    """Mean temperature per ISO week of year from the training data.

    Used as a realistic stand-in for future temperature, which is not known in
    advance. Week 53 is merged into week 52.
    """
    wk = train["week_start"].dt.isocalendar().week.astype(int).clip(upper=52)
    clim = train.groupby(wk.to_numpy())["avg_temp_c"].mean()
    target_wk = np.minimum(np.asarray(target_dates.isocalendar().week, dtype=int), 52)
    return clim.reindex(target_wk).to_numpy()


def build_exog(dates: pd.DatetimeIndex, promo, temp, origin: pd.Timestamp, k: int = 3) -> pd.DataFrame:
    """Exogenous matrix: promo flag, temperature, linear time (years) and Fourier terms."""
    exog = fourier_terms(dates, k)
    exog.insert(0, "t_years", (dates - origin).days.to_numpy() / 365.25)
    exog.insert(0, "temp_c", np.asarray(temp, dtype=float))
    exog.insert(0, "promo", np.asarray(promo, dtype=float))
    return exog


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2)))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error in percent (zero actuals are skipped)."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def forecast_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE and MAPE (%) in one dict."""
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "mape": mape(y_true, y_pred)}


def interval_coverage(y_true, lower, upper) -> float:
    """Share of actual values inside [lower, upper]."""
    y_true = np.asarray(y_true, float)
    return float(np.mean((y_true >= np.asarray(lower, float)) & (y_true <= np.asarray(upper, float))))


# ---------------------------------------------------------------------------
# Seasonality
# ---------------------------------------------------------------------------
def seasonal_profile(series: pd.Series, period: int = SEASON_WEEKS) -> dict:
    """Additive decomposition of log(series); returns strength, amplitude and peak month.

    - strength: max(0, 1 - Var(resid) / Var(seasonal + resid)); 0 = none, 1 = fully seasonal
    - peak_to_trough: ratio of highest to lowest seasonal factor (exp of log difference)
    - peak_month / trough_month: calendar month with the highest / lowest mean seasonal component
    """
    log_y = np.log(series.astype(float))
    dec = seasonal_decompose(log_y, model="additive", period=period, extrapolate_trend="freq")
    seasonal, resid = dec.seasonal, dec.resid
    strength = max(0.0, 1.0 - float(np.var(resid) / np.var(seasonal + resid)))
    by_month = seasonal.groupby(seasonal.index.month).mean()
    return {
        "strength": strength,
        "peak_to_trough": float(np.exp(seasonal.max() - seasonal.min())),
        "peak_month": int(by_month.idxmax()),
        "trough_month": int(by_month.idxmin()),
    }
