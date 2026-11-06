"""Reusable helpers for the crypto volatility analysis.

Pure functions (no file or plotting side effects) so they can be unit tested.
All inputs are expected to come from the SYNTHETIC tables created by
src/data_generator.py.
"""

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

DAYS_PER_YEAR = 365


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def returns_wide(prices: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long price table to a date x asset matrix of log returns."""
    return prices.pivot(index="date", columns="asset", values="log_return").sort_index()


def regime_series(regimes: pd.DataFrame) -> pd.Series:
    """Return the regime label as a Series indexed by date."""
    return regimes.set_index("date")["regime"].sort_index()


def split_by_regime(returns: pd.DataFrame, regime: pd.Series) -> dict:
    """Split a returns matrix into one DataFrame per regime label.

    Dates without a regime label are dropped.
    """
    labels = regime.reindex(returns.index)
    return {name: returns[labels == name] for name in sorted(labels.dropna().unique())}


# ---------------------------------------------------------------------------
# Volatility and risk metrics
# ---------------------------------------------------------------------------
def annualized_vol(returns, days_per_year: int = DAYS_PER_YEAR):
    """Annualized volatility: daily standard deviation times sqrt(days per year)."""
    return returns.std() * np.sqrt(days_per_year)


def rolling_annualized_vol(returns, window: int, days_per_year: int = DAYS_PER_YEAR):
    """Rolling annualized volatility over a window of days."""
    return returns.rolling(window).std() * np.sqrt(days_per_year)


def historical_var(returns, level: float = 0.95) -> float:
    """Historical Value-at-Risk as a positive loss (in log-return units).

    It is the empirical (1 - level) quantile of returns with the sign flipped.
    """
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    return float(-np.quantile(np.asarray(returns, dtype=float), 1 - level))


def expected_shortfall(returns, level: float = 0.95) -> float:
    """Historical Expected Shortfall: average loss on days at or beyond the VaR."""
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    arr = np.asarray(returns, dtype=float)
    cutoff = np.quantile(arr, 1 - level)
    return float(-arr[arr <= cutoff].mean())


def volatility_clustering_stats(returns: pd.Series, lags: int = 10) -> dict:
    """Simple evidence of volatility clustering for one return series.

    - lag-1 autocorrelation of absolute returns,
    - Ljung-Box p-value on squared returns (small p = clustering).
    """
    abs_ret = returns.abs()
    lb = acorr_ljungbox(returns**2, lags=[lags], return_df=True)
    return {
        "abs_return_autocorr_lag1": float(abs_ret.autocorr(lag=1)),
        "ljung_box_sq_pvalue": float(lb["lb_pvalue"].iloc[0]),
    }


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------
def mean_pairwise_corr(corr) -> float:
    """Average of the off-diagonal (upper triangle) entries of a correlation matrix."""
    mat = np.asarray(corr, dtype=float)
    iu = np.triu_indices_from(mat, k=1)
    return float(mat[iu].mean())


def rolling_mean_pairwise_corr(returns: pd.DataFrame, window: int) -> pd.Series:
    """Rolling average pairwise correlation across all assets."""
    values = returns.to_numpy()
    out = np.full(len(returns), np.nan)
    for end in range(window, len(returns) + 1):
        corr = np.corrcoef(values[end - window:end], rowvar=False)
        out[end - 1] = mean_pairwise_corr(corr)
    return pd.Series(out, index=returns.index, name="mean_pairwise_corr")


# ---------------------------------------------------------------------------
# Known simulation settings (ground truth for comparison)
# ---------------------------------------------------------------------------
def expected_annual_vol(assets_cfg: pd.DataFrame, multiplier: float) -> pd.Series:
    """Annualized volatility implied by the simulation settings for one regime."""
    return pd.Series(
        assets_cfg["annual_volatility"].to_numpy() * multiplier,
        index=assets_cfg["asset"].to_numpy(),
    )


def expected_mean_corr(loadings, boost: float = 0.0, max_loading: float = 0.95) -> float:
    """Average pairwise correlation implied by the one-factor simulation.

    With unit-variance shocks the correlation of assets i and j is b_i * b_j,
    where b is the (possibly boosted and capped) market loading.
    """
    b = np.minimum(np.asarray(loadings, dtype=float) + boost, max_loading)
    corr = np.outer(b, b)
    iu = np.triu_indices(len(b), k=1)
    return float(corr[iu].mean())


# ---------------------------------------------------------------------------
# Output helper
# ---------------------------------------------------------------------------
def to_jsonable(obj):
    """Recursively convert numpy / pandas values to plain Python for JSON."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj
