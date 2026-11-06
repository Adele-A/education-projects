"""Main analysis of the SYNTHETIC crypto price data.

Answers the business question: how heavy are the tails, how much does
volatility change over time, and how strongly do assets move together
(especially under stress)?

Steps:
  1. rolling volatility and volatility clustering,
  2. regime-conditional volatility versus the known simulation settings,
  3. overall, rolling and regime-conditional correlations,
  4. historical VaR / Expected Shortfall per asset and for an equal-weight basket.

Outputs: printed metrics, figures/ (2 PNG charts) and reports/metrics.json.
The data is simulated and is NOT real market data.

Run from the project root:
    python src/analysis.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import data_generator as dg
import utils

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "figures"
REPORTS_DIR = PROJECT_DIR / "reports"

VOL_WINDOW = 30    # days, rolling volatility
CORR_WINDOW = 60   # days, rolling correlation
VAR_LEVELS = (0.95, 0.99)


def load_data() -> tuple:
    """Load prices and regimes from data/ (run the generator first)."""
    paths = {n: DATA_DIR / f"{n}.csv" for n in ("daily_prices", "market_regimes")}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run 'python src/data_generator.py' first."
            )
    prices = pd.read_csv(paths["daily_prices"], parse_dates=["date"])
    regimes = pd.read_csv(paths["market_regimes"], parse_dates=["date"])
    return prices, regimes


def volatility_section(returns, by_regime) -> tuple:
    """Estimated vs expected annualized volatility per asset and regime."""
    est = pd.DataFrame({name: utils.annualized_vol(r) for name, r in by_regime.items()})
    est["all_days"] = utils.annualized_vol(returns)

    exp = pd.DataFrame(
        {
            "calm": utils.expected_annual_vol(dg.ASSETS, dg.VOL_MULTIPLIER[0]),
            "turbulent": utils.expected_annual_vol(dg.ASSETS, dg.VOL_MULTIPLIER[1]),
        }
    )
    return est, exp


def correlation_section(returns, by_regime) -> dict:
    """Mean pairwise correlation: estimated vs implied by the settings."""
    loadings = dg.ASSETS["market_loading"].to_numpy()
    result = {"all_days": {"estimated": utils.mean_pairwise_corr(returns.corr())}}
    for name, r in by_regime.items():
        boost = dg.LOADING_BOOST_TURBULENT if name == "turbulent" else 0.0
        result[name] = {
            "estimated": utils.mean_pairwise_corr(r.corr()),
            "expected_from_settings": utils.expected_mean_corr(
                loadings, boost, dg.MAX_LOADING
            ),
        }
    return result


def risk_section(returns, basket) -> pd.DataFrame:
    """Historical VaR and Expected Shortfall per asset and for the basket."""
    series = returns.copy()
    series["BASKET"] = basket
    rows = {}
    for col in series.columns:
        row = {}
        for lvl in VAR_LEVELS:
            pct = int(lvl * 100)
            row[f"var_{pct}"] = utils.historical_var(series[col], lvl)
            row[f"es_{pct}"] = utils.expected_shortfall(series[col], lvl)
        row["excess_kurtosis"] = float(stats.kurtosis(series[col]))
        rows[col] = row
    return pd.DataFrame(rows).T


def plot_vol_and_corr(basket_vol, rolling_corr, regime) -> None:
    """Chart 1: rolling basket volatility and rolling mean correlation."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    turbulent_days = regime.index[regime == "turbulent"]
    for ax in axes:
        for d in turbulent_days:
            ax.axvspan(d, d + pd.Timedelta(days=1), color="red", alpha=0.08, lw=0)

    axes[0].plot(basket_vol.index, basket_vol.values, color="navy")
    axes[0].set_ylabel("Annualized volatility")
    axes[0].set_title(f"Equal-weight basket: {VOL_WINDOW}-day rolling volatility "
                      "(red = turbulent days, synthetic data)")

    axes[1].plot(rolling_corr.index, rolling_corr.values, color="darkgreen")
    axes[1].set_ylabel("Mean pairwise correlation")
    axes[1].set_xlabel("Date")
    axes[1].set_title(f"{CORR_WINDOW}-day rolling average pairwise correlation")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rolling_vol_and_correlation.png", dpi=120)
    plt.close(fig)


def plot_vol_comparison(est, exp) -> None:
    """Chart 2: estimated vs expected annualized volatility by regime."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    x = np.arange(len(exp))
    width = 0.38
    for ax, regime in zip(axes, ["calm", "turbulent"]):
        ax.bar(x - width / 2, est.loc[exp.index, regime], width, label="Estimated")
        ax.bar(x + width / 2, exp[regime], width, label="Expected from settings")
        ax.set_xticks(x)
        ax.set_xticklabels(exp.index)
        ax.set_title(f"{regime.capitalize()} regime")
        ax.set_xlabel("Asset")
    axes[0].set_ylabel("Annualized volatility")
    axes[0].legend()
    fig.suptitle("Estimated vs simulated volatility (synthetic data)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "volatility_estimated_vs_expected.png", dpi=120)
    plt.close(fig)


def main() -> None:
    """Run the analysis, print metrics and save charts and metrics.json."""
    FIGURES_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

    prices, regimes = load_data()
    returns = utils.returns_wide(prices)
    regime = utils.regime_series(regimes)
    by_regime = utils.split_by_regime(returns, regime)
    # Equal-weight basket, daily rebalanced (mean of log returns, an approximation)
    basket = returns.mean(axis=1)

    # 1. Volatility over time and clustering
    basket_vol = utils.rolling_annualized_vol(basket, VOL_WINDOW)
    print("=== 1. Rolling volatility (basket, annualized) ===")
    print(basket_vol.describe().to_string())

    clustering = {c: utils.volatility_clustering_stats(returns[c]) for c in returns}
    clustering["BASKET"] = utils.volatility_clustering_stats(basket)
    print("\n=== 1b. Volatility clustering (lag-1 autocorr of |r|, Ljung-Box p on r^2) ===")
    print(pd.DataFrame(clustering).T.to_string())

    # 2. Regime-conditional volatility vs known settings
    est_vol, exp_vol = volatility_section(returns, by_regime)
    print("\n=== 2. Estimated annualized volatility by regime ===")
    print(est_vol.to_string())
    print("\n=== 2b. Expected annualized volatility from simulation settings ===")
    print(exp_vol.to_string())
    ratio = est_vol[["calm", "turbulent"]] / exp_vol
    print("\n=== 2c. Estimated / expected ratio ===")
    print(ratio.to_string())
    stress_ratio = est_vol["turbulent"] / est_vol["calm"]
    print("\nTurbulent / calm volatility ratio per asset (estimated):")
    print(stress_ratio.to_string())

    # 3. Correlation
    rolling_corr = utils.rolling_mean_pairwise_corr(returns, CORR_WINDOW)
    corr_summary = correlation_section(returns, by_regime)
    print("\n=== 3. Mean pairwise correlation of daily returns ===")
    for name, vals in corr_summary.items():
        print(f"{name}: {vals}")

    # 4. Risk metrics
    risk = risk_section(returns, basket)
    print("\n=== 4. Historical VaR / Expected Shortfall (daily, positive = loss) ===")
    print(risk.to_string())

    # Charts
    plot_vol_and_corr(basket_vol, rolling_corr, regime)
    plot_vol_comparison(est_vol, exp_vol)
    print(f"\nSaved 2 charts to {FIGURES_DIR}")

    # Metrics file
    metrics = {
        "data_note": "Synthetic data generated by src/data_generator.py",
        "n_days": int(len(returns)),
        "n_assets": int(returns.shape[1]),
        "windows": {"volatility_days": VOL_WINDOW, "correlation_days": CORR_WINDOW},
        "annualized_vol_estimated": est_vol.to_dict(orient="index"),
        "annualized_vol_expected": exp_vol.to_dict(orient="index"),
        "vol_ratio_estimated_over_expected": ratio.to_dict(orient="index"),
        "turbulent_over_calm_vol_ratio": stress_ratio.to_dict(),
        "mean_pairwise_correlation": corr_summary,
        "risk_metrics": risk.to_dict(orient="index"),
        "volatility_clustering": clustering,
    }
    out_path = REPORTS_DIR / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(utils.to_jsonable(metrics), f, indent=2)
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
