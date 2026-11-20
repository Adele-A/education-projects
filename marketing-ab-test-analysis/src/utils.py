"""Reusable helpers for the SYNTHETIC landing page A/B test analysis.

Contains data loading, feature preparation and the statistical functions
(two-proportion z-test, confidence intervals, effect size, power and sample
size, segment breakdowns, logistic regression) used by src/analysis.py and
covered by tests/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_confint, proportion_effectsize

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"


# --------------------------------------------------------------------------
# Data loading and preparation
# --------------------------------------------------------------------------
def load_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read users.csv and orders.csv."""
    users = pd.read_csv(data_dir / "users.csv", parse_dates=["visit_date"])
    orders = pd.read_csv(data_dir / "orders.csv", parse_dates=["order_date"])
    return users, orders


def prepare_features(users: pd.DataFrame) -> pd.DataFrame:
    """Add a 0/1 `is_variant` column (1 = variant, 0 = control)."""
    df = users.copy()
    df["is_variant"] = (df["group"] == "variant").astype(int)
    return df


def add_revenue(users: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Attach `revenue` per visitor; visitors without an order get 0."""
    order_value = orders.groupby("user_id")["order_value"].sum().rename("revenue")
    merged = users.merge(order_value, on="user_id", how="left")
    merged["revenue"] = merged["revenue"].fillna(0.0)
    return merged


# --------------------------------------------------------------------------
# Conversion tests
# --------------------------------------------------------------------------
def group_counts(users: pd.DataFrame) -> dict:
    """Visitors and buyers for control and variant."""
    out = {}
    for g in ["control", "variant"]:
        sub = users.loc[users["group"] == g, "converted"]
        out[g] = {"n": int(len(sub)), "x": int(sub.sum())}
    return out


def two_proportion_ztest(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    """Two-sided pooled z-test for H0: p1 == p2. Returns (z, p_value).

    Group 1 is control, group 2 is variant, so z > 0 means variant is higher.
    """
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan"), float("nan")
    z = (p2 - p1) / se
    return float(z), float(2 * stats.norm.sf(abs(z)))


def diff_confidence_interval(
    x1: int, n1: int, x2: int, n2: int, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Wald CI for p2 - p1. Returns (difference, lower, upper)."""
    p1, p2 = x1 / n1, x2 / n2
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    zc = stats.norm.ppf(1 - alpha / 2)
    diff = p2 - p1
    return float(diff), float(diff - zc * se), float(diff + zc * se)


def relative_uplift_ci(
    x1: int, n1: int, x2: int, n2: int, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Relative uplift p2/p1 - 1 with a log risk-ratio CI.

    Returns (uplift, lower, upper); NaNs if either group has no conversions.
    """
    if x1 == 0 or x2 == 0:
        return float("nan"), float("nan"), float("nan")
    rr = (x2 / n2) / (x1 / n1)
    se = np.sqrt(1 / x1 - 1 / n1 + 1 / x2 - 1 / n2)
    zc = stats.norm.ppf(1 - alpha / 2)
    lo, hi = np.exp(np.log(rr) + np.array([-zc, zc]) * se)
    return float(rr - 1), float(lo - 1), float(hi - 1)


def wilson_ci(x: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson confidence interval for a single proportion."""
    lo, hi = proportion_confint(x, n, alpha=alpha, method="wilson")
    return float(lo), float(hi)


def chi_square_test(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    """Chi-square test of independence on the 2x2 group-by-conversion table.

    Returns (statistic, p_value). No continuity correction, so it matches
    the pooled z-test (chi2 = z^2).
    """
    table = np.array([[x1, n1 - x1], [x2, n2 - x2]])
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    return float(chi2), float(p)


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions (p2 vs p1)."""
    return float(proportion_effectsize(p2, p1))


def sample_ratio_test(n_control: int, n_variant: int) -> float:
    """Chi-square goodness-of-fit p-value against an intended 50/50 split."""
    total = n_control + n_variant
    return float(stats.chisquare([n_control, n_variant], [total / 2, total / 2]).pvalue)


# --------------------------------------------------------------------------
# Power and sample size
# --------------------------------------------------------------------------
def achieved_power(
    p1: float, p2: float, n1: int, n2: int, alpha: float = 0.05
) -> float:
    """Power of a two-sided test to detect p1 vs p2 with the given group sizes.

    When p2 is the *observed* rate this is post-hoc power, which only
    restates the p-value; prefer `minimum_detectable_rate` for planning.
    """
    h = abs(cohens_h(p1, p2))
    return float(
        NormalIndPower().solve_power(
            effect_size=h, nobs1=n1, alpha=alpha, ratio=n2 / n1,
            alternative="two-sided",
        )
    )


def required_sample_size(
    p1: float, p2: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Visitors needed PER GROUP to detect p1 vs p2 (equal group sizes)."""
    h = abs(cohens_h(p1, p2))
    n = NormalIndPower().solve_power(
        effect_size=h, power=power, alpha=alpha, ratio=1.0, alternative="two-sided"
    )
    return int(np.ceil(n))


def minimum_detectable_rate(
    p1: float, n_per_group: int, alpha: float = 0.05, power: float = 0.80
) -> float:
    """Smallest variant rate (above p1) detectable with the given power."""
    h = NormalIndPower().solve_power(
        effect_size=None, nobs1=n_per_group, alpha=alpha, power=power,
        ratio=1.0, alternative="two-sided",
    )
    # Invert Cohen's h: h = 2*asin(sqrt(p2)) - 2*asin(sqrt(p1))
    p2 = np.sin((h + 2 * np.arcsin(np.sqrt(p1))) / 2) ** 2
    return float(p2)


# --------------------------------------------------------------------------
# Secondary metric: revenue per visitor
# --------------------------------------------------------------------------
def revenue_comparison(
    df: pd.DataFrame, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05
) -> dict:
    """Compare revenue per visitor (variant - control).

    Uses Welch's t-test and a percentile bootstrap CI for the difference
    in means. `df` needs `group` and `revenue` columns.
    """
    ctrl = df.loc[df["group"] == "control", "revenue"].to_numpy()
    var = df.loc[df["group"] == "variant", "revenue"].to_numpy()
    t_res = stats.ttest_ind(var, ctrl, equal_var=False)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = (
            rng.choice(var, size=len(var)).mean()
            - rng.choice(ctrl, size=len(ctrl)).mean()
        )
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "rpv_control": float(ctrl.mean()),
        "rpv_variant": float(var.mean()),
        "rpv_difference": float(var.mean() - ctrl.mean()),
        "rpv_diff_ci_low": float(lo),
        "rpv_diff_ci_high": float(hi),
        "welch_t": float(t_res.statistic),
        "welch_p_value": float(t_res.pvalue),
    }


# --------------------------------------------------------------------------
# Segments and regression
# --------------------------------------------------------------------------
def segment_breakdown(users: pd.DataFrame, column: str, alpha: float = 0.05) -> pd.DataFrame:
    """Variant-vs-control comparison within each level of `column`."""
    rows = []
    for level, sub in users.groupby(column):
        c = sub[sub["group"] == "control"]["converted"]
        v = sub[sub["group"] == "variant"]["converted"]
        n1, x1, n2, x2 = len(c), int(c.sum()), len(v), int(v.sum())
        if n1 == 0 or n2 == 0:
            continue
        _, p = two_proportion_ztest(x1, n1, x2, n2)
        diff, lo, hi = diff_confidence_interval(x1, n1, x2, n2, alpha)
        rows.append(
            {
                "segment": f"{column}: {level}",
                "n_control": n1,
                "n_variant": n2,
                "rate_control": x1 / n1,
                "rate_variant": x2 / n2,
                "diff": diff,
                "ci_low": lo,
                "ci_high": hi,
                "p_value": p,
            }
        )
    return pd.DataFrame(rows)


def holm_adjust(p_values: pd.Series) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (NaNs are treated as 1)."""
    from statsmodels.stats.multitest import multipletests

    p = np.nan_to_num(p_values.to_numpy(dtype=float), nan=1.0)
    return multipletests(p, method="holm")[1]


def fit_logistic(df: pd.DataFrame):
    """Logistic regression of conversion on variant, adjusting for covariates."""
    formula = (
        "converted ~ is_variant + C(device) + C(traffic_source) + C(country)"
        " + is_returning + page_views + time_on_page_sec"
    )
    return smf.logit(formula, data=df).fit(disp=0)


def variant_odds_ratio(model, alpha: float = 0.05) -> dict:
    """Odds ratio, CI and p-value for the `is_variant` coefficient."""
    ci = model.conf_int(alpha=alpha).loc["is_variant"]
    return {
        "odds_ratio": float(np.exp(model.params["is_variant"])),
        "or_ci_low": float(np.exp(ci[0])),
        "or_ci_high": float(np.exp(ci[1])),
        "p_value": float(model.pvalues["is_variant"]),
    }
