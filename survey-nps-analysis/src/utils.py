"""Reusable helpers for the SYNTHETIC customer survey NPS analysis.

Contains the NPS metric, bootstrap confidence intervals, group comparison
tests and simple text keyword counting. Kept free of file I/O so every
function can be unit tested.
"""

import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

CATEGORY_ORDER = ["Detractor", "Passive", "Promoter"]

# Small stop word list for the short English comments in this dataset
STOPWORDS = {
    "a", "an", "and", "as", "at", "be", "but", "for", "i", "in", "is", "it",
    "of", "on", "or", "the", "to", "too", "overall", "think", "mainly", "are",
    "was", "with", "this", "that",
}


# ---------------------------------------------------------------------------
# NPS metric
# ---------------------------------------------------------------------------
def nps_category(score: int) -> str:
    """Standard NPS buckets: 0-6 Detractor, 7-8 Passive, 9-10 Promoter."""
    if score >= 9:
        return "Promoter"
    if score >= 7:
        return "Passive"
    return "Detractor"


def _as_scores(scores) -> np.ndarray:
    """Convert input to a float array and validate the 0-10 range."""
    arr = np.asarray(scores, dtype=float)
    if arr.size and (np.nanmin(arr) < 0 or np.nanmax(arr) > 10):
        raise ValueError("NPS scores must be between 0 and 10")
    return arr


def calculate_nps(scores) -> float:
    """NPS = % promoters (9-10) minus % detractors (0-6), range -100 to 100."""
    arr = _as_scores(scores)
    if arr.size == 0:
        return float("nan")
    return float((np.mean(arr >= 9) - np.mean(arr <= 6)) * 100)


def bootstrap_nps_ci(
    scores, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the NPS."""
    arr = _as_scores(scores)
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    samples = arr[idx]
    boot = ((samples >= 9).mean(axis=1) - (samples <= 6).mean(axis=1)) * 100
    low, high = np.percentile(boot, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return float(low), float(high)


def bootstrap_nps_difference(
    scores_a, scores_b, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float, float]:
    """NPS(a) - NPS(b) with a percentile bootstrap CI (groups resampled independently)."""
    a, b = _as_scores(scores_a), _as_scores(scores_b)
    rng = np.random.default_rng(seed)
    sa = a[rng.integers(0, a.size, size=(n_boot, a.size))]
    sb = b[rng.integers(0, b.size, size=(n_boot, b.size))]
    nps_a = ((sa >= 9).mean(axis=1) - (sa <= 6).mean(axis=1)) * 100
    nps_b = ((sb >= 9).mean(axis=1) - (sb <= 6).mean(axis=1)) * 100
    diff = nps_a - nps_b
    low, high = np.percentile(diff, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return calculate_nps(a) - calculate_nps(b), float(low), float(high)


# ---------------------------------------------------------------------------
# Segment comparison
# ---------------------------------------------------------------------------
def add_tenure_band(df: pd.DataFrame) -> pd.DataFrame:
    """Add a tenure_band column (1-12, 13-24, 25-48, 49-72 months)."""
    out = df.copy()
    out["tenure_band"] = pd.cut(
        out["tenure_months"],
        bins=[0, 12, 24, 48, 72],
        labels=["1-12", "13-24", "25-48", "49-72"],
    ).astype(str)
    return out


def nps_by_group(
    df: pd.DataFrame, group_col: str, n_boot: int = 2000, seed: int = 42
) -> pd.DataFrame:
    """NPS, category shares and bootstrap 95% CI for each group, best group first."""
    rows = []
    for group, part in df.groupby(group_col):
        scores = part["nps_score"].to_numpy()
        low, high = bootstrap_nps_ci(scores, n_boot=n_boot, seed=seed)
        rows.append(
            {
                "group": group,
                "responses": int(len(part)),
                "mean_score": float(part["nps_score"].mean()),
                "promoter_pct": float((scores >= 9).mean() * 100),
                "passive_pct": float(((scores >= 7) & (scores <= 8)).mean() * 100),
                "detractor_pct": float((scores <= 6).mean() * 100),
                "nps": calculate_nps(scores),
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows).sort_values("nps", ascending=False).reset_index(drop=True)


def compare_groups(df: pd.DataFrame, group_col: str) -> dict:
    """Test whether groups differ: chi-square on categories, Kruskal-Wallis on scores."""
    table = pd.crosstab(df[group_col], df["nps_category"])
    chi2, p_chi, dof, _ = stats.chi2_contingency(table)
    samples = [g["nps_score"].to_numpy() for _, g in df.groupby(group_col)]
    h_stat, p_kw = stats.kruskal(*samples)
    return {
        "chi2": float(chi2),
        "chi2_dof": int(dof),
        "chi2_p_value": float(p_chi),
        "kruskal_h": float(h_stat),
        "kruskal_p_value": float(p_kw),
    }


def monthly_nps(df: pd.DataFrame) -> pd.DataFrame:
    """NPS and number of responses per survey month."""
    month = df["survey_date"].dt.to_period("M").astype(str)
    rows = [
        {"month": m, "responses": int(len(g)), "nps": calculate_nps(g["nps_score"])}
        for m, g in df.groupby(month)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Text keywords
# ---------------------------------------------------------------------------
def extract_terms(text: str) -> set[str]:
    """Unique unigrams and bigrams of a comment, ignoring stop words."""
    tokens = re.findall(r"[a-z']+", str(text).lower())
    unigrams = {t for t in tokens if t not in STOPWORDS}
    bigrams = {
        f"{a} {b}"
        for a, b in zip(tokens, tokens[1:])
        if a not in STOPWORDS and b not in STOPWORDS
    }
    return unigrams | bigrams


def keyword_frequency(
    df: pd.DataFrame,
    text_col: str = "comment",
    category_col: str = "nps_category",
    top_n: int = 10,
) -> pd.DataFrame:
    """Top terms per category: number and share of that category's comments containing the term."""
    rows = []
    data = df.dropna(subset=[text_col])
    for category in CATEGORY_ORDER:
        comments = data.loc[data[category_col] == category, text_col]
        if comments.empty:
            continue
        counter = Counter()
        for text in comments:
            counter.update(extract_terms(text))
        for term, count in counter.most_common(top_n):
            rows.append(
                {
                    "category": category,
                    "term": term,
                    "comments": int(count),
                    "share_pct": float(count / len(comments) * 100),
                }
            )
    return pd.DataFrame(rows, columns=["category", "term", "comments", "share_pct"])
