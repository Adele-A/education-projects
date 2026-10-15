"""Generate SYNTHETIC data for the Website Funnel Analysis project.

Creates two CSV tables in the data/ folder:
  - sessions.csv         : one row per session with funnel stage flags
  - traffic_sources.csv  : channel type and cost per session for each source

All data is artificial. The random seed is fixed so results are reproducible.
Run from the project root:  python src/data_generator.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

SEED = 42
N_SESSIONS = 5000
START_DATE = "2025-01-01"
N_DAYS = 90

# --- Traffic source configuration -------------------------------------------
# share: share of all sessions; mult: multiplier applied to funnel step
# probabilities (organic/email visitors tend to be more engaged than social).
SOURCES = {
    #                share  view  cart  checkout  purchase  channel_type  cost/session
    "organic_search": (0.30, 1.00, 1.05, 1.00, 1.00, "unpaid", 0.00),
    "paid_search":    (0.22, 1.05, 1.00, 1.00, 1.00, "paid",   1.20),
    "direct":         (0.18, 1.05, 1.10, 1.05, 1.10, "unpaid", 0.00),
    "social":         (0.15, 0.80, 0.75, 0.90, 0.90, "paid",   0.65),
    "email":          (0.10, 1.15, 1.20, 1.10, 1.15, "owned",  0.10),
    "referral":       (0.05, 0.95, 0.95, 1.00, 1.00, "unpaid", 0.30),
}

DEVICES = {
    # share, multiplier on checkout and purchase steps (mobile checkout is harder)
    "desktop": (0.45, 1.00),
    "mobile":  (0.45, 0.85),
    "tablet":  (0.10, 0.95),
}

COUNTRIES = {"US": 0.40, "UK": 0.15, "DE": 0.15, "FR": 0.10, "CA": 0.10, "AU": 0.10}

# Base conditional probabilities of moving to the next funnel step
BASE_VIEW = 0.62      # session -> product view
BASE_CART = 0.35      # product view -> add to cart
BASE_CHECKOUT = 0.60  # add to cart -> checkout
BASE_PURCHASE = 0.65  # checkout -> purchase

RETURNING_SHARE = 0.35
RETURNING_MULT = 1.20  # returning users convert better at every step


def _clip(p: np.ndarray) -> np.ndarray:
    """Keep probabilities in a sensible range."""
    return np.clip(p, 0.02, 0.95)


def generate_sessions(rng: np.random.Generator) -> pd.DataFrame:
    """Build the sessions table."""
    n = N_SESSIONS

    # Dates and categorical attributes
    day_offset = np.sort(rng.integers(0, N_DAYS, size=n))
    dates = pd.to_datetime(START_DATE) + pd.to_timedelta(day_offset, unit="D")

    src_names = list(SOURCES)
    source = rng.choice(src_names, size=n, p=[SOURCES[s][0] for s in src_names])
    dev_names = list(DEVICES)
    device = rng.choice(dev_names, size=n, p=[DEVICES[d][0] for d in dev_names])
    country = rng.choice(list(COUNTRIES), size=n, p=list(COUNTRIES.values()))
    user_type = np.where(rng.random(n) < RETURNING_SHARE, "returning", "new")

    # Per-session multipliers
    src_series = pd.Series(source)
    m_view = src_series.map({s: v[1] for s, v in SOURCES.items()}).to_numpy()
    m_cart = src_series.map({s: v[2] for s, v in SOURCES.items()}).to_numpy()
    m_check = src_series.map({s: v[3] for s, v in SOURCES.items()}).to_numpy()
    m_purch = src_series.map({s: v[4] for s, v in SOURCES.items()}).to_numpy()
    m_dev = pd.Series(device).map({d: v[1] for d, v in DEVICES.items()}).to_numpy()
    m_user = np.where(user_type == "returning", RETURNING_MULT, 1.0)

    # Nested funnel: a later step can only happen if the earlier one happened
    p_view = _clip(BASE_VIEW * m_view * m_user)
    p_cart = _clip(BASE_CART * m_cart * m_user)
    p_check = _clip(BASE_CHECKOUT * m_check * m_dev)
    p_purch = _clip(BASE_PURCHASE * m_purch * m_dev * m_user)

    viewed = rng.random(n) < p_view
    carted = viewed & (rng.random(n) < p_cart)
    checkout = carted & (rng.random(n) < p_check)
    purchased = checkout & (rng.random(n) < p_purch)

    # Engagement metrics consistent with how deep the session went
    pages = (
        1
        + viewed * (1 + rng.poisson(1.5, n))
        + carted * rng.poisson(1.0, n)
        + checkout * rng.poisson(0.5, n)
    )
    duration = np.round(
        rng.lognormal(mean=3.2, sigma=0.5, size=n) * pages
    ).astype(int)
    duration = np.clip(duration, 3, 3600)

    # Order value only for purchasing sessions (mobile orders slightly smaller)
    value = rng.lognormal(mean=4.2, sigma=0.5, size=n)
    value = value * np.where(device == "mobile", 0.9, 1.0)
    order_value = np.where(purchased, np.round(value, 2), 0.0)

    df = pd.DataFrame(
        {
            "session_id": np.arange(1, n + 1),
            "session_date": dates.strftime("%Y-%m-%d"),
            "traffic_source": source,
            "device": device,
            "country": country,
            "user_type": user_type,
            "pages_viewed": pages.astype(int),
            "session_duration_sec": duration,
            "viewed_product": viewed.astype(int),
            "added_to_cart": carted.astype(int),
            "reached_checkout": checkout.astype(int),
            "purchased": purchased.astype(int),
            "order_value": order_value,
        }
    )
    return df


def generate_traffic_sources() -> pd.DataFrame:
    """Build the small traffic source dimension table."""
    return pd.DataFrame(
        {
            "traffic_source": list(SOURCES),
            "channel_type": [v[5] for v in SOURCES.values()],
            "cost_per_session": [v[6] for v in SOURCES.values()],
        }
    )


def main() -> None:
    rng = np.random.default_rng(SEED)
    os.makedirs(DATA_DIR, exist_ok=True)

    sessions = generate_sessions(rng)
    sources = generate_traffic_sources()

    # Sanity check: the funnel must be nested
    assert (sessions["added_to_cart"] <= sessions["viewed_product"]).all()
    assert (sessions["reached_checkout"] <= sessions["added_to_cart"]).all()
    assert (sessions["purchased"] <= sessions["reached_checkout"]).all()

    sessions.to_csv(DATA_DIR / "sessions.csv", index=False)
    sources.to_csv(DATA_DIR / "traffic_sources.csv", index=False)

    print(f"Saved {len(sessions)} rows to {DATA_DIR / 'sessions.csv'}")
    print(f"Saved {len(sources)} rows to {DATA_DIR / 'traffic_sources.csv'}")


if __name__ == "__main__":
    main()
