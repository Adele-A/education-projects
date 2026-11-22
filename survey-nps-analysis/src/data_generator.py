"""Generate SYNTHETIC data for the customer survey NPS analysis.

Creates two tables in the data/ folder of the project:
  - customers.csv        (4000 rows)
  - survey_responses.csv (3000 rows)

All values are random and produced with numpy (seed 42). They do not
describe real customers.

Run from the project root:  python src/data_generator.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

SEED = 42
N_CUSTOMERS = 4000
N_RESPONSES = 3000
REFERENCE_DATE = pd.Timestamp("2025-12-31")  # tenure is measured up to this date
SURVEY_START = pd.Timestamp("2025-01-01")
COMMENT_MISSING_SHARE = 0.30  # share of responses without a comment

SEGMENTS = ["Consumer", "Small Business", "Enterprise"]
SEGMENT_PROBS = [0.60, 0.30, 0.10]
PLANS = ["Basic", "Standard", "Premium"]
REGIONS = ["North", "South", "East", "West"]
CHANNELS = ["Online", "Retail Store", "Referral", "Partner"]

# Effects on the latent satisfaction level (higher = more satisfied)
SEGMENT_EFFECT = {"Consumer": 0.0, "Small Business": -0.4, "Enterprise": 0.5}
PLAN_EFFECT = {"Basic": -0.6, "Standard": 0.0, "Premium": 0.7}
REGION_EFFECT = {"North": 0.2, "South": -0.2, "East": 0.0, "West": 0.1}

# Comment building blocks, grouped by sentiment
POSITIVE_PHRASES = [
    "great customer support",
    "easy to use",
    "good value for money",
    "fast delivery",
    "reliable service",
    "friendly staff",
    "the app works well",
]
NEGATIVE_PHRASES = [
    "slow support response",
    "too expensive",
    "frequent outages",
    "billing errors",
    "confusing app",
    "long wait times",
    "poor communication",
]
NEUTRAL_PHRASES = [
    "the service is okay",
    "nothing special",
    "average experience",
    "works as expected",
]
TEMPLATES = [
    "{p}",
    "Overall, {p}.",
    "I think {p}.",
    "Mainly {p}.",
]


def generate_customers(rng: np.random.Generator) -> pd.DataFrame:
    """Create the customers table."""
    segment = rng.choice(SEGMENTS, size=N_CUSTOMERS, p=SEGMENT_PROBS)

    # Enterprise customers lean towards higher plans
    plan = np.empty(N_CUSTOMERS, dtype=object)
    plan_probs = {
        "Consumer": [0.50, 0.35, 0.15],
        "Small Business": [0.25, 0.50, 0.25],
        "Enterprise": [0.05, 0.30, 0.65],
    }
    for seg, probs in plan_probs.items():
        mask = segment == seg
        plan[mask] = rng.choice(PLANS, size=mask.sum(), p=probs)

    region = rng.choice(REGIONS, size=N_CUSTOMERS, p=[0.30, 0.25, 0.25, 0.20])
    channel = rng.choice(CHANNELS, size=N_CUSTOMERS, p=[0.40, 0.25, 0.20, 0.15])

    # Right-skewed tenure in months, limited to 1..72
    tenure = np.clip(np.round(rng.gamma(shape=2.0, scale=14.0, size=N_CUSTOMERS)), 1, 72)
    tenure = tenure.astype(int)
    signup_date = REFERENCE_DATE - pd.to_timedelta(tenure * 30, unit="D")

    return pd.DataFrame(
        {
            "customer_id": np.arange(1, N_CUSTOMERS + 1),
            "segment": segment,
            "plan": plan,
            "region": region,
            "acquisition_channel": channel,
            "signup_date": signup_date.strftime("%Y-%m-%d"),
            "tenure_months": tenure,
        }
    )


def nps_category(score: int) -> str:
    """Standard NPS buckets: 0-6 detractor, 7-8 passive, 9-10 promoter."""
    if score >= 9:
        return "Promoter"
    if score >= 7:
        return "Passive"
    return "Detractor"


def make_comment(category: str, rng: np.random.Generator) -> str:
    """Build a short free-text comment consistent with the NPS category."""
    if category == "Promoter":
        n = int(rng.integers(1, 3))
        phrases = list(rng.choice(POSITIVE_PHRASES, size=n, replace=False))
    elif category == "Detractor":
        n = int(rng.integers(1, 3))
        phrases = list(rng.choice(NEGATIVE_PHRASES, size=n, replace=False))
    else:
        # Passives: a neutral remark, sometimes with one positive or negative point
        phrases = [str(rng.choice(NEUTRAL_PHRASES))]
        extra = rng.random()
        if extra < 0.35:
            phrases.append(str(rng.choice(POSITIVE_PHRASES)))
        elif extra < 0.60:
            phrases.append(str(rng.choice(NEGATIVE_PHRASES)))

    text = " and ".join(phrases)
    template = TEMPLATES[int(rng.integers(0, len(TEMPLATES)))]
    comment = template.format(p=text)
    return comment[0].upper() + comment[1:]


def generate_responses(customers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Create the survey responses table for a random subset of customers."""
    sampled = customers.sample(n=N_RESPONSES, random_state=SEED).reset_index(drop=True)

    # Latent satisfaction -> 0-10 score
    latent = (
        7.2
        + sampled["segment"].map(SEGMENT_EFFECT)
        + sampled["plan"].map(PLAN_EFFECT)
        + sampled["region"].map(REGION_EFFECT)
        + 0.01 * np.minimum(sampled["tenure_months"], 48)
        + rng.normal(0, 2.0, size=N_RESPONSES)
    )
    score = np.clip(np.round(latent), 0, 10).astype(int)
    category = [nps_category(s) for s in score]

    comments = [make_comment(c, rng) for c in category]
    missing = rng.random(N_RESPONSES) < COMMENT_MISSING_SHARE
    comments = [None if m else c for c, m in zip(comments, missing)]

    survey_date = SURVEY_START + pd.to_timedelta(
        rng.integers(0, 365, size=N_RESPONSES), unit="D"
    )

    responses = pd.DataFrame(
        {
            "response_id": np.arange(1, N_RESPONSES + 1),
            "customer_id": sampled["customer_id"].to_numpy(),
            "survey_date": survey_date.strftime("%Y-%m-%d"),
            "nps_score": score,
            "nps_category": category,
            "comment": comments,
        }
    )
    return responses.sort_values("survey_date").reset_index(drop=True).assign(
        response_id=lambda d: np.arange(1, len(d) + 1)
    )


def main() -> None:
    rng = np.random.default_rng(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    customers = generate_customers(rng)
    responses = generate_responses(customers, rng)

    customers.to_csv(DATA_DIR / "customers.csv", index=False)
    responses.to_csv(DATA_DIR / "survey_responses.csv", index=False)

    print(f"Saved customers.csv: {customers.shape[0]} rows, {customers.shape[1]} columns")
    print(f"Saved survey_responses.csv: {responses.shape[0]} rows, {responses.shape[1]} columns")
    print(f"Output folder: {DATA_DIR}")


if __name__ == "__main__":
    main()
