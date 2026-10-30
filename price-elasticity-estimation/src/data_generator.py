"""Synthetic data generator for the price elasticity project.

Creates two tables and saves them to <project root>/data/:
    products.csv      - product master data (60 products)
    weekly_sales.csv  - weekly price and sales per product (60 x 52 = 3120 rows)

Demand is generated from a constant-elasticity (log-log) model:

    units ~ Poisson( base_units * (price / base_price) ** elasticity
                     * (competitor_price / base_price) ** CROSS_ELASTICITY
                     * seasonality * holiday_lift * promo_lift )

The category elasticities below are the "ground truth" used for generation
only; they are not stored in the CSV files, so the model has to recover them.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

SEED = 42
N_PRODUCTS_PER_CATEGORY = 12
N_WEEKS = 52
START_DATE = "2024-01-01"  # a Monday

# True (generation) price elasticity per category.
CATEGORY_ELASTICITY = {
    "Beverages": -2.1,
    "Snacks": -1.6,
    "Personal Care": -1.2,
    "Household": -0.9,
    "Dairy": -0.7,
}

# Typical regular price level (median) per category.
CATEGORY_BASE_PRICE = {
    "Beverages": 2.5,
    "Snacks": 3.2,
    "Personal Care": 6.5,
    "Household": 7.5,
    "Dairy": 3.8,
}

BRAND_TIERS = ["Budget", "Mainstream", "Premium"]
BRAND_TIER_PROBS = [0.3, 0.5, 0.2]
BRAND_TIER_PRICE_MULT = {"Budget": 0.8, "Mainstream": 1.0, "Premium": 1.35}

CROSS_ELASTICITY = 0.3  # effect of competitor price on demand
PROMO_PROBABILITY = 0.15
PROMO_DISCOUNT_RANGE = (0.10, 0.30)
PROMO_LIFT = 1.15  # extra lift from display/advertising, beyond the price cut
HOLIDAY_WEEKS = {47, 48, 51, 52}
HOLIDAY_LIFT = 1.20


def generate_products(rng: np.random.Generator) -> pd.DataFrame:
    """Create the product master table."""
    rows = []
    product_id = 1
    for category, median_price in CATEGORY_BASE_PRICE.items():
        for _ in range(N_PRODUCTS_PER_CATEGORY):
            tier = rng.choice(BRAND_TIERS, p=BRAND_TIER_PROBS)
            price = median_price * BRAND_TIER_PRICE_MULT[tier] * rng.lognormal(0, 0.15)
            rows.append(
                {
                    "product_id": product_id,
                    "category": category,
                    "brand_tier": tier,
                    "base_price": round(float(np.clip(price, 1.5, 15.0)), 2),
                }
            )
            product_id += 1
    return pd.DataFrame(rows)


def generate_weekly_sales(products: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Create weekly price and sales data for every product."""
    weeks = pd.date_range(START_DATE, periods=N_WEEKS, freq="W-MON")
    week_of_year = np.arange(1, N_WEEKS + 1)
    seasonality = 1 + 0.15 * np.sin(2 * np.pi * week_of_year / 52)
    is_holiday = np.isin(week_of_year, list(HOLIDAY_WEEKS))

    frames = []
    for prod in products.itertuples(index=False):
        # Product-level elasticity varies slightly around the category value.
        elasticity = min(CATEGORY_ELASTICITY[prod.category] + rng.normal(0, 0.15), -0.2)
        # Product-level demand scale (typical weekly units at the base price).
        base_units = rng.lognormal(mean=5.0, sigma=0.5)

        # Regular price fluctuates a little; promotions add a discount on top.
        regular_price = prod.base_price * np.exp(rng.normal(0, 0.06, N_WEEKS))
        is_promo = rng.random(N_WEEKS) < PROMO_PROBABILITY
        discount = np.where(is_promo, rng.uniform(*PROMO_DISCOUNT_RANGE, N_WEEKS), 0.0)
        price = np.round(regular_price * (1 - discount), 2)

        # Competitor price moves independently around the product's base price.
        competitor_price = np.round(prod.base_price * np.exp(rng.normal(0, 0.08, N_WEEKS)), 2)

        expected_units = (
            base_units
            * (price / prod.base_price) ** elasticity
            * (competitor_price / prod.base_price) ** CROSS_ELASTICITY
            * seasonality
            * np.where(is_holiday, HOLIDAY_LIFT, 1.0)
            * np.where(is_promo, PROMO_LIFT, 1.0)
        )
        units = rng.poisson(expected_units)

        frames.append(
            pd.DataFrame(
                {
                    "product_id": prod.product_id,
                    "week_start": weeks,
                    "week_of_year": week_of_year,
                    "price": price,
                    "competitor_price": competitor_price,
                    "is_promo": is_promo.astype(int),
                    "is_holiday_week": is_holiday.astype(int),
                    "units_sold": units,
                    "revenue": np.round(price * units, 2),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    rng = np.random.default_rng(SEED)
    products = generate_products(rng)
    sales = generate_weekly_sales(products, rng)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    products.to_csv(DATA_DIR / "products.csv", index=False)
    sales.to_csv(DATA_DIR / "weekly_sales.csv", index=False)

    print(f"Saved {len(products)} rows to {DATA_DIR / 'products.csv'}")
    print(f"Saved {len(sales)} rows to {DATA_DIR / 'weekly_sales.csv'}")


if __name__ == "__main__":
    main()
