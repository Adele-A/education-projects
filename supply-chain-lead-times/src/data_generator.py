"""Generate synthetic supply chain data (suppliers and purchase orders).

All data is SYNTHETIC and produced with a fixed random seed (42).
Output tables are saved to <project root>/data/.

Run from the project root:
    python src/data_generator.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

SEED = 42
N_SUPPLIERS = 60
N_ORDERS = 5000
START_DATE = pd.Timestamp("2023-01-01")
END_DATE = pd.Timestamp("2025-06-30")

# country -> (region, min contractual lead time, max contractual lead time,
#             transport modes, probabilities of each mode)
COUNTRIES = {
    "USA": ("North America", 3, 10, ["road", "rail"], [0.8, 0.2]),
    "Mexico": ("North America", 5, 12, ["road", "rail"], [0.85, 0.15]),
    "Germany": ("Europe", 7, 16, ["road", "rail", "air"], [0.6, 0.25, 0.15]),
    "Poland": ("Europe", 8, 18, ["road", "rail"], [0.8, 0.2]),
    "Turkey": ("Europe", 10, 22, ["road", "sea"], [0.6, 0.4]),
    "China": ("Asia", 28, 45, ["sea", "air"], [0.8, 0.2]),
    "Vietnam": ("Asia", 26, 42, ["sea", "air"], [0.85, 0.15]),
    "India": ("Asia", 24, 40, ["sea", "air"], [0.8, 0.2]),
}
COUNTRY_WEIGHTS = [0.12, 0.08, 0.12, 0.08, 0.07, 0.25, 0.14, 0.14]

# category -> base unit price (USD)
CATEGORIES = {
    "Electronics": 45.0,
    "Raw Materials": 8.0,
    "Packaging": 1.5,
    "Components": 22.0,
    "MRO": 12.0,
}

# reliability tier -> (share of suppliers, lead time noise sigma,
#                      mean log shift, disruption probability)
TIERS = {
    "A": (0.30, 0.08, 0.00, 0.04),
    "B": (0.45, 0.15, 0.03, 0.10),
    "C": (0.25, 0.25, 0.08, 0.22),
}


def generate_suppliers(rng: np.random.Generator) -> pd.DataFrame:
    """Create the supplier master table."""
    country_names = list(COUNTRIES)
    countries = rng.choice(country_names, size=N_SUPPLIERS, p=COUNTRY_WEIGHTS)
    tier_names = list(TIERS)
    tiers = rng.choice(tier_names, size=N_SUPPLIERS, p=[TIERS[t][0] for t in tier_names])
    categories = rng.choice(list(CATEGORIES), size=N_SUPPLIERS)

    rows = []
    for i in range(N_SUPPLIERS):
        region, lo, hi, modes, probs = COUNTRIES[countries[i]]
        rows.append(
            {
                "supplier_id": f"S{i + 1:03d}",
                "supplier_name": f"Supplier {i + 1:03d}",
                "country": countries[i],
                "region": region,
                "category": categories[i],
                "transport_mode": rng.choice(modes, p=probs),
                "reliability_tier": tiers[i],
                "contract_lead_time_days": int(rng.integers(lo, hi + 1)),
                # price level relative to the category base price
                "price_index": round(float(rng.normal(1.0, 0.12)), 3),
            }
        )
    suppliers = pd.DataFrame(rows)
    suppliers["price_index"] = suppliers["price_index"].clip(0.7, 1.4)
    return suppliers


def generate_orders(rng: np.random.Generator, suppliers: pd.DataFrame) -> pd.DataFrame:
    """Create purchase orders with promised and actual delivery dates."""
    # Some suppliers receive many more orders than others
    weights = rng.lognormal(mean=0.0, sigma=0.7, size=len(suppliers))
    weights /= weights.sum()
    idx = rng.choice(len(suppliers), size=N_ORDERS, p=weights)
    sup = suppliers.iloc[idx].reset_index(drop=True)

    # Order dates uniformly spread over the period
    n_days = (END_DATE - START_DATE).days + 1
    order_date = START_DATE + pd.to_timedelta(rng.integers(0, n_days, N_ORDERS), unit="D")

    # Promised lead time: contract value with small order-level adjustments
    adjust = rng.choice([-1, 0, 0, 0, 1, 2], size=N_ORDERS)
    promised_lt = np.maximum(sup["contract_lead_time_days"].to_numpy() + adjust, 1)

    # Actual lead time: multiplicative noise driven by the supplier tier
    sigma = sup["reliability_tier"].map(lambda t: TIERS[t][1]).to_numpy()
    shift = sup["reliability_tier"].map(lambda t: TIERS[t][2]).to_numpy()
    p_disrupt = sup["reliability_tier"].map(lambda t: TIERS[t][3]).to_numpy()

    noise = np.exp(rng.normal(shift, sigma))
    # Q4 seasonal slowdown (peak season congestion)
    season = np.where(order_date.month >= 10, 1.06, 1.0)
    # Expedited orders arrive faster
    is_expedited = rng.random(N_ORDERS) < 0.08
    expedite = np.where(is_expedited, 0.75, 1.0)
    # Random disruptions add extra days (port delays, shortages, etc.)
    disrupted = rng.random(N_ORDERS) < p_disrupt
    extra = rng.exponential(scale=0.25 * promised_lt) + 1
    actual_lt = promised_lt * noise * season * expedite + np.where(disrupted, extra, 0)
    actual_lt = np.clip(np.rint(actual_lt), 1, 150).astype(int)

    promised_date = order_date + pd.to_timedelta(promised_lt, unit="D")
    delivery_date = order_date + pd.to_timedelta(actual_lt, unit="D")
    delay_days = (delivery_date - promised_date).days

    # Order size and price
    quantity = np.clip(np.rint(rng.lognormal(5.0, 1.0, N_ORDERS)), 10, 5000).astype(int)
    base_price = sup["category"].map(CATEGORIES).to_numpy()
    unit_price = base_price * sup["price_index"].to_numpy() * rng.lognormal(0, 0.08, N_ORDERS)
    unit_price = np.round(unit_price * np.where(is_expedited, 1.10, 1.0), 2)

    orders = pd.DataFrame(
        {
            "supplier_id": sup["supplier_id"],
            "order_date": order_date,
            "promised_date": promised_date,
            "delivery_date": delivery_date,
            "promised_lead_time_days": promised_lt,
            "actual_lead_time_days": actual_lt,
            "delay_days": np.asarray(delay_days),
            "on_time": (np.asarray(delay_days) <= 0).astype(int),
            "order_qty": quantity,
            "unit_price": unit_price,
            "is_expedited": is_expedited.astype(int),
        }
    )
    orders["order_value"] = (orders["order_qty"] * orders["unit_price"]).round(2)

    # Sort chronologically and assign sequential PO ids
    orders = orders.sort_values("order_date").reset_index(drop=True)
    orders.insert(0, "po_id", [f"PO-{i + 1:06d}" for i in range(len(orders))])
    for col in ["order_date", "promised_date", "delivery_date"]:
        orders[col] = orders[col].dt.strftime("%Y-%m-%d")
    return orders


def main() -> None:
    rng = np.random.default_rng(SEED)
    os.makedirs(DATA_DIR, exist_ok=True)

    suppliers = generate_suppliers(rng)
    orders = generate_orders(rng, suppliers)

    suppliers.to_csv(DATA_DIR / "suppliers.csv", index=False)
    orders.to_csv(DATA_DIR / "purchase_orders.csv", index=False)

    print(f"Saved {len(suppliers)} rows to {DATA_DIR / 'suppliers.csv'}")
    print(f"Saved {len(orders)} rows to {DATA_DIR / 'purchase_orders.csv'}")
    print("On-time share of generated orders: "
          f"{orders['on_time'].mean():.3f}")


if __name__ == "__main__":
    main()
