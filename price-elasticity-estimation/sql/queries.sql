-- Price elasticity project: SQL analysis (SQLite dialect).
-- Tables: products (product_id, category, brand_tier, base_price)
--         weekly_sales (product_id, week_start, week_of_year, price, competitor_price,
--                       is_promo, is_holiday_week, units_sold, revenue)
-- All data is synthetic. Each query starts with a comment line "-- Q<n>:" with its question.
-- Run with: python src/run_sql.py

-- Q1: How much revenue does each category generate, and what is its share of total revenue?
SELECT
    p.category,
    COUNT(DISTINCT p.product_id)               AS products,
    ROUND(AVG(s.price), 2)                     AS avg_price,
    SUM(s.units_sold)                          AS total_units,
    ROUND(SUM(s.revenue), 2)                   AS total_revenue,
    ROUND(100.0 * SUM(s.revenue) / SUM(SUM(s.revenue)) OVER (), 1) AS revenue_share_pct
FROM weekly_sales AS s
JOIN products AS p ON p.product_id = s.product_id
GROUP BY p.category
ORDER BY total_revenue DESC;

-- Q2: How deep are the discounts in promo weeks compared with non-promo weeks, per category?
-- The price ratio is the selling price divided by the product's regular base price.
SELECT
    p.category,
    s.is_promo,
    COUNT(*)                                   AS weeks,
    ROUND(AVG(s.price / p.base_price), 3)      AS avg_price_ratio,
    ROUND(AVG(s.units_sold), 1)                AS avg_units
FROM weekly_sales AS s
JOIN products AS p ON p.product_id = s.product_id
GROUP BY p.category, s.is_promo
ORDER BY p.category, s.is_promo;

-- Q3: What is the raw promo uplift in average weekly units per category?
-- This is an uncontrolled comparison (products differ in size), so it is only indicative.
WITH per_product AS (
    SELECT
        product_id,
        AVG(CASE WHEN is_promo = 1 THEN units_sold END) AS promo_units,
        AVG(CASE WHEN is_promo = 0 THEN units_sold END) AS regular_units
    FROM weekly_sales
    GROUP BY product_id
)
SELECT
    p.category,
    ROUND(AVG(pp.promo_units), 1)              AS avg_units_promo,
    ROUND(AVG(pp.regular_units), 1)            AS avg_units_regular,
    ROUND(AVG(pp.promo_units / pp.regular_units), 2) AS avg_promo_to_regular_ratio
FROM per_product AS pp
JOIN products AS p ON p.product_id = pp.product_id
WHERE pp.promo_units IS NOT NULL
GROUP BY p.category
ORDER BY avg_promo_to_regular_ratio DESC;

-- Q4: How do brand tiers differ in base price, average units and revenue?
SELECT
    p.brand_tier,
    COUNT(DISTINCT p.product_id)               AS products,
    ROUND(AVG(p.base_price), 2)                AS avg_base_price,
    ROUND(AVG(s.units_sold), 1)                AS avg_weekly_units,
    ROUND(SUM(s.revenue), 2)                   AS total_revenue
FROM products AS p
JOIN weekly_sales AS s ON s.product_id = p.product_id
GROUP BY p.brand_tier
ORDER BY avg_base_price;

-- Q5: How much higher are average units in holiday weeks than in regular weeks, per category?
SELECT
    p.category,
    ROUND(AVG(CASE WHEN s.is_holiday_week = 1 THEN s.units_sold END), 1) AS avg_units_holiday,
    ROUND(AVG(CASE WHEN s.is_holiday_week = 0 THEN s.units_sold END), 1) AS avg_units_regular,
    ROUND(AVG(CASE WHEN s.is_holiday_week = 1 THEN s.units_sold END) * 1.0
          / AVG(CASE WHEN s.is_holiday_week = 0 THEN s.units_sold END), 3) AS holiday_to_regular_ratio
FROM weekly_sales AS s
JOIN products AS p ON p.product_id = s.product_id
GROUP BY p.category
ORDER BY holiday_to_regular_ratio DESC;

-- Q6: Which 10 products have the highest revenue, and how do they rank within their category?
SELECT
    p.product_id,
    p.category,
    p.brand_tier,
    ROUND(SUM(s.revenue), 2)                   AS total_revenue,
    RANK() OVER (ORDER BY SUM(s.revenue) DESC) AS overall_rank,
    RANK() OVER (PARTITION BY p.category ORDER BY SUM(s.revenue) DESC) AS rank_in_category
FROM weekly_sales AS s
JOIN products AS p ON p.product_id = s.product_id
GROUP BY p.product_id, p.category, p.brand_tier
ORDER BY overall_rank
LIMIT 10;

-- Q7: How does total revenue develop month by month, and what is the change vs. the previous month?
-- Months are taken from the week start date (the week is assigned to the month it starts in).
WITH monthly AS (
    SELECT
        SUBSTR(week_start, 1, 7)               AS month,
        SUM(revenue)                           AS revenue
    FROM weekly_sales
    GROUP BY SUBSTR(week_start, 1, 7)
)
SELECT
    month,
    ROUND(revenue, 2)                          AS revenue,
    ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
          / LAG(revenue) OVER (ORDER BY month), 1) AS change_vs_prev_month_pct
FROM monthly
ORDER BY month;

-- Q8: How do average units change across price-ratio buckets within each category?
-- Raw descriptive view of the price-demand relation, before any regression controls.
WITH bucketed AS (
    SELECT
        p.category,
        CASE
            WHEN s.price / p.base_price < 0.80 THEN '1: below 0.80'
            WHEN s.price / p.base_price < 0.95 THEN '2: 0.80 to 0.95'
            WHEN s.price / p.base_price < 1.05 THEN '3: 0.95 to 1.05'
            ELSE '4: 1.05 and above'
        END                                    AS price_ratio_bucket,
        s.units_sold
    FROM weekly_sales AS s
    JOIN products AS p ON p.product_id = s.product_id
)
SELECT
    category,
    price_ratio_bucket,
    COUNT(*)                                   AS weeks,
    ROUND(AVG(units_sold), 1)                  AS avg_units
FROM bucketed
GROUP BY category, price_ratio_bucket
ORDER BY category, price_ratio_bucket;
