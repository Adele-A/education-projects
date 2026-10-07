-- SQL analysis of the synthetic weekly retail sales data (SQLite dialect).
-- Tables: stores(store_id, region, size_sqm, opened_year)
--         weekly_sales(week_start, store_id, category, units_sold, unit_price,
--                      promo_flag, holiday_week, revenue)
-- week_start is stored as a text date in the format YYYY-MM-DD.
-- Each query is preceded by a comment line that starts with "Q<n>:".
-- Statements are separated by a semicolon at the end of each query.

-- Q1: What is the total revenue and number of units per month, across all stores and categories?
SELECT
    strftime('%Y-%m', week_start) AS month,
    SUM(units_sold)               AS units,
    ROUND(SUM(revenue), 2)        AS revenue
FROM weekly_sales
GROUP BY strftime('%Y-%m', week_start)
ORDER BY month;

-- Q2: How does yearly revenue per category change compared with the previous year?
WITH yearly AS (
    SELECT
        category,
        CAST(strftime('%Y', week_start) AS INTEGER) AS year,
        SUM(revenue)                                AS revenue
    FROM weekly_sales
    GROUP BY category, CAST(strftime('%Y', week_start) AS INTEGER)
)
SELECT
    category,
    year,
    ROUND(revenue, 2) AS revenue,
    ROUND((revenue / LAG(revenue) OVER (PARTITION BY category ORDER BY year) - 1) * 100, 2)
        AS yoy_growth_pct
FROM yearly
ORDER BY category, year;

-- Q3: How do the stores rank by total revenue, and how does revenue per square meter compare?
SELECT
    RANK() OVER (ORDER BY SUM(w.revenue) DESC)  AS revenue_rank,
    s.store_id,
    s.region,
    s.size_sqm,
    ROUND(SUM(w.revenue), 2)                    AS revenue,
    ROUND(SUM(w.revenue) / s.size_sqm, 2)       AS revenue_per_sqm
FROM weekly_sales AS w
JOIN stores AS s ON s.store_id = w.store_id
GROUP BY s.store_id, s.region, s.size_sqm
ORDER BY revenue_rank;

-- Q4: Which category brings the most revenue in each store?
WITH store_cat AS (
    SELECT
        store_id,
        category,
        SUM(revenue) AS revenue
    FROM weekly_sales
    GROUP BY store_id, category
),
ranked AS (
    SELECT
        store_id,
        category,
        revenue,
        RANK() OVER (PARTITION BY store_id ORDER BY revenue DESC) AS rnk
    FROM store_cat
)
SELECT store_id, category AS top_category, ROUND(revenue, 2) AS revenue
FROM ranked
WHERE rnk = 1
ORDER BY store_id;

-- Q5: How much higher are units in promotion weeks than in regular weeks, per category (relative to the store and category average)?
WITH base AS (
    SELECT store_id, category, AVG(units_sold) AS avg_units
    FROM weekly_sales
    GROUP BY store_id, category
),
rel AS (
    SELECT
        w.category,
        w.promo_flag,
        AVG(w.units_sold * 1.0 / b.avg_units) AS rel_units
    FROM weekly_sales AS w
    JOIN base AS b ON b.store_id = w.store_id AND b.category = w.category
    GROUP BY w.category, w.promo_flag
)
SELECT
    p.category,
    ROUND(r.rel_units, 3)                    AS rel_units_regular,
    ROUND(p.rel_units, 3)                    AS rel_units_promo,
    ROUND((p.rel_units / r.rel_units - 1) * 100, 1) AS uplift_pct
FROM rel AS p
JOIN rel AS r ON r.category = p.category AND r.promo_flag = 0
WHERE p.promo_flag = 1
ORDER BY uplift_pct DESC;

-- Q6: How do holiday weeks differ from regular weeks in average weekly units per store, by category?
SELECT
    category,
    ROUND(AVG(CASE WHEN holiday_week = 0 THEN units_sold END), 1) AS avg_units_regular,
    ROUND(AVG(CASE WHEN holiday_week = 1 THEN units_sold END), 1) AS avg_units_holiday,
    COUNT(DISTINCT CASE WHEN holiday_week = 1 THEN week_start END) AS holiday_weeks
FROM weekly_sales
GROUP BY category
ORDER BY category;

-- Q7: What share of regional revenue does each category contribute?
WITH region_cat AS (
    SELECT
        s.region,
        w.category,
        SUM(w.revenue) AS revenue
    FROM weekly_sales AS w
    JOIN stores AS s ON s.store_id = w.store_id
    GROUP BY s.region, w.category
)
SELECT
    region,
    category,
    ROUND(revenue, 2) AS revenue,
    ROUND(100.0 * revenue / SUM(revenue) OVER (PARTITION BY region), 1) AS share_of_region_pct
FROM region_cat
ORDER BY region, share_of_region_pct DESC;

-- Q8: Which five weeks had the highest total revenue, and were they holiday or promotion weeks?
SELECT
    week_start,
    ROUND(SUM(revenue), 2)   AS revenue,
    MAX(holiday_week)        AS holiday_week,
    SUM(promo_flag)          AS promo_rows
FROM weekly_sales
GROUP BY week_start
ORDER BY revenue DESC
LIMIT 5;
