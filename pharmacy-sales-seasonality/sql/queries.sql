-- SQL analysis of the SYNTHETIC pharmacy sales data (SQLite dialect).
-- Tables: categories(category, seasonality_pattern, base_weekly_units, base_unit_price)
--         weekly_sales(week_start, category, units_sold, unit_price, promo_flag, revenue, avg_temp_c)
-- week_start is stored as text 'YYYY-MM-DD', so strftime() works on it.
-- Run with: python src/run_sql.py
-- Note: statements are separated by a semicolon, so do not use one inside comments.

-- Q1: How large is each category in total units and revenue, and what is its revenue rank?
SELECT
    s.category,
    c.seasonality_pattern,
    SUM(s.units_sold)                       AS total_units,
    ROUND(SUM(s.revenue), 2)                AS total_revenue,
    ROUND(AVG(s.units_sold), 1)             AS avg_weekly_units,
    RANK() OVER (ORDER BY SUM(s.revenue) DESC) AS revenue_rank
FROM weekly_sales AS s
JOIN categories AS c ON c.category = s.category
GROUP BY s.category, c.seasonality_pattern
ORDER BY revenue_rank;

-- Q2: What are the monthly totals and the seasonal index (1.00 = average month) for cold_and_flu?
WITH monthly AS (
    SELECT
        CAST(strftime('%m', week_start) AS INTEGER) AS month,
        SUM(units_sold)                             AS total_units,
        AVG(units_sold)                             AS avg_weekly_units
    FROM weekly_sales
    WHERE category = 'cold_and_flu'
    GROUP BY month
)
SELECT
    month,
    total_units,
    ROUND(avg_weekly_units, 1)                                AS avg_weekly_units,
    ROUND(avg_weekly_units / AVG(avg_weekly_units) OVER (), 2) AS seasonal_index
FROM monthly
ORDER BY month;

-- Q3: Which month is the peak and which the trough for every category, and how strong is the swing?
WITH monthly AS (
    SELECT
        category,
        CAST(strftime('%m', week_start) AS INTEGER) AS month,
        AVG(units_sold)                             AS avg_units
    FROM weekly_sales
    GROUP BY category, month
),
idx AS (
    SELECT
        category,
        month,
        avg_units / AVG(avg_units) OVER (PARTITION BY category) AS seasonal_index
    FROM monthly
),
ranked AS (
    SELECT
        category,
        month,
        seasonal_index,
        ROW_NUMBER() OVER (PARTITION BY category ORDER BY seasonal_index DESC) AS rn_peak,
        ROW_NUMBER() OVER (PARTITION BY category ORDER BY seasonal_index ASC)  AS rn_trough
    FROM idx
)
SELECT
    category,
    MAX(CASE WHEN rn_peak = 1   THEN month END)                  AS peak_month,
    ROUND(MAX(CASE WHEN rn_peak = 1   THEN seasonal_index END), 2) AS peak_index,
    MAX(CASE WHEN rn_trough = 1 THEN month END)                  AS trough_month,
    ROUND(MAX(CASE WHEN rn_trough = 1 THEN seasonal_index END), 2) AS trough_index,
    ROUND(MAX(CASE WHEN rn_peak = 1 THEN seasonal_index END)
        / MAX(CASE WHEN rn_trough = 1 THEN seasonal_index END), 2) AS peak_to_trough
FROM ranked
GROUP BY category
ORDER BY peak_to_trough DESC;

-- Q4: How does average weekly demand grow year over year in each category?
WITH yearly AS (
    SELECT
        category,
        CAST(strftime('%Y', week_start) AS INTEGER) AS year,
        COUNT(*)                                    AS weeks,
        AVG(units_sold)                             AS avg_weekly_units
    FROM weekly_sales
    GROUP BY category, year
)
SELECT
    category,
    year,
    weeks,
    ROUND(avg_weekly_units, 1) AS avg_weekly_units,
    ROUND(100.0 * (avg_weekly_units / LAG(avg_weekly_units) OVER (PARTITION BY category ORDER BY year) - 1), 2)
        AS yoy_growth_pct
FROM yearly
ORDER BY category, year;

-- Q5: How much higher are units in promo weeks than in non-promo weeks of the same month?
-- Comparing within the same category and month removes most of the seasonal effect.
WITH cm AS (
    SELECT
        category,
        CAST(strftime('%m', week_start) AS INTEGER) AS month,
        AVG(CASE WHEN promo_flag = 1 THEN units_sold END) AS promo_units,
        AVG(CASE WHEN promo_flag = 0 THEN units_sold END) AS regular_units
    FROM weekly_sales
    GROUP BY category, month
)
SELECT
    category,
    COUNT(*)                                       AS months_compared,
    ROUND(AVG(promo_units / regular_units), 3)     AS avg_promo_ratio,
    ROUND(100 * (AVG(promo_units / regular_units) - 1), 1) AS avg_uplift_pct
FROM cm
WHERE promo_units IS NOT NULL AND regular_units IS NOT NULL
GROUP BY category
ORDER BY avg_promo_ratio DESC;

-- Q6: How do average weekly units change across temperature bands in each category?
SELECT
    category,
    CASE
        WHEN avg_temp_c < 5  THEN '1: below 5 C'
        WHEN avg_temp_c < 15 THEN '2: 5 to 15 C'
        WHEN avg_temp_c < 22 THEN '3: 15 to 22 C'
        ELSE                      '4: 22 C and above'
    END                          AS temp_band,
    COUNT(*)                     AS weeks,
    ROUND(AVG(units_sold), 1)    AS avg_weekly_units
FROM weekly_sales
GROUP BY category, temp_band
ORDER BY category, temp_band;

-- Q7: What share of each year's revenue does each category contribute?
WITH yearly AS (
    SELECT
        CAST(strftime('%Y', week_start) AS INTEGER) AS year,
        category,
        SUM(revenue)                                AS revenue
    FROM weekly_sales
    GROUP BY year, category
)
SELECT
    year,
    category,
    ROUND(revenue, 0)                                              AS revenue,
    ROUND(100.0 * revenue / SUM(revenue) OVER (PARTITION BY year), 1) AS revenue_share_pct
FROM yearly
ORDER BY year, revenue_share_pct DESC;
