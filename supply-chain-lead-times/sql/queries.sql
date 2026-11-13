-- SQL analysis of supplier on-time delivery (SQLite dialect).
-- Tables: suppliers and purchase_orders, loaded from the synthetic CSV files.
-- Dates are stored as YYYY-MM-DD text. Every query is preceded by a "-- Q<n>:" line.
-- Statements are separated by a semicolon, so no comment may contain one.

-- Q1: What is the overall on-time rate by reliability tier and transport mode?
SELECT
    s.reliability_tier,
    s.transport_mode,
    COUNT(*)                           AS orders,
    ROUND(AVG(po.on_time), 3)          AS on_time_rate,
    ROUND(AVG(po.delay_days), 2)       AS mean_delay_days
FROM purchase_orders AS po
JOIN suppliers AS s ON s.supplier_id = po.supplier_id
GROUP BY s.reliability_tier, s.transport_mode
ORDER BY s.reliability_tier, on_time_rate DESC;

-- Q2: Which suppliers with at least 30 orders have the lowest on-time rate (ranking)?
SELECT
    RANK() OVER (ORDER BY AVG(po.on_time) ASC) AS worst_rank,
    s.supplier_id,
    s.country,
    s.reliability_tier,
    COUNT(*)                                   AS orders,
    ROUND(AVG(po.on_time), 3)                  AS on_time_rate,
    ROUND(AVG(po.actual_lead_time_days), 1)    AS mean_lead_time
FROM purchase_orders AS po
JOIN suppliers AS s ON s.supplier_id = po.supplier_id
GROUP BY s.supplier_id, s.country, s.reliability_tier
HAVING COUNT(*) >= 30
ORDER BY worst_rank, s.supplier_id
LIMIT 15;

-- Q3: How does the monthly on-time rate evolve, with a 3-month moving average?
WITH monthly AS (
    SELECT
        SUBSTR(order_date, 1, 7) AS order_month,
        COUNT(*)                 AS orders,
        AVG(on_time)             AS on_time_rate
    FROM purchase_orders
    GROUP BY SUBSTR(order_date, 1, 7)
)
SELECT
    order_month,
    orders,
    ROUND(on_time_rate, 3) AS on_time_rate,
    ROUND(AVG(on_time_rate) OVER (
        ORDER BY order_month
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 3) AS on_time_rate_3m_avg
FROM monthly
ORDER BY order_month;

-- Q4: How variable are lead times by country (std and coefficient of variation)?
-- SQLite has no STDDEV function, so the population std is computed from averages.
WITH stats AS (
    SELECT
        s.country,
        COUNT(*)                                          AS orders,
        AVG(po.actual_lead_time_days)                     AS mean_lt,
        AVG(po.actual_lead_time_days * po.actual_lead_time_days) AS mean_sq_lt,
        AVG(po.on_time)                                   AS on_time_rate
    FROM purchase_orders AS po
    JOIN suppliers AS s ON s.supplier_id = po.supplier_id
    GROUP BY s.country
)
SELECT
    country,
    orders,
    ROUND(mean_lt, 2)                                            AS mean_lead_time,
    ROUND(SQRT(MAX(mean_sq_lt - mean_lt * mean_lt, 0)), 2)       AS std_lead_time,
    ROUND(SQRT(MAX(mean_sq_lt - mean_lt * mean_lt, 0)) / mean_lt, 3) AS cv_lead_time,
    ROUND(on_time_rate, 3)                                       AS on_time_rate
FROM stats
ORDER BY cv_lead_time DESC;

-- Q5: Is on-time performance worse for orders placed in Q4 than in the rest of the year?
SELECT
    CASE WHEN CAST(STRFTIME('%m', order_date) AS INTEGER) >= 10
         THEN 'Q4' ELSE 'Q1-Q3' END     AS season,
    COUNT(*)                            AS orders,
    ROUND(AVG(on_time), 3)              AS on_time_rate,
    ROUND(AVG(delay_days), 2)           AS mean_delay_days
FROM purchase_orders
GROUP BY season
ORDER BY season;

-- Q6: Do expedited orders arrive on time more often than standard orders?
SELECT
    CASE WHEN is_expedited = 1 THEN 'expedited' ELSE 'standard' END AS order_type,
    COUNT(*)                                  AS orders,
    ROUND(AVG(on_time), 3)                    AS on_time_rate,
    ROUND(AVG(actual_lead_time_days), 1)      AS mean_lead_time,
    ROUND(AVG(promised_lead_time_days), 1)    AS mean_promised_lead_time
FROM purchase_orders
GROUP BY is_expedited
ORDER BY is_expedited;

-- Q7: Which suppliers with at least 30 orders perform worst relative to their own tier average?
WITH supplier_stats AS (
    SELECT
        s.supplier_id,
        s.reliability_tier,
        s.country,
        COUNT(*)     AS orders,
        AVG(po.on_time) AS on_time_rate
    FROM purchase_orders AS po
    JOIN suppliers AS s ON s.supplier_id = po.supplier_id
    GROUP BY s.supplier_id, s.reliability_tier, s.country
),
tier_stats AS (
    SELECT
        s.reliability_tier,
        AVG(po.on_time) AS tier_on_time_rate
    FROM purchase_orders AS po
    JOIN suppliers AS s ON s.supplier_id = po.supplier_id
    GROUP BY s.reliability_tier
)
SELECT
    ss.supplier_id,
    ss.country,
    ss.reliability_tier,
    ss.orders,
    ROUND(ss.on_time_rate, 3)                       AS on_time_rate,
    ROUND(ts.tier_on_time_rate, 3)                  AS tier_on_time_rate,
    ROUND(ss.on_time_rate - ts.tier_on_time_rate, 3) AS gap_vs_tier
FROM supplier_stats AS ss
JOIN tier_stats AS ts ON ts.reliability_tier = ss.reliability_tier
WHERE ss.orders >= 30
ORDER BY gap_vs_tier ASC
LIMIT 10;

-- Q8: What share of order value is delivered late, by product category?
SELECT
    s.category,
    COUNT(*)                                        AS orders,
    ROUND(SUM(po.order_value), 0)                   AS total_order_value,
    ROUND(SUM(CASE WHEN po.on_time = 0 THEN po.order_value ELSE 0 END), 0) AS late_order_value,
    ROUND(1.0 * SUM(CASE WHEN po.on_time = 0 THEN po.order_value ELSE 0 END)
          / SUM(po.order_value), 3)                 AS late_value_share
FROM purchase_orders AS po
JOIN suppliers AS s ON s.supplier_id = po.supplier_id
GROUP BY s.category
ORDER BY late_value_share DESC;
