-- Analytical queries for the SYNTHETIC warehouse order picking data (SQLite dialect).
-- Tables: orders (one row per order) and pickers (one row per picker).
-- Run with: python src/run_sql.py
-- Note: statements are separated by semicolons, so do not use a semicolon inside comments.

-- Q1: How many orders are picked per shift, and what are the mean and total picking times?
SELECT
    shift,
    COUNT(*)                          AS orders,
    ROUND(AVG(pick_time_min), 2)      AS mean_time_min,
    ROUND(SUM(pick_time_min) / 60, 1) AS total_hours
FROM orders
GROUP BY shift
ORDER BY mean_time_min DESC;

-- Q2: How do priority and contract type combine? Mean picking time and order count per combination.
SELECT
    o.priority,
    p.contract_type,
    COUNT(*)                     AS orders,
    ROUND(AVG(o.pick_time_min), 2) AS mean_time_min
FROM orders AS o
JOIN pickers AS p ON p.picker_id = o.picker_id
GROUP BY o.priority, p.contract_type
ORDER BY o.priority, mean_time_min DESC;

-- Q3: Does picker experience matter? Mean picking time and mean order size by experience band.
SELECT
    CASE
        WHEN p.experience_months <= 12 THEN '1-12'
        WHEN p.experience_months <= 36 THEN '13-36'
        WHEN p.experience_months <= 72 THEN '37-72'
        ELSE '73-120'
    END                                AS experience_band,
    COUNT(DISTINCT p.picker_id)        AS pickers,
    COUNT(*)                           AS orders,
    ROUND(AVG(o.n_lines), 2)           AS mean_lines,
    ROUND(AVG(o.pick_time_min), 2)     AS mean_time_min,
    ROUND(AVG(o.pick_time_min / o.n_lines), 2) AS mean_min_per_line
FROM orders AS o
JOIN pickers AS p ON p.picker_id = o.picker_id
GROUP BY experience_band
ORDER BY MIN(p.experience_months);

-- Q4: How do walking distance and picking time change with the number of zones visited?
SELECT
    n_zones,
    COUNT(*)                           AS orders,
    ROUND(AVG(n_lines), 2)             AS mean_lines,
    ROUND(AVG(distance_m), 1)          AS mean_distance_m,
    ROUND(AVG(pick_time_min), 2)       AS mean_time_min
FROM orders
GROUP BY n_zones
ORDER BY n_zones;

-- Q5: Which order size classes (by number of lines) take the longest, and what share of all orders are they?
WITH classified AS (
    SELECT
        CASE
            WHEN n_lines <= 2  THEN '1: 1-2 lines'
            WHEN n_lines <= 6  THEN '2: 3-6 lines'
            WHEN n_lines <= 12 THEN '3: 7-12 lines'
            ELSE '4: 13+ lines'
        END AS size_class,
        pick_time_min,
        total_weight_kg
    FROM orders
)
SELECT
    size_class,
    COUNT(*)                                              AS orders,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)    AS share_pct,
    ROUND(AVG(pick_time_min), 2)                          AS mean_time_min,
    ROUND(AVG(total_weight_kg), 2)                        AS mean_weight_kg
FROM classified
GROUP BY size_class
ORDER BY size_class;

-- Q6: How does the mean picking time develop month by month?
SELECT
    STRFTIME('%Y-%m', order_date)      AS month,
    COUNT(*)                           AS orders,
    ROUND(AVG(pick_time_min), 2)       AS mean_time_min,
    ROUND(AVG(n_lines), 2)             AS mean_lines
FROM orders
GROUP BY month
ORDER BY month;

-- Q7: Which 10 pickers have the highest mean picking time per line (pickers with at least 50 orders)?
SELECT
    p.picker_id,
    p.contract_type,
    p.experience_months,
    COUNT(*)                                       AS orders,
    ROUND(AVG(o.pick_time_min), 2)                 AS mean_time_min,
    ROUND(AVG(o.pick_time_min / o.n_lines), 2)     AS mean_min_per_line
FROM orders AS o
JOIN pickers AS p ON p.picker_id = o.picker_id
GROUP BY p.picker_id, p.contract_type, p.experience_months
HAVING COUNT(*) >= 50
ORDER BY mean_min_per_line DESC
LIMIT 10;

-- Q8: Who is the fastest picker (lowest mean time per line) within each contract type?
WITH picker_stats AS (
    SELECT
        p.picker_id,
        p.contract_type,
        p.experience_months,
        COUNT(*)                                   AS orders,
        AVG(o.pick_time_min / o.n_lines)           AS mean_min_per_line
    FROM orders AS o
    JOIN pickers AS p ON p.picker_id = o.picker_id
    GROUP BY p.picker_id, p.contract_type, p.experience_months
    HAVING COUNT(*) >= 50
),
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY contract_type ORDER BY mean_min_per_line
        ) AS rank_in_contract
    FROM picker_stats
)
SELECT
    contract_type,
    picker_id,
    experience_months,
    orders,
    ROUND(mean_min_per_line, 2) AS mean_min_per_line
FROM ranked
WHERE rank_in_contract = 1
ORDER BY contract_type;
