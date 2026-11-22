-- SQL queries for the SYNTHETIC landing page A/B test (SQLite dialect).
-- Tables: users (one row per visitor) and orders (one row per order).
-- Run them with: python src/run_sql.py
-- Note: "group" is a reserved word in SQL, so it is always quoted.
-- Note: do not put semicolons inside comments or strings, because
-- src/run_sql.py splits this file into statements on the semicolon character.

-- Q1: How many visitors and buyers does each group have, and what is its conversion rate?
SELECT
    "group",
    COUNT(*)                                  AS visitors,
    SUM(converted)                            AS buyers,
    ROUND(100.0 * SUM(converted) / COUNT(*), 2) AS conversion_pct
FROM users
GROUP BY "group"
ORDER BY "group";

-- Q2: What are revenue per visitor and average order value by group?
-- Revenue per visitor divides by ALL visitors (non-buyers count as 0), while
-- average order value divides by buyers only. Each buyer has exactly one order.
SELECT
    u."group",
    COUNT(*)                                        AS visitors,
    COUNT(o.order_id)                               AS orders,
    ROUND(COALESCE(SUM(o.order_value), 0), 2)       AS total_revenue,
    ROUND(COALESCE(SUM(o.order_value), 0) * 1.0 / COUNT(*), 2) AS revenue_per_visitor,
    ROUND(AVG(o.order_value), 2)                    AS avg_order_value
FROM users AS u
LEFT JOIN orders AS o ON o.user_id = u.user_id
GROUP BY u."group"
ORDER BY u."group";

-- Q3: How does conversion differ by device and group?
-- AVG ignores NULLs, so each CASE expression averages only one group.
SELECT
    device,
    COUNT(*) AS visitors,
    ROUND(100.0 * AVG(CASE WHEN "group" = 'control' THEN converted END), 2) AS control_pct,
    ROUND(100.0 * AVG(CASE WHEN "group" = 'variant' THEN converted END), 2) AS variant_pct,
    ROUND(100.0 * (AVG(CASE WHEN "group" = 'variant' THEN converted END)
                 - AVG(CASE WHEN "group" = 'control' THEN converted END)), 2) AS diff_pp
FROM users
GROUP BY device
ORDER BY visitors DESC;

-- Q4: How does conversion differ by traffic source and group, sorted by the difference?
-- Segments are small, so differences here are exploratory and noisy.
SELECT
    traffic_source,
    COUNT(*) AS visitors,
    ROUND(100.0 * AVG(CASE WHEN "group" = 'control' THEN converted END), 2) AS control_pct,
    ROUND(100.0 * AVG(CASE WHEN "group" = 'variant' THEN converted END), 2) AS variant_pct,
    ROUND(100.0 * (AVG(CASE WHEN "group" = 'variant' THEN converted END)
                 - AVG(CASE WHEN "group" = 'control' THEN converted END)), 2) AS diff_pp
FROM users
GROUP BY traffic_source
ORDER BY diff_pp DESC;

-- Q5: Is the country mix balanced between groups (share of each group's visitors)?
-- A randomized experiment should show similar shares in both groups.
SELECT
    u.country,
    u."group",
    COUNT(*) AS visitors,
    ROUND(100.0 * COUNT(*) / t.n, 2) AS share_of_group_pct
FROM users AS u
JOIN (
    SELECT "group" AS g, COUNT(*) AS n
    FROM users
    GROUP BY "group"
) AS t ON t.g = u."group"
GROUP BY u.country, u."group"
ORDER BY u.country, u."group";

-- Q6: How many visitors and conversions does each day have, and what is the daily conversion rate?
SELECT
    visit_date,
    COUNT(*)                                    AS visitors,
    SUM(converted)                              AS buyers,
    ROUND(100.0 * SUM(converted) / COUNT(*), 2) AS conversion_pct
FROM users
GROUP BY visit_date
ORDER BY visit_date;

-- Q7: How are orders distributed by number of items, and what is the average order value per group?
SELECT
    o.items,
    u."group",
    COUNT(*)                     AS orders,
    ROUND(AVG(o.order_value), 2) AS avg_order_value,
    ROUND(MIN(o.order_value), 2) AS min_order_value,
    ROUND(MAX(o.order_value), 2) AS max_order_value
FROM orders AS o
JOIN users AS u ON u.user_id = o.user_id
GROUP BY o.items, u."group"
ORDER BY o.items, u."group";

-- Q8: How many days pass between the visit and the order in each group?
SELECT
    u."group",
    CAST(julianday(o.order_date) - julianday(u.visit_date) AS INTEGER) AS days_to_order,
    COUNT(*) AS orders
FROM orders AS o
JOIN users AS u ON u.user_id = o.user_id
GROUP BY u."group", days_to_order
ORDER BY u."group", days_to_order;
