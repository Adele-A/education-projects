-- SQL analysis for the SYNTHETIC CLV dataset (SQLite dialect).
-- Tables: customers(customer_id, acquisition_date, acquisition_channel, country)
--         transactions(transaction_id, customer_id, transaction_date, order_value)
-- Dates are stored as text in the format YYYY-MM-DD.
-- Each query is preceded by a line starting with "-- Q<n>:" that states the question.
-- Statements are separated by a semicolon, so comments must not contain one.

-- Q1: What are the overall totals (customers, orders, revenue, average order value)?
SELECT
    (SELECT COUNT(*) FROM customers)                      AS customers,
    COUNT(*)                                              AS orders,
    ROUND(SUM(order_value), 2)                            AS total_revenue,
    ROUND(AVG(order_value), 2)                            AS avg_order_value,
    ROUND(1.0 * COUNT(*) / COUNT(DISTINCT customer_id), 2) AS orders_per_customer
FROM transactions;

-- Q2: How many customers placed 1, 2, 3, ... orders (5 or more are grouped)?
WITH per_customer AS (
    SELECT customer_id, COUNT(*) AS n_orders
    FROM transactions
    GROUP BY customer_id
)
SELECT
    CASE WHEN n_orders >= 5 THEN '5+' ELSE CAST(n_orders AS TEXT) END AS orders_bucket,
    COUNT(*)                                                          AS customers,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM per_customer), 1)  AS pct_of_customers
FROM per_customer
GROUP BY orders_bucket
ORDER BY orders_bucket;

-- Q3: How do acquisition channels compare in orders and revenue per customer?
SELECT
    c.acquisition_channel,
    COUNT(DISTINCT c.customer_id)                                    AS customers,
    COUNT(t.transaction_id)                                          AS orders,
    ROUND(1.0 * COUNT(t.transaction_id) / COUNT(DISTINCT c.customer_id), 2) AS avg_orders,
    ROUND(SUM(t.order_value) / COUNT(DISTINCT c.customer_id), 2)     AS avg_revenue_per_customer,
    ROUND(AVG(t.order_value), 2)                                     AS avg_order_value
FROM customers AS c
LEFT JOIN transactions AS t ON t.customer_id = c.customer_id
GROUP BY c.acquisition_channel
ORDER BY avg_revenue_per_customer DESC;

-- Q4: What is the revenue per acquisition cohort (quarter)? Later cohorts were observed for less time.
SELECT
    substr(c.acquisition_date, 1, 4) || '-Q' ||
        ((CAST(substr(c.acquisition_date, 6, 2) AS INTEGER) + 2) / 3) AS cohort_quarter,
    COUNT(DISTINCT c.customer_id)                                    AS customers,
    ROUND(SUM(t.order_value), 2)                                     AS total_revenue,
    ROUND(SUM(t.order_value) / COUNT(DISTINCT c.customer_id), 2)     AS revenue_per_customer
FROM customers AS c
JOIN transactions AS t ON t.customer_id = c.customer_id
GROUP BY cohort_quarter
ORDER BY cohort_quarter;

-- Q5: What share of customers in each country made a repeat purchase?
WITH per_customer AS (
    SELECT c.customer_id, c.country, COUNT(t.transaction_id) AS n_orders
    FROM customers AS c
    LEFT JOIN transactions AS t ON t.customer_id = c.customer_id
    GROUP BY c.customer_id, c.country
)
SELECT
    country,
    COUNT(*)                                                       AS customers,
    SUM(CASE WHEN n_orders > 1 THEN 1 ELSE 0 END)                  AS repeat_customers,
    ROUND(100.0 * SUM(CASE WHEN n_orders > 1 THEN 1 ELSE 0 END) / COUNT(*), 1) AS repeat_rate_pct
FROM per_customer
GROUP BY country
ORDER BY repeat_rate_pct DESC;

-- Q6: What is the revenue per calendar month with a running total?
WITH monthly AS (
    SELECT
        substr(transaction_date, 1, 7) AS month,
        COUNT(*)                       AS orders,
        SUM(order_value)               AS revenue
    FROM transactions
    GROUP BY month
)
SELECT
    month,
    orders,
    ROUND(revenue, 2)                                  AS revenue,
    ROUND(SUM(revenue) OVER (ORDER BY month), 2)       AS cumulative_revenue
FROM monthly
ORDER BY month;

-- Q7: Who are the top 10 customers by historical revenue, and when did they last order?
SELECT
    c.customer_id,
    c.acquisition_channel,
    c.country,
    COUNT(t.transaction_id)                                   AS orders,
    ROUND(SUM(t.order_value), 2)                              AS revenue,
    MAX(t.transaction_date)                                   AS last_order_date,
    CAST(julianday('2025-12-31') - julianday(MAX(t.transaction_date)) AS INTEGER) AS days_since_last_order
FROM customers AS c
JOIN transactions AS t ON t.customer_id = c.customer_id
GROUP BY c.customer_id, c.acquisition_channel, c.country
ORDER BY revenue DESC
LIMIT 10;

-- Q8: What share of total revenue comes from the top 10 percent of customers?
WITH per_customer AS (
    SELECT customer_id, SUM(order_value) AS revenue
    FROM transactions
    GROUP BY customer_id
),
ranked AS (
    SELECT
        customer_id,
        revenue,
        ROW_NUMBER() OVER (ORDER BY revenue DESC) AS rank_no,
        COUNT(*) OVER ()                          AS n_customers
    FROM per_customer
)
SELECT
    MAX(n_customers)                                                          AS customers,
    SUM(CASE WHEN rank_no <= (n_customers + 9) / 10 THEN 1 ELSE 0 END)        AS top_group_size,
    ROUND(SUM(CASE WHEN rank_no <= (n_customers + 9) / 10 THEN revenue ELSE 0 END), 2) AS top_group_revenue,
    ROUND(SUM(revenue), 2)                                                    AS total_revenue,
    ROUND(100.0 * SUM(CASE WHEN rank_no <= (n_customers + 9) / 10 THEN revenue ELSE 0 END)
          / SUM(revenue), 1)                                                  AS top_group_share_pct
FROM ranked;
