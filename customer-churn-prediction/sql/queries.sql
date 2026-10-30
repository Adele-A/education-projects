-- Customer Churn Prediction: analytical queries (SQLite dialect).
-- Tables: customers, support_tickets (SYNTHETIC data loaded from data/*.csv).
-- Each query starts with a description line and ends with a semicolon.
-- Avoid semicolons inside comments, because the runner splits on them.

-- Q1: What are the customer count and churn rate for each contract type?
SELECT
    contract_type,
    COUNT(*) AS customers,
    ROUND(AVG(churned), 3) AS churn_rate
FROM customers
GROUP BY contract_type
ORDER BY churn_rate DESC;

-- Q2: How do churn rate and average monthly price differ by plan and discount flag?
SELECT
    plan,
    discount_flag,
    COUNT(*) AS customers,
    ROUND(AVG(monthly_price), 2) AS avg_monthly_price,
    ROUND(AVG(churned), 3) AS churn_rate
FROM customers
GROUP BY plan, discount_flag
ORDER BY plan, discount_flag;

-- Q3: How does churn rate change across tenure bands?
SELECT
    CASE
        WHEN tenure_months <= 6  THEN '1) 1-6 months'
        WHEN tenure_months <= 12 THEN '2) 7-12 months'
        WHEN tenure_months <= 24 THEN '3) 13-24 months'
        WHEN tenure_months <= 36 THEN '4) 25-36 months'
        ELSE '5) 37-60 months'
    END AS tenure_band,
    COUNT(*) AS customers,
    ROUND(AVG(churned), 3) AS churn_rate
FROM customers
GROUP BY tenure_band
ORDER BY tenure_band;

-- Q4: Is a higher support ticket load associated with higher churn?
WITH ticket_counts AS (
    SELECT customer_id, COUNT(*) AS n_tickets
    FROM support_tickets
    GROUP BY customer_id
)
SELECT
    CASE
        WHEN COALESCE(t.n_tickets, 0) >= 3 THEN '3+'
        ELSE CAST(COALESCE(t.n_tickets, 0) AS TEXT)
    END AS ticket_group,
    COUNT(*) AS customers,
    ROUND(AVG(c.churned), 3) AS churn_rate
FROM customers AS c
LEFT JOIN ticket_counts AS t ON t.customer_id = c.customer_id
GROUP BY ticket_group
ORDER BY ticket_group;

-- Q5: Do customers with at least one Billing ticket churn more than those without?
WITH billing AS (
    SELECT customer_id, COUNT(*) AS n_billing
    FROM support_tickets
    WHERE category = 'Billing'
    GROUP BY customer_id
)
SELECT
    CASE WHEN b.n_billing IS NULL THEN 'No billing ticket' ELSE 'Has billing ticket' END AS billing_group,
    COUNT(*) AS customers,
    ROUND(AVG(c.late_payments_6m), 2) AS avg_late_payments,
    ROUND(AVG(c.churned), 3) AS churn_rate
FROM customers AS c
LEFT JOIN billing AS b ON b.customer_id = c.customer_id
GROUP BY billing_group
ORDER BY churn_rate DESC;

-- Q6: Are resolution time and satisfaction different for tickets of churned vs retained customers?
SELECT
    c.churned,
    COUNT(*) AS tickets,
    ROUND(AVG(t.resolution_hours), 2) AS avg_resolution_hours,
    ROUND(AVG(t.satisfaction_score), 2) AS avg_satisfaction,
    ROUND(AVG(CASE WHEN t.priority = 'High' THEN 1.0 ELSE 0.0 END), 3) AS share_high_priority
FROM support_tickets AS t
INNER JOIN customers AS c ON c.customer_id = t.customer_id
GROUP BY c.churned
ORDER BY c.churned;

-- Q7: How risky is the segment of month-to-month customers without auto-renew and with 2+ late payments?
SELECT
    CASE
        WHEN contract_type = 'Month-to-month' AND auto_renew = 0 AND late_payments_6m >= 2
            THEN 'Risk segment'
        ELSE 'All other customers'
    END AS segment,
    COUNT(*) AS customers,
    SUM(churned) AS churned_customers,
    ROUND(AVG(churned), 3) AS churn_rate
FROM customers
GROUP BY segment
ORDER BY churn_rate DESC;

-- Q8: Which countries have the highest churn rate, and how do they rank against the overall rate?
WITH by_country AS (
    SELECT
        country,
        COUNT(*) AS customers,
        AVG(churned) AS churn_rate
    FROM customers
    GROUP BY country
)
SELECT
    RANK() OVER (ORDER BY churn_rate DESC) AS risk_rank,
    country,
    customers,
    ROUND(churn_rate, 3) AS churn_rate,
    ROUND(churn_rate - (SELECT AVG(churned) FROM customers), 3) AS diff_vs_overall
FROM by_country
ORDER BY risk_rank;
