-- Analytical queries for the SYNTHETIC fraud data (SQLite dialect).
-- Tables: customers, transactions (loaded from data/*.csv by src/run_sql.py).
-- Format: every query is preceded by a comment line starting with '-- Q<n>:'
-- and ends with a semicolon. Do not use semicolons inside comments, because
-- the runner splits the file on them.

-- Q1: What is the class balance, i.e. how many transactions are fraudulent?
SELECT
    is_fraud,
    COUNT(*) AS transactions,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM transactions), 2) AS share_pct
FROM transactions
GROUP BY is_fraud
ORDER BY is_fraud;

-- Q2: How does the fraud rate differ by channel?
SELECT
    channel,
    COUNT(*) AS transactions,
    SUM(is_fraud) AS frauds,
    ROUND(100.0 * AVG(is_fraud), 2) AS fraud_rate_pct
FROM transactions
GROUP BY channel
ORDER BY fraud_rate_pct DESC;

-- Q3: Which merchant categories have the highest fraud rate (at least 50 transactions)?
SELECT
    merchant_category,
    COUNT(*) AS transactions,
    SUM(is_fraud) AS frauds,
    ROUND(100.0 * AVG(is_fraud), 2) AS fraud_rate_pct
FROM transactions
GROUP BY merchant_category
HAVING COUNT(*) >= 50
ORDER BY fraud_rate_pct DESC;

-- Q4: How do amounts compare between legitimate and fraudulent transactions?
SELECT
    is_fraud,
    COUNT(*) AS transactions,
    ROUND(AVG(amount), 2) AS avg_amount,
    ROUND(MIN(amount), 2) AS min_amount,
    ROUND(MAX(amount), 2) AS max_amount,
    ROUND(AVG(distance_from_home_km), 1) AS avg_distance_km
FROM transactions
GROUP BY is_fraud
ORDER BY is_fraud;

-- Q5: Is the fraud rate higher at night and for foreign transactions?
SELECT
    CASE WHEN CAST(strftime('%H', timestamp) AS INTEGER) IN (0, 1, 2, 3, 4, 5, 22, 23)
         THEN 'night' ELSE 'day' END AS time_of_day,
    CASE WHEN is_foreign = 1 THEN 'foreign' ELSE 'domestic' END AS location,
    COUNT(*) AS transactions,
    SUM(is_fraud) AS frauds,
    ROUND(100.0 * AVG(is_fraud), 2) AS fraud_rate_pct
FROM transactions
GROUP BY time_of_day, location
ORDER BY fraud_rate_pct DESC;

-- Q6: How does the number of frauds and the fraud rate develop month by month?
SELECT
    strftime('%Y-%m', timestamp) AS month,
    COUNT(*) AS transactions,
    SUM(is_fraud) AS frauds,
    ROUND(100.0 * AVG(is_fraud), 2) AS fraud_rate_pct
FROM transactions
GROUP BY month
ORDER BY month;

-- Q7: Which customers have the most fraudulent transactions (customer-level aggregate)?
SELECT
    c.customer_id,
    c.country,
    c.credit_limit,
    COUNT(*) AS total_transactions,
    SUM(t.is_fraud) AS frauds,
    ROUND(SUM(CASE WHEN t.is_fraud = 1 THEN t.amount ELSE 0 END), 2) AS fraud_amount
FROM transactions AS t
JOIN customers AS c ON c.customer_id = t.customer_id
GROUP BY c.customer_id, c.country, c.credit_limit
HAVING SUM(t.is_fraud) > 0
ORDER BY frauds DESC, fraud_amount DESC, c.customer_id
LIMIT 10;

-- Q8: How is the fraud rate distributed across customer home countries, ranked from highest to lowest?
WITH by_country AS (
    SELECT
        c.country,
        COUNT(*) AS transactions,
        SUM(t.is_fraud) AS frauds,
        100.0 * AVG(t.is_fraud) AS fraud_rate_pct
    FROM transactions AS t
    JOIN customers AS c ON c.customer_id = t.customer_id
    GROUP BY c.country
)
SELECT
    RANK() OVER (ORDER BY fraud_rate_pct DESC) AS rank_by_rate,
    country,
    transactions,
    frauds,
    ROUND(fraud_rate_pct, 2) AS fraud_rate_pct
FROM by_country
ORDER BY rank_by_rate;
