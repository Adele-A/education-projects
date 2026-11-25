-- SQL analysis for the SYNTHETIC customer survey NPS dataset (SQLite dialect).
-- Tables: customers, survey_responses (loaded from the CSV files in data/).
-- NPS = 100 * (promoters with score 9-10 minus detractors with score 0-6) / responses.
-- Run with: python src/run_sql.py

-- Q1: What is the overall NPS and how are responses split across categories?
SELECT
    COUNT(*) AS responses,
    SUM(CASE WHEN nps_score >= 9 THEN 1 ELSE 0 END) AS promoters,
    SUM(CASE WHEN nps_score BETWEEN 7 AND 8 THEN 1 ELSE 0 END) AS passives,
    SUM(CASE WHEN nps_score <= 6 THEN 1 ELSE 0 END) AS detractors,
    ROUND(AVG(nps_score), 2) AS mean_score,
    ROUND(100.0 * (SUM(CASE WHEN nps_score >= 9 THEN 1 ELSE 0 END)
                 - SUM(CASE WHEN nps_score <= 6 THEN 1 ELSE 0 END)) / COUNT(*), 1) AS nps
FROM survey_responses;

-- Q2: What is the survey response rate by segment?
-- Every customer is counted once, a customer with no response row counts as not surveyed.
SELECT
    c.segment,
    COUNT(*) AS customers,
    COUNT(r.response_id) AS responses,
    ROUND(100.0 * COUNT(r.response_id) / COUNT(*), 1) AS response_rate_pct
FROM customers AS c
LEFT JOIN survey_responses AS r ON r.customer_id = c.customer_id
GROUP BY c.segment
ORDER BY response_rate_pct DESC;

-- Q3: How does NPS differ by customer segment, from best to worst?
SELECT
    c.segment,
    COUNT(*) AS responses,
    ROUND(AVG(r.nps_score), 2) AS mean_score,
    ROUND(100.0 * (SUM(CASE WHEN r.nps_score >= 9 THEN 1 ELSE 0 END)
                 - SUM(CASE WHEN r.nps_score <= 6 THEN 1 ELSE 0 END)) / COUNT(*), 1) AS nps
FROM survey_responses AS r
JOIN customers AS c ON c.customer_id = r.customer_id
GROUP BY c.segment
ORDER BY nps DESC;

-- Q4: How does NPS differ by plan and acquisition channel?
-- Both dimensions are stacked into one result with a UNION ALL.
SELECT 'plan' AS dimension, c.plan AS grp, COUNT(*) AS responses,
       ROUND(100.0 * (SUM(CASE WHEN r.nps_score >= 9 THEN 1 ELSE 0 END)
                    - SUM(CASE WHEN r.nps_score <= 6 THEN 1 ELSE 0 END)) / COUNT(*), 1) AS nps
FROM survey_responses AS r
JOIN customers AS c ON c.customer_id = r.customer_id
GROUP BY c.plan
UNION ALL
SELECT 'acquisition_channel' AS dimension, c.acquisition_channel AS grp, COUNT(*) AS responses,
       ROUND(100.0 * (SUM(CASE WHEN r.nps_score >= 9 THEN 1 ELSE 0 END)
                    - SUM(CASE WHEN r.nps_score <= 6 THEN 1 ELSE 0 END)) / COUNT(*), 1) AS nps
FROM survey_responses AS r
JOIN customers AS c ON c.customer_id = r.customer_id
GROUP BY c.acquisition_channel
ORDER BY dimension, nps DESC;

-- Q5: How does NPS change by survey month?
-- Monthly samples are small, so month-to-month swings can be noise.
SELECT
    strftime('%Y-%m', survey_date) AS month,
    COUNT(*) AS responses,
    ROUND(100.0 * (SUM(CASE WHEN nps_score >= 9 THEN 1 ELSE 0 END)
                 - SUM(CASE WHEN nps_score <= 6 THEN 1 ELSE 0 END)) / COUNT(*), 1) AS nps
FROM survey_responses
GROUP BY strftime('%Y-%m', survey_date)
ORDER BY month;

-- Q6: How does NPS change by tenure band?
SELECT
    CASE
        WHEN c.tenure_months <= 12 THEN '1-12'
        WHEN c.tenure_months <= 24 THEN '13-24'
        WHEN c.tenure_months <= 48 THEN '25-48'
        ELSE '49-72'
    END AS tenure_band,
    COUNT(*) AS responses,
    ROUND(AVG(r.nps_score), 2) AS mean_score,
    ROUND(100.0 * (SUM(CASE WHEN r.nps_score >= 9 THEN 1 ELSE 0 END)
                 - SUM(CASE WHEN r.nps_score <= 6 THEN 1 ELSE 0 END)) / COUNT(*), 1) AS nps
FROM survey_responses AS r
JOIN customers AS c ON c.customer_id = r.customer_id
GROUP BY tenure_band
ORDER BY MIN(c.tenure_months);

-- Q7: Which segment and plan combinations have the lowest NPS (at least 30 responses)?
SELECT
    c.segment,
    c.plan,
    COUNT(*) AS responses,
    ROUND(100.0 * (SUM(CASE WHEN r.nps_score >= 9 THEN 1 ELSE 0 END)
                 - SUM(CASE WHEN r.nps_score <= 6 THEN 1 ELSE 0 END)) / COUNT(*), 1) AS nps
FROM survey_responses AS r
JOIN customers AS c ON c.customer_id = r.customer_id
GROUP BY c.segment, c.plan
HAVING COUNT(*) >= 30
ORDER BY nps ASC
LIMIT 5;

-- Q8: Which share of comments in each NPS category mention selected keywords?
-- Only responses with a comment are used. LIKE returns 1 or 0, so AVG gives the share.
SELECT
    nps_category,
    COUNT(*) AS comments,
    ROUND(100.0 * AVG(comment LIKE '%support%'), 1) AS support_pct,
    ROUND(100.0 * AVG(comment LIKE '%expensive%'), 1) AS expensive_pct,
    ROUND(100.0 * AVG(comment LIKE '%outages%'), 1) AS outages_pct,
    ROUND(100.0 * AVG(comment LIKE '%easy to use%'), 1) AS easy_to_use_pct
FROM survey_responses
WHERE comment IS NOT NULL
GROUP BY nps_category
ORDER BY CASE nps_category WHEN 'Detractor' THEN 1 WHEN 'Passive' THEN 2 ELSE 3 END;
