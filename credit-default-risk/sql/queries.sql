-- Portfolio queries for the SYNTHETIC credit default dataset (SQLite dialect).
-- Tables: customers (borrower profile) and loans (applications with the target "default").
-- "default" is a reserved word in SQL, so it is always written in double quotes.
-- Each query starts with a comment line '-- Q<n>:' that describes the question.
-- Run all queries with:  python src/run_sql.py

-- Q1: What is the size, default rate and average interest rate of the whole portfolio?
SELECT
    COUNT(*)                          AS loans,
    SUM(l."default")                  AS defaults,
    ROUND(AVG(l."default"), 4)        AS default_rate,
    ROUND(SUM(l.loan_amount), 0)      AS total_amount,
    ROUND(AVG(l.interest_rate), 2)    AS avg_interest_rate
FROM loans AS l;

-- Q2: Which loan purposes have the highest default rate?
SELECT
    l.loan_purpose,
    COUNT(*)                          AS loans,
    SUM(l."default")                  AS defaults,
    ROUND(AVG(l."default"), 4)        AS default_rate
FROM loans AS l
GROUP BY l.loan_purpose
ORDER BY default_rate DESC;

-- Q3: How does the default rate change across credit score bands?
WITH banded AS (
    SELECT
        l."default" AS is_default,
        l.interest_rate,
        CASE
            WHEN c.credit_score < 580 THEN '300-579'
            WHEN c.credit_score < 670 THEN '580-669'
            WHEN c.credit_score < 740 THEN '670-739'
            WHEN c.credit_score < 800 THEN '740-799'
            ELSE '800-850'
        END AS score_band
    FROM loans AS l
    JOIN customers AS c ON c.customer_id = l.customer_id
)
SELECT
    score_band,
    COUNT(*)                          AS loans,
    ROUND(AVG(is_default), 4)         AS default_rate,
    ROUND(AVG(interest_rate), 2)      AS avg_interest_rate
FROM banded
GROUP BY score_band
ORDER BY score_band;

-- Q4: Which region and home ownership combinations are the riskiest (at least 50 loans)?
SELECT
    c.region,
    c.home_ownership,
    COUNT(*)                          AS loans,
    ROUND(AVG(l."default"), 4)        AS default_rate
FROM loans AS l
JOIN customers AS c ON c.customer_id = l.customer_id
GROUP BY c.region, c.home_ownership
HAVING COUNT(*) >= 50
ORDER BY default_rate DESC;

-- Q5: How strongly do past delinquencies (grouped, 3 means 3 or more) relate to default?
SELECT
    CASE
        WHEN l.num_delinquencies_2y >= 3 THEN '3+'
        ELSE CAST(l.num_delinquencies_2y AS TEXT)
    END                               AS delinquencies_2y,
    COUNT(*)                          AS loans,
    SUM(l."default")                  AS defaults,
    ROUND(AVG(l."default"), 4)        AS default_rate
FROM loans AS l
GROUP BY delinquencies_2y
ORDER BY delinquencies_2y;

-- Q6: Which repeat borrowers (2 or more loans) have the most defaults?
SELECT
    l.customer_id,
    c.credit_score,
    COUNT(*)                          AS loans,
    SUM(l."default")                  AS defaults,
    ROUND(SUM(l.loan_amount), 2)      AS total_borrowed
FROM loans AS l
JOIN customers AS c ON c.customer_id = l.customer_id
GROUP BY l.customer_id, c.credit_score
HAVING COUNT(*) >= 2
ORDER BY defaults DESC, loans DESC, l.customer_id
LIMIT 10;

-- Q7: Is the default rate stable over time (by application quarter)?
WITH quarterly AS (
    SELECT
        strftime('%Y', l.application_date) || '-Q'
            || ((CAST(strftime('%m', l.application_date) AS INTEGER) + 2) / 3) AS quarter,
        l."default" AS is_default,
        l.loan_amount
    FROM loans AS l
)
SELECT
    quarter,
    COUNT(*)                          AS loans,
    ROUND(AVG(is_default), 4)         AS default_rate,
    ROUND(AVG(loan_amount), 0)        AS avg_loan_amount
FROM quarterly
GROUP BY quarter
ORDER BY quarter;

-- Q8: Which loan purposes account for the largest share of the defaulted amount?
SELECT
    l.loan_purpose,
    COUNT(*)                          AS defaulted_loans,
    ROUND(SUM(l.loan_amount), 0)      AS defaulted_amount,
    ROUND(100.0 * SUM(l.loan_amount) / SUM(SUM(l.loan_amount)) OVER (), 1)
                                      AS share_of_defaulted_amount_pct
FROM loans AS l
WHERE l."default" = 1
GROUP BY l.loan_purpose
ORDER BY defaulted_amount DESC;
