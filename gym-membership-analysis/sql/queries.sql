-- SQL analysis of the SYNTHETIC gym membership dataset (SQLite dialect).
-- Tables: members (one row per member), upgrades (one row per upgrade event).
-- Dates are stored as ISO text (YYYY-MM-DD), so julianday() can be used on them.
-- Snapshot date of the dataset: 2026-10-31.
-- Run with: python src/run_sql.py

-- Q1: What is the churn rate by initial plan and contract type?
SELECT
    initial_plan,
    contract_type,
    COUNT(*)                         AS members,
    SUM(is_churned)                  AS churned_members,
    ROUND(100.0 * AVG(is_churned), 1) AS churn_rate_pct
FROM members
GROUP BY initial_plan, contract_type
ORDER BY initial_plan, contract_type;

-- Q2: How do churn rate and days since the last visit change across visit-frequency bands?
WITH banded AS (
    SELECT
        CASE
            WHEN avg_visits_per_week < 1 THEN '0-1'
            WHEN avg_visits_per_week < 2 THEN '1-2'
            WHEN avg_visits_per_week < 3 THEN '2-3'
            WHEN avg_visits_per_week < 4 THEN '3-4'
            ELSE '4-7'
        END AS visit_band,
        is_churned,
        days_since_last_visit
    FROM members
)
SELECT
    visit_band,
    COUNT(*)                          AS members,
    ROUND(100.0 * AVG(is_churned), 1) AS churn_rate_pct,
    ROUND(AVG(days_since_last_visit), 1) AS avg_days_since_last_visit
FROM banded
GROUP BY visit_band
ORDER BY visit_band;

-- Q3: Upgrade funnel - how many eligible members (not Premium at sign-up, at least 90 days of membership) upgraded, by initial plan?
WITH eligible AS (
    SELECT
        m.member_id,
        m.initial_plan,
        CASE WHEN u.member_id IS NOT NULL THEN 1 ELSE 0 END AS upgraded
    FROM members AS m
    LEFT JOIN upgrades AS u ON u.member_id = m.member_id
    WHERE m.initial_plan <> 'Premium'
      AND julianday(COALESCE(m.churn_date, '2026-10-31')) - julianday(m.signup_date) >= 90
)
SELECT
    initial_plan,
    COUNT(*)                            AS eligible_members,
    SUM(upgraded)                       AS upgraded_members,
    ROUND(100.0 * AVG(upgraded), 1)     AS upgrade_rate_pct
FROM eligible
GROUP BY initial_plan
ORDER BY initial_plan;

-- Q4: Which upgrade paths are most common and how much monthly revenue do they add?
SELECT
    from_plan,
    to_plan,
    COUNT(*)                                          AS upgrades,
    ROUND(AVG(monthly_fee_after - monthly_fee_before), 2) AS avg_fee_increase,
    ROUND(SUM(monthly_fee_after - monthly_fee_before), 2) AS added_monthly_revenue
FROM upgrades
GROUP BY from_plan, to_plan
ORDER BY upgrades DESC;

-- Q5: How long after sign-up do members upgrade, and how often do upgraders visit, by upgrade path?
SELECT
    u.from_plan,
    u.to_plan,
    COUNT(*)                                                         AS upgrades,
    ROUND(AVG(julianday(u.upgrade_date) - julianday(m.signup_date)), 0) AS avg_days_to_upgrade,
    ROUND(AVG(m.avg_visits_per_week), 2)                             AS avg_visits_per_week
FROM upgrades AS u
INNER JOIN members AS m ON m.member_id = u.member_id
GROUP BY u.from_plan, u.to_plan
ORDER BY u.from_plan, u.to_plan;

-- Q6: Which active members look at risk by a simple rule (under 1 visit per week, no visit for 21+ days, month-to-month contract)?
SELECT
    member_id,
    initial_plan,
    current_plan,
    avg_visits_per_week,
    days_since_last_visit,
    distance_to_gym_km
FROM members
WHERE is_churned = 0
  AND avg_visits_per_week < 1
  AND days_since_last_visit >= 21
  AND contract_type = 'Month-to-month'
ORDER BY days_since_last_visit DESC, avg_visits_per_week ASC, member_id
LIMIT 20;

-- Q7: How do cities rank by churn rate (window function)?
SELECT
    city,
    COUNT(*)                          AS members,
    ROUND(100.0 * AVG(is_churned), 1) AS churn_rate_pct,
    RANK() OVER (ORDER BY AVG(is_churned) DESC) AS churn_rank
FROM members
GROUP BY city
ORDER BY churn_rank;

-- Q8: How do churn rate and average visits differ by sign-up year cohort?
SELECT
    STRFTIME('%Y', signup_date)       AS signup_year,
    COUNT(*)                          AS members,
    ROUND(100.0 * AVG(is_churned), 1) AS churn_rate_pct,
    ROUND(AVG(avg_visits_per_week), 2) AS avg_visits_per_week
FROM members
GROUP BY signup_year
ORDER BY signup_year;
