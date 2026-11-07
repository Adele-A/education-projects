-- Analytical SQL queries for the synthetic employee attrition data (SQLite dialect).
-- Tables: employees (one row per employee) and satisfaction_surveys (respondents only).
-- attrition is 1 = left, 0 = stayed, so AVG(attrition) is the attrition rate.
-- Run with: python src/run_sql.py

-- Q1: What is the overall headcount, number of leavers and attrition rate?
SELECT
    COUNT(*)                   AS employees,
    SUM(attrition)             AS leavers,
    ROUND(AVG(attrition), 3)   AS attrition_rate
FROM employees;

-- Q2: Which departments have the highest attrition rate?
SELECT
    department,
    COUNT(*)                   AS employees,
    SUM(attrition)             AS leavers,
    ROUND(AVG(attrition), 3)   AS attrition_rate
FROM employees
GROUP BY department
ORDER BY attrition_rate DESC;

-- Q3: How does attrition differ between employees with and without overtime?
SELECT
    overtime,
    COUNT(*)                          AS employees,
    ROUND(AVG(attrition), 3)          AS attrition_rate,
    ROUND(AVG(monthly_income), 0)     AS avg_monthly_income
FROM employees
GROUP BY overtime
ORDER BY overtime DESC;

-- Q4: How do attrition and average income change with job level?
SELECT
    job_level,
    COUNT(*)                          AS employees,
    ROUND(AVG(monthly_income), 0)     AS avg_monthly_income,
    ROUND(AVG(attrition), 3)          AS attrition_rate
FROM employees
GROUP BY job_level
ORDER BY job_level;

-- Q5: How does attrition change with tenure band (years at the company)?
SELECT
    CASE
        WHEN years_at_company < 2  THEN '0-2 years'
        WHEN years_at_company < 5  THEN '2-5 years'
        WHEN years_at_company < 10 THEN '5-10 years'
        ELSE '10+ years'
    END                              AS tenure_band,
    COUNT(*)                         AS employees,
    ROUND(AVG(attrition), 3)         AS attrition_rate
FROM employees
GROUP BY tenure_band
ORDER BY MIN(years_at_company);

-- Q6: How does attrition vary with job satisfaction, including employees who did not answer the survey?
SELECT
    COALESCE(CAST(s.job_satisfaction AS TEXT), 'No response') AS job_satisfaction,
    COUNT(*)                                                  AS employees,
    ROUND(AVG(e.attrition), 3)                                AS attrition_rate
FROM employees AS e
LEFT JOIN satisfaction_surveys AS s
    ON s.employee_id = e.employee_id
GROUP BY COALESCE(CAST(s.job_satisfaction AS TEXT), 'No response')
ORDER BY job_satisfaction;

-- Q7: Which department and overtime combinations (at least 30 employees) have the highest attrition?
SELECT
    department,
    overtime,
    COUNT(*)                   AS employees,
    ROUND(AVG(attrition), 3)   AS attrition_rate
FROM employees
GROUP BY department, overtime
HAVING COUNT(*) >= 30
ORDER BY attrition_rate DESC
LIMIT 8;

-- Q8: How does attrition differ across income quartiles (1 = lowest income, 4 = highest)?
WITH ranked AS (
    SELECT
        monthly_income,
        attrition,
        NTILE(4) OVER (ORDER BY monthly_income) AS income_quartile
    FROM employees
)
SELECT
    income_quartile,
    COUNT(*)                   AS employees,
    MIN(monthly_income)        AS min_income,
    MAX(monthly_income)        AS max_income,
    ROUND(AVG(attrition), 3)   AS attrition_rate
FROM ranked
GROUP BY income_quartile
ORDER BY income_quartile;
