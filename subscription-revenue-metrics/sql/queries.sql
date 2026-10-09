-- Subscription revenue metrics in SQLite.
-- Tables (loaded by src/run_sql.py): customers, subscription_events.
-- Dates are stored as 'YYYY-MM-DD' text, so strftime() and julianday() work.
-- All data is SYNTHETIC, produced by src/data_generator.py.
-- Each query is preceded by a comment line starting with 'Q<n>:'.
-- Statements are separated by semicolons.

-- Q1: What is the monthly MRR bridge (opening + new + expansion + contraction + churned = closing)?
WITH ev AS (
    SELECT strftime('%Y-%m', event_date) AS month, event_type, mrr_change
    FROM subscription_events
),
m AS (
    SELECT month,
           SUM(CASE WHEN event_type = 'new'         THEN mrr_change ELSE 0 END) AS new_mrr,
           SUM(CASE WHEN event_type = 'expansion'   THEN mrr_change ELSE 0 END) AS expansion_mrr,
           SUM(CASE WHEN event_type = 'contraction' THEN mrr_change ELSE 0 END) AS contraction_mrr,
           SUM(CASE WHEN event_type = 'churn'       THEN mrr_change ELSE 0 END) AS churned_mrr,
           SUM(mrr_change) AS net_change
    FROM ev
    GROUP BY month
)
SELECT month,
       ROUND(SUM(net_change) OVER (ORDER BY month) - net_change, 2) AS opening_mrr,
       ROUND(new_mrr, 2)         AS new_mrr,
       ROUND(expansion_mrr, 2)   AS expansion_mrr,
       ROUND(contraction_mrr, 2) AS contraction_mrr,
       ROUND(churned_mrr, 2)     AS churned_mrr,
       ROUND(SUM(net_change) OVER (ORDER BY month), 2) AS closing_mrr
FROM m
ORDER BY month;

-- Q2: What are the monthly customer churn rate, revenue churn rate, GRR and NRR?
-- Rates use the opening position of the month as the base (NULL when the base is 0).
WITH ev AS (
    SELECT strftime('%Y-%m', event_date) AS month, event_type, mrr_change
    FROM subscription_events
),
m AS (
    SELECT month,
           SUM(CASE WHEN event_type = 'expansion'   THEN mrr_change ELSE 0 END) AS expansion_mrr,
           SUM(CASE WHEN event_type = 'contraction' THEN mrr_change ELSE 0 END) AS contraction_mrr,
           SUM(CASE WHEN event_type = 'churn'       THEN mrr_change ELSE 0 END) AS churned_mrr,
           SUM(CASE WHEN event_type = 'new'   THEN 1 ELSE 0 END) AS new_customers,
           SUM(CASE WHEN event_type = 'churn' THEN 1 ELSE 0 END) AS churned_customers,
           SUM(mrr_change) AS net_change
    FROM ev
    GROUP BY month
),
b AS (
    SELECT *,
           SUM(net_change) OVER (ORDER BY month) - net_change AS opening_mrr,
           SUM(new_customers - churned_customers) OVER (ORDER BY month)
               - (new_customers - churned_customers) AS active_start
    FROM m
)
SELECT month,
       active_start,
       ROUND(opening_mrr, 2) AS opening_mrr,
       ROUND(1.0 * churned_customers / NULLIF(active_start, 0), 4) AS customer_churn_rate,
       ROUND(-churned_mrr / NULLIF(opening_mrr, 0), 4) AS revenue_churn_rate,
       ROUND((opening_mrr + contraction_mrr + churned_mrr) / NULLIF(opening_mrr, 0), 4) AS grr,
       ROUND((opening_mrr + expansion_mrr + contraction_mrr + churned_mrr)
             / NULLIF(opening_mrr, 0), 4) AS nrr
FROM b
ORDER BY month;

-- Q3: How many customers churned in each acquisition channel, and how much MRR did they take with them?
SELECT c.acquisition_channel,
       COUNT(*) AS customers,
       COUNT(ce.event_id) AS churned_customers,
       ROUND(100.0 * COUNT(ce.event_id) / COUNT(*), 2) AS churned_share_pct,
       ROUND(-COALESCE(SUM(ce.mrr_change), 0), 2) AS churned_mrr
FROM customers AS c
LEFT JOIN subscription_events AS ce
       ON ce.customer_id = c.customer_id AND ce.event_type = 'churn'
GROUP BY c.acquisition_channel
ORDER BY churned_share_pct DESC;

-- Q4: How do monthly GRR and NRR (opening-MRR weighted) differ by initial plan?
-- A month x segment grid keeps months without events for a segment in the base.
WITH months AS (
    SELECT DISTINCT strftime('%Y-%m', event_date) AS month FROM subscription_events
),
segs AS (
    SELECT DISTINCT initial_plan AS segment FROM customers
),
mv AS (
    SELECT c.initial_plan AS segment,
           strftime('%Y-%m', e.event_date) AS month,
           SUM(CASE WHEN e.event_type = 'expansion'   THEN e.mrr_change ELSE 0 END) AS expansion_mrr,
           SUM(CASE WHEN e.event_type = 'contraction' THEN e.mrr_change ELSE 0 END) AS contraction_mrr,
           SUM(CASE WHEN e.event_type = 'churn'       THEN e.mrr_change ELSE 0 END) AS churned_mrr,
           SUM(e.mrr_change) AS net_change
    FROM subscription_events AS e
    JOIN customers AS c ON c.customer_id = e.customer_id
    GROUP BY c.initial_plan, strftime('%Y-%m', e.event_date)
),
grid AS (
    SELECT s.segment, m.month,
           COALESCE(mv.expansion_mrr, 0)   AS expansion_mrr,
           COALESCE(mv.contraction_mrr, 0) AS contraction_mrr,
           COALESCE(mv.churned_mrr, 0)     AS churned_mrr,
           COALESCE(mv.net_change, 0)      AS net_change
    FROM segs AS s
    CROSS JOIN months AS m
    LEFT JOIN mv ON mv.segment = s.segment AND mv.month = m.month
),
b AS (
    SELECT *,
           SUM(net_change) OVER (PARTITION BY segment ORDER BY month) - net_change AS opening_mrr
    FROM grid
)
SELECT segment AS initial_plan,
       COUNT(*) AS months_used,
       ROUND(1 + (SUM(contraction_mrr) + SUM(churned_mrr)) / SUM(opening_mrr), 4) AS grr,
       ROUND(1 + (SUM(expansion_mrr) + SUM(contraction_mrr) + SUM(churned_mrr))
                 / SUM(opening_mrr), 4) AS nrr
FROM b
WHERE opening_mrr > 0
GROUP BY segment
ORDER BY segment;

-- Q5: How does cohort MRR (by signup month) develop 0, 3, 6 and 12 months after signup?
-- Offsets that a cohort has not reached yet are NULL.
WITH ev AS (
    SELECT strftime('%Y-%m', c.signup_date) AS cohort,
           CAST(strftime('%Y', c.signup_date) AS INTEGER) * 12
               + CAST(strftime('%m', c.signup_date) AS INTEGER) AS cohort_idx,
           CAST(strftime('%Y', e.event_date) AS INTEGER) * 12
               + CAST(strftime('%m', e.event_date) AS INTEGER) AS event_idx,
           e.mrr_change
    FROM subscription_events AS e
    JOIN customers AS c ON c.customer_id = e.customer_id
),
c AS (
    SELECT cohort,
           SUM(CASE WHEN event_idx - cohort_idx <= 0 THEN mrr_change END) AS mrr_m0,
           CASE WHEN MIN(cohort_idx) + 3 <= (SELECT MAX(event_idx) FROM ev)
                THEN SUM(CASE WHEN event_idx - cohort_idx <= 3 THEN mrr_change END) END AS mrr_m3,
           CASE WHEN MIN(cohort_idx) + 6 <= (SELECT MAX(event_idx) FROM ev)
                THEN SUM(CASE WHEN event_idx - cohort_idx <= 6 THEN mrr_change END) END AS mrr_m6,
           CASE WHEN MIN(cohort_idx) + 12 <= (SELECT MAX(event_idx) FROM ev)
                THEN SUM(CASE WHEN event_idx - cohort_idx <= 12 THEN mrr_change END) END AS mrr_m12
    FROM ev
    GROUP BY cohort
)
SELECT cohort,
       ROUND(mrr_m0, 2) AS mrr_m0,
       ROUND(100.0 * mrr_m3 / mrr_m0, 1)  AS retention_m3_pct,
       ROUND(100.0 * mrr_m6 / mrr_m0, 1)  AS retention_m6_pct,
       ROUND(100.0 * mrr_m12 / mrr_m0, 1) AS retention_m12_pct
FROM c
ORDER BY cohort;

-- Q6: Which 10 customers have the highest current MRR?
SELECT c.customer_id,
       c.initial_plan,
       c.acquisition_channel,
       c.company_size,
       c.signup_date,
       ROUND(SUM(e.mrr_change), 2) AS current_mrr
FROM customers AS c
JOIN subscription_events AS e ON e.customer_id = c.customer_id
GROUP BY c.customer_id, c.initial_plan, c.acquisition_channel, c.company_size, c.signup_date
HAVING SUM(e.mrr_change) > 0
ORDER BY current_mrr DESC, c.customer_id
LIMIT 10;

-- Q7: How long do churned customers stay, on average, by initial plan?
-- Lifetime is the time from signup to the churn event, in months of 30.44 days.
SELECT c.initial_plan,
       COUNT(*) AS customers,
       COUNT(ce.event_id) AS churned_customers,
       ROUND(AVG((julianday(ce.event_date) - julianday(c.signup_date)) / 30.44), 2)
           AS avg_months_to_churn
FROM customers AS c
LEFT JOIN subscription_events AS ce
       ON ce.customer_id = c.customer_id AND ce.event_type = 'churn'
GROUP BY c.initial_plan
ORDER BY c.initial_plan;
