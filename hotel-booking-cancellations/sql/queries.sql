-- Analytical queries for the SYNTHETIC hotel bookings data (SQLite dialect).
-- Tables: bookings (one row per booking) and guests (one row per guest).
-- The Python runner loads data/guests.csv and data/bookings.csv into
-- an in-memory database with these table names.
-- Each query starts with a comment line "-- Q<n>:" that describes the question.
-- Statements are separated by a semicolon, so avoid that character inside comments.

-- Q1: What is the overall cancellation rate, and how many bookings does it cover?
SELECT
    COUNT(*)                         AS bookings,
    SUM(is_canceled)                 AS canceled,
    ROUND(AVG(is_canceled), 3)       AS cancel_rate
FROM bookings;

-- Q2: How does the cancellation rate differ by deposit type?
SELECT
    deposit_type,
    COUNT(*)                         AS bookings,
    ROUND(AVG(is_canceled), 3)       AS cancel_rate
FROM bookings
GROUP BY deposit_type
ORDER BY cancel_rate DESC;

-- Q3: Which market segments cancel the most, split by hotel type?
SELECT
    hotel_type,
    market_segment,
    COUNT(*)                         AS bookings,
    ROUND(AVG(is_canceled), 3)       AS cancel_rate
FROM bookings
GROUP BY hotel_type, market_segment
ORDER BY hotel_type, cancel_rate DESC;

-- Q4: How does the cancellation rate change with lead time bucket (days between booking and arrival)?
SELECT
    CASE
        WHEN lead_time_days <= 7   THEN '0-7'
        WHEN lead_time_days <= 30  THEN '8-30'
        WHEN lead_time_days <= 90  THEN '31-90'
        WHEN lead_time_days <= 180 THEN '91-180'
        ELSE '181-450'
    END                              AS lead_bucket,
    COUNT(*)                         AS bookings,
    ROUND(AVG(is_canceled), 3)       AS cancel_rate
FROM bookings
GROUP BY lead_bucket
ORDER BY MIN(lead_time_days);

-- Q5: Do guests with earlier cancellations cancel more often? (join with guest history)
SELECT
    CASE
        WHEN g.previous_cancellations = 0 THEN '0 previous'
        WHEN g.previous_cancellations = 1 THEN '1 previous'
        ELSE '2+ previous'
    END                              AS guest_history,
    COUNT(*)                         AS bookings,
    ROUND(AVG(b.is_canceled), 3)     AS cancel_rate
FROM bookings AS b
JOIN guests AS g ON g.guest_id = b.guest_id
GROUP BY guest_history
ORDER BY MIN(g.previous_cancellations);

-- Q6: How do booking volume and cancellation rate vary by arrival month?
SELECT
    arrival_month,
    COUNT(*)                         AS bookings,
    ROUND(AVG(is_canceled), 3)       AS cancel_rate
FROM bookings
GROUP BY arrival_month
ORDER BY arrival_month;

-- Q7: Which guest countries have the highest cancellation rate (at least 100 bookings)?
WITH by_country AS (
    SELECT
        g.country,
        COUNT(*)                     AS bookings,
        ROUND(AVG(b.is_canceled), 3) AS cancel_rate
    FROM bookings AS b
    JOIN guests AS g ON g.guest_id = b.guest_id
    GROUP BY g.country
)
SELECT country, bookings, cancel_rate
FROM by_country
WHERE bookings >= 100
ORDER BY cancel_rate DESC;

-- Q8: Within each hotel type, how do the deposit types rank by cancellation rate? (window function)
SELECT
    hotel_type,
    deposit_type,
    bookings,
    cancel_rate,
    RANK() OVER (PARTITION BY hotel_type ORDER BY cancel_rate DESC) AS rank_in_hotel
FROM (
    SELECT
        hotel_type,
        deposit_type,
        COUNT(*)                     AS bookings,
        ROUND(AVG(is_canceled), 3)   AS cancel_rate
    FROM bookings
    GROUP BY hotel_type, deposit_type
)
ORDER BY hotel_type, rank_in_hotel;
