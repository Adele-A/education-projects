-- Delivery Time Analysis: analytical queries (SQLite dialect)
-- Tables: orders (one row per order) and couriers (one row per courier).
-- The data is SYNTHETIC, so results reflect assumptions of the generator.
-- Run with: python src/run_sql.py

-- Q1: How do the number of orders and the average delivery time differ by weather?
SELECT
    weather,
    COUNT(*) AS orders,
    ROUND(AVG(delivery_time_min), 2) AS avg_delivery_min,
    ROUND(AVG(distance_km), 2) AS avg_distance_km
FROM orders
GROUP BY weather
ORDER BY avg_delivery_min;

-- Q2: How does the average delivery time change with courier load (active orders)?
SELECT
    courier_active_orders,
    COUNT(*) AS orders,
    ROUND(AVG(delivery_time_min), 2) AS avg_delivery_min,
    ROUND(AVG(prep_time_min), 2) AS avg_prep_min
FROM orders
GROUP BY courier_active_orders
ORDER BY courier_active_orders;

-- Q3: Which vehicle type delivers fastest, and how does that compare per km?
SELECT
    c.vehicle_type,
    COUNT(*) AS orders,
    ROUND(AVG(o.delivery_time_min), 2) AS avg_delivery_min,
    ROUND(AVG(o.distance_km), 2) AS avg_distance_km,
    ROUND(AVG(o.delivery_time_min / o.distance_km), 2) AS avg_min_per_km
FROM orders AS o
JOIN couriers AS c ON c.courier_id = o.courier_id
GROUP BY c.vehicle_type
ORDER BY avg_delivery_min;

-- Q4: What is the average delivery time in each distance band?
SELECT
    CASE
        WHEN distance_km < 2 THEN '1) 0-2 km'
        WHEN distance_km < 4 THEN '2) 2-4 km'
        WHEN distance_km < 6 THEN '3) 4-6 km'
        WHEN distance_km < 8 THEN '4) 6-8 km'
        ELSE '5) 8+ km'
    END AS distance_band,
    COUNT(*) AS orders,
    ROUND(AVG(delivery_time_min), 2) AS avg_delivery_min,
    ROUND(MAX(delivery_time_min), 1) AS max_delivery_min
FROM orders
GROUP BY distance_band
ORDER BY distance_band;

-- Q5: Does the weather effect differ between peak and off-peak hours?
SELECT
    weather,
    is_peak_hour,
    COUNT(*) AS orders,
    ROUND(AVG(delivery_time_min), 2) AS avg_delivery_min
FROM orders
GROUP BY weather, is_peak_hour
ORDER BY weather, is_peak_hour;

-- Q6: What share of orders take longer than 45 minutes, by weather?
SELECT
    weather,
    COUNT(*) AS orders,
    SUM(CASE WHEN delivery_time_min > 45 THEN 1 ELSE 0 END) AS slow_orders,
    ROUND(100.0 * SUM(CASE WHEN delivery_time_min > 45 THEN 1 ELSE 0 END) / COUNT(*), 2) AS slow_share_pct
FROM orders
GROUP BY weather
ORDER BY slow_share_pct DESC;

-- Q7: How does courier experience relate to delivery time and distance?
SELECT
    CASE
        WHEN c.experience_months < 12 THEN '1) under 1 year'
        WHEN c.experience_months < 36 THEN '2) 1-3 years'
        ELSE '3) 3+ years'
    END AS experience_band,
    COUNT(*) AS orders,
    ROUND(AVG(o.delivery_time_min), 2) AS avg_delivery_min,
    ROUND(AVG(o.distance_km), 2) AS avg_distance_km,
    ROUND(AVG(c.avg_rating), 2) AS avg_rating
FROM orders AS o
JOIN couriers AS c ON c.courier_id = o.courier_id
GROUP BY experience_band
ORDER BY experience_band;

-- Q8: Which 10 couriers with at least 20 orders have the highest average delivery time?
WITH courier_stats AS (
    SELECT
        courier_id,
        COUNT(*) AS orders,
        AVG(delivery_time_min) AS avg_delivery_min,
        AVG(distance_km) AS avg_distance_km
    FROM orders
    GROUP BY courier_id
    HAVING COUNT(*) >= 20
)
SELECT
    s.courier_id,
    c.vehicle_type,
    c.experience_months,
    s.orders,
    ROUND(s.avg_delivery_min, 2) AS avg_delivery_min,
    ROUND(s.avg_distance_km, 2) AS avg_distance_km
FROM courier_stats AS s
JOIN couriers AS c ON c.courier_id = s.courier_id
ORDER BY s.avg_delivery_min DESC
LIMIT 10;
