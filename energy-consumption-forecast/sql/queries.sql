-- SQL analysis for the SYNTHETIC household energy data (SQLite dialect).
-- Tables: households, weather_hourly, energy_hourly (loaded from data/*.csv by src/run_sql.py).
-- Timestamps are stored as text 'YYYY-MM-DD HH:MM:SS', so SQLite date functions work on them.
-- Note for the parser in src/run_sql.py: statements are separated by a semicolon,
-- so comments must not contain that character. Each query starts with a '-- Q<n>:' line.

-- Q1: Which five hours of the day have the highest mean consumption per household-hour?
SELECT
    CAST(strftime('%H', timestamp) AS INTEGER) AS hour,
    ROUND(AVG(kwh), 3)                         AS mean_kwh
FROM energy_hourly
GROUP BY hour
ORDER BY mean_kwh DESC, hour
LIMIT 5;

-- Q2: Which ten days had the highest total consumption of all households, and what was the mean temperature on those days?
SELECT
    date(e.timestamp)              AS day,
    ROUND(SUM(e.kwh), 2)           AS total_kwh,
    ROUND(AVG(w.temperature_c), 1) AS mean_temp_c
FROM energy_hourly AS e
JOIN weather_hourly AS w ON w.timestamp = e.timestamp
GROUP BY day
ORDER BY total_kwh DESC, day
LIMIT 10;

-- Q3: How do households rank by total consumption, and how much is that per square metre of floor area?
SELECT
    RANK() OVER (ORDER BY SUM(e.kwh) DESC)       AS rank_total,
    h.household_id,
    h.dwelling_type,
    h.n_residents,
    h.floor_area_m2,
    h.has_electric_heating,
    ROUND(SUM(e.kwh), 1)                         AS total_kwh,
    ROUND(SUM(e.kwh) / h.floor_area_m2, 2)       AS kwh_per_m2
FROM energy_hourly AS e
JOIN households AS h ON h.household_id = e.household_id
GROUP BY h.household_id
ORDER BY rank_total, h.household_id;

-- Q4: How does mean hourly consumption of each household differ between weekdays and weekends?
WITH labelled AS (
    SELECT
        household_id,
        kwh,
        CASE WHEN strftime('%w', timestamp) IN ('0', '6')
             THEN 'weekend' ELSE 'weekday' END AS day_type
    FROM energy_hourly
),
by_type AS (
    SELECT
        household_id,
        AVG(CASE WHEN day_type = 'weekday' THEN kwh END) AS weekday_mean,
        AVG(CASE WHEN day_type = 'weekend' THEN kwh END) AS weekend_mean
    FROM labelled
    GROUP BY household_id
)
SELECT
    household_id,
    ROUND(weekday_mean, 3)                AS weekday_mean_kwh,
    ROUND(weekend_mean, 3)                AS weekend_mean_kwh,
    ROUND(weekend_mean / weekday_mean, 3) AS weekend_to_weekday_ratio
FROM by_type
ORDER BY household_id;

-- Q5: What is the mean consumption in each outdoor temperature band for households with and without electric heating?
SELECT
    CASE WHEN w.temperature_c < 0 THEN '1: below 0 C'
         WHEN w.temperature_c < 5 THEN '2: 0 to 5 C'
         ELSE '3: 5 C and above' END                AS temp_band,
    CASE h.has_electric_heating WHEN 1 THEN 'electric heating'
         ELSE 'no electric heating' END             AS heating,
    COUNT(*)                                        AS household_hours,
    ROUND(AVG(e.kwh), 3)                            AS mean_kwh
FROM energy_hourly AS e
JOIN households AS h    ON h.household_id = e.household_id
JOIN weather_hourly AS w ON w.timestamp = e.timestamp
GROUP BY temp_band, heating
ORDER BY temp_band, heating;

-- Q6: What share of each household's consumption falls into the evening peak (18:00 to 21:59)?
SELECT
    household_id,
    ROUND(SUM(CASE WHEN CAST(strftime('%H', timestamp) AS INTEGER) BETWEEN 18 AND 21
                   THEN kwh ELSE 0 END), 1)                       AS evening_kwh,
    ROUND(SUM(kwh), 1)                                            AS total_kwh,
    ROUND(100.0 * SUM(CASE WHEN CAST(strftime('%H', timestamp) AS INTEGER) BETWEEN 18 AND 21
                           THEN kwh ELSE 0 END) / SUM(kwh), 1)    AS evening_share_pct
FROM energy_hourly
GROUP BY household_id
ORDER BY evening_share_pct DESC, household_id;

-- Q7: Which five single household-hours had the highest consumption, and what was the temperature then?
SELECT
    e.household_id,
    e.timestamp,
    e.kwh,
    w.temperature_c,
    h.has_electric_heating
FROM energy_hourly AS e
JOIN weather_hourly AS w ON w.timestamp = e.timestamp
JOIN households AS h     ON h.household_id = e.household_id
ORDER BY e.kwh DESC, e.household_id, e.timestamp
LIMIT 5;

-- Q8: Which five days had the largest increase in total consumption compared with the previous day?
WITH daily AS (
    SELECT date(timestamp) AS day, SUM(kwh) AS total_kwh
    FROM energy_hourly
    GROUP BY day
),
changes AS (
    SELECT
        day,
        total_kwh,
        total_kwh - LAG(total_kwh) OVER (ORDER BY day) AS change_kwh
    FROM daily
)
SELECT
    day,
    ROUND(total_kwh, 2)  AS total_kwh,
    ROUND(change_kwh, 2) AS change_vs_previous_day_kwh
FROM changes
WHERE change_kwh IS NOT NULL
ORDER BY change_kwh DESC, day
LIMIT 5;
