-- SQL analysis of the SYNTHETIC air quality data (SQLite dialect).
--
-- Tables (loaded from the CSV files by src/run_sql.py):
--   stations(station_id, station_name, station_type, latitude, longitude, elevation_m)
--   daily_measurements(date, station_id, pm25, pm10, no2, o3, temperature_c,
--                      humidity_pct, wind_speed_ms, precipitation_mm, pressure_hpa)
-- The date column is text in YYYY-MM-DD format. NULL means a missing value,
-- and AVG/COUNT ignore NULLs.
-- Each query starts with a comment line "-- Q<n>:" and ends with a semicolon.

-- Q1: What is the monthly mean PM2.5 and PM10 across all stations (city level)?
SELECT strftime('%Y-%m', date) AS year_month,
       ROUND(AVG(pm25), 1)     AS mean_pm25,
       ROUND(AVG(pm10), 1)     AS mean_pm10,
       COUNT(pm25)             AS n_pm25_obs
FROM daily_measurements
GROUP BY year_month
ORDER BY year_month;

-- Q2: How do the stations rank by overall mean PM2.5 (1 = most polluted)?
SELECT RANK() OVER (ORDER BY AVG(m.pm25) DESC) AS pollution_rank,
       s.station_id,
       s.station_name,
       s.station_type,
       ROUND(AVG(m.pm25), 1) AS mean_pm25,
       ROUND(AVG(m.no2), 1)  AS mean_no2
FROM daily_measurements AS m
JOIN stations AS s ON s.station_id = m.station_id
GROUP BY s.station_id, s.station_name, s.station_type
ORDER BY pollution_rank;

-- Q3: How many days per station and year exceeded a daily PM2.5 threshold of 25 ug/m3 (illustrative threshold)?
SELECT m.station_id,
       CAST(substr(m.date, 1, 4) AS INTEGER) AS year,
       SUM(CASE WHEN m.pm25 > 25 THEN 1 ELSE 0 END) AS days_above_25,
       COUNT(m.pm25) AS days_with_data,
       ROUND(100.0 * SUM(CASE WHEN m.pm25 > 25 THEN 1 ELSE 0 END) / COUNT(m.pm25), 1) AS pct_days_above
FROM daily_measurements AS m
GROUP BY m.station_id, year
ORDER BY m.station_id, year;

-- Q4: What is the seasonal pattern, i.e. mean pollutants and temperature by calendar month over all years?
SELECT CAST(strftime('%m', date) AS INTEGER) AS month,
       ROUND(AVG(pm25), 1)          AS mean_pm25,
       ROUND(AVG(no2), 1)           AS mean_no2,
       ROUND(AVG(o3), 1)            AS mean_o3,
       ROUND(AVG(temperature_c), 1) AS mean_temp_c
FROM daily_measurements
GROUP BY month
ORDER BY month;

-- Q5: Are dry and rainy days (precipitation above 1 mm) different in PM2.5 for each station type?
SELECT s.station_type,
       CASE WHEN m.precipitation_mm > 1.0 THEN 'rainy' ELSE 'dry' END AS day_type,
       COUNT(m.pm25)         AS n_obs,
       ROUND(AVG(m.pm25), 1) AS mean_pm25
FROM daily_measurements AS m
JOIN stations AS s ON s.station_id = m.station_id
GROUP BY s.station_type, day_type
ORDER BY s.station_type, day_type;

-- Q6: Which 10 days had the highest city-wide mean PM2.5, and what was the weather like?
SELECT date,
       ROUND(AVG(pm25), 1)          AS city_mean_pm25,
       ROUND(AVG(temperature_c), 1) AS mean_temp_c,
       ROUND(AVG(wind_speed_ms), 1) AS mean_wind_ms,
       ROUND(AVG(precipitation_mm), 1) AS mean_precip_mm
FROM daily_measurements
GROUP BY date
HAVING COUNT(pm25) >= 3
ORDER BY city_mean_pm25 DESC
LIMIT 10;

-- Q7: How does PM2.5 depend on wind speed (wind bands) and how many observations are in each band?
SELECT CASE
           WHEN wind_speed_ms < 2 THEN '1: below 2 m/s'
           WHEN wind_speed_ms < 4 THEN '2: 2 to 4 m/s'
           WHEN wind_speed_ms < 6 THEN '3: 4 to 6 m/s'
           ELSE '4: 6 m/s and above'
       END AS wind_band,
       COUNT(pm25)         AS n_obs,
       ROUND(AVG(pm25), 1) AS mean_pm25,
       ROUND(AVG(no2), 1)  AS mean_no2
FROM daily_measurements
GROUP BY wind_band
ORDER BY wind_band;

-- Q8: How did the annual city mean PM2.5 change from year to year?
WITH yearly AS (
    SELECT CAST(substr(date, 1, 4) AS INTEGER) AS year,
           AVG(pm25) AS mean_pm25
    FROM daily_measurements
    GROUP BY year
)
SELECT year,
       ROUND(mean_pm25, 2) AS mean_pm25,
       ROUND(LAG(mean_pm25) OVER (ORDER BY year), 2) AS prev_year_mean,
       ROUND(mean_pm25 - LAG(mean_pm25) OVER (ORDER BY year), 2) AS change_vs_prev_year
FROM yearly
ORDER BY year;
