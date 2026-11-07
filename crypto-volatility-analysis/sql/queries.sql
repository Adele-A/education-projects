-- SQL analysis of the SYNTHETIC crypto tables (SQLite dialect).
-- Tables: assets, market_regimes, daily_prices (loaded from data/*.csv by src/run_sql.py).
-- Notes:
--   * Volatility is the POPULATION standard deviation computed as
--     SQRT(AVG(r*r) - AVG(r)*AVG(r)), so it differs slightly from the pandas
--     (sample) values in the analysis step.
--   * SQRT is registered by src/run_sql.py, so the script does not depend on
--     how SQLite was compiled.
--   * Statements are separated by a semicolon at the end of each query.

-- Q1: What are the return and annualized volatility of each asset over the whole period?
SELECT
    p.asset,
    a.category,
    COUNT(*) AS n_days,
    ROUND(AVG(p.log_return), 5) AS mean_daily_return,
    ROUND(SQRT(AVG(p.log_return * p.log_return)
               - AVG(p.log_return) * AVG(p.log_return)) * SQRT(365.0), 4) AS annualized_vol,
    ROUND(MIN(p.log_return), 4) AS worst_day,
    ROUND(MAX(p.log_return), 4) AS best_day
FROM daily_prices AS p
JOIN assets AS a ON a.asset = p.asset
GROUP BY p.asset, a.category
ORDER BY annualized_vol DESC;

-- Q2: How much higher is each asset's volatility on turbulent days than on calm days?
SELECT
    p.asset,
    r.regime,
    COUNT(*) AS n_days,
    ROUND(SQRT(AVG(p.log_return * p.log_return)
               - AVG(p.log_return) * AVG(p.log_return)) * SQRT(365.0), 4) AS annualized_vol
FROM daily_prices AS p
JOIN market_regimes AS r ON r.date = p.date
GROUP BY p.asset, r.regime
ORDER BY p.asset, r.regime;

-- Q3: What was the worst single day of each asset and which regime was it in?
WITH ranked AS (
    SELECT
        asset,
        date,
        log_return,
        ROW_NUMBER() OVER (PARTITION BY asset ORDER BY log_return ASC) AS rn
    FROM daily_prices
)
SELECT
    k.asset,
    k.date,
    ROUND(k.log_return, 4) AS log_return,
    r.regime
FROM ranked AS k
JOIN market_regimes AS r ON r.date = k.date
WHERE k.rn = 1
ORDER BY k.log_return ASC;

-- Q4: When was BTC's 30-day rolling annualized volatility highest (window function)?
WITH btc AS (
    SELECT
        date,
        log_return,
        log_return * log_return AS sq_return
    FROM daily_prices
    WHERE asset = 'BTC'
),
rolling AS (
    SELECT
        date,
        COUNT(*) OVER w AS n_obs,
        AVG(sq_return) OVER w AS mean_sq,
        AVG(log_return) OVER w AS mean_ret
    FROM btc
    WINDOW w AS (ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)
)
SELECT
    date,
    ROUND(SQRT(mean_sq - mean_ret * mean_ret) * SQRT(365.0), 4) AS rolling_30d_vol
FROM rolling
WHERE n_obs = 30
ORDER BY rolling_30d_vol DESC
LIMIT 10;

-- Q5: How often does each asset lose more than 5 percent in a day, by regime (tail events)?
SELECT
    p.asset,
    r.regime,
    COUNT(*) AS n_days,
    SUM(CASE WHEN p.log_return < -0.05 THEN 1 ELSE 0 END) AS n_big_losses,
    ROUND(1.0 * SUM(CASE WHEN p.log_return < -0.05 THEN 1 ELSE 0 END) / COUNT(*), 4) AS share_of_days
FROM daily_prices AS p
JOIN market_regimes AS r ON r.date = p.date
GROUP BY p.asset, r.regime
ORDER BY p.asset, r.regime;

-- Q6: What is the monthly return of an equal-weight basket and how many turbulent days did each month have?
WITH basket AS (
    SELECT date, AVG(log_return) AS basket_return
    FROM daily_prices
    GROUP BY date
)
SELECT
    SUBSTR(b.date, 1, 7) AS month,
    ROUND(SUM(b.basket_return), 4) AS basket_log_return,
    SUM(CASE WHEN r.regime = 'turbulent' THEN 1 ELSE 0 END) AS turbulent_days,
    COUNT(*) AS n_days
FROM basket AS b
JOIN market_regimes AS r ON r.date = b.date
GROUP BY month
ORDER BY month;

-- Q7: Does trading volume rise with the size of the move (volume relative to asset average, by |return| quartile)?
WITH scored AS (
    SELECT
        asset,
        volume_usd / AVG(volume_usd) OVER (PARTITION BY asset) AS rel_volume,
        NTILE(4) OVER (PARTITION BY asset ORDER BY ABS(log_return)) AS move_quartile
    FROM daily_prices
)
SELECT
    move_quartile,
    COUNT(*) AS n_obs,
    ROUND(AVG(rel_volume), 4) AS mean_relative_volume
FROM scored
GROUP BY move_quartile
ORDER BY move_quartile;

-- Q8: Do assets fall together more often under stress (days on which all or none of the 6 assets fell)?
WITH per_day AS (
    SELECT
        date,
        SUM(CASE WHEN log_return < 0 THEN 1 ELSE 0 END) AS n_down,
        COUNT(*) AS n_assets
    FROM daily_prices
    GROUP BY date
)
SELECT
    r.regime,
    COUNT(*) AS n_days,
    ROUND(1.0 * SUM(CASE WHEN d.n_down = d.n_assets THEN 1 ELSE 0 END) / COUNT(*), 4) AS share_all_down,
    ROUND(1.0 * SUM(CASE WHEN d.n_down = 0 THEN 1 ELSE 0 END) / COUNT(*), 4) AS share_all_up
FROM per_day AS d
JOIN market_regimes AS r ON r.date = d.date
GROUP BY r.regime
ORDER BY r.regime;
