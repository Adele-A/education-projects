-- SQL analysis of the synthetic stock data (SQLite dialect).
-- Tables: tickers(ticker, sector, annual_drift, annual_volatility, start_price)
--         prices(date, ticker, open, high, low, close, volume, volatility_regime)
-- All data is SYNTHETIC. Dates are stored as text in YYYY-MM-DD format.
-- Daily returns use simple returns (close / previous close - 1) so that no
-- optional SQLite math functions are needed.
-- Every query ends with a semicolon and comments never contain one.

-- Q1: What is the price range, total return and average volume of each ticker, together with its sector?
WITH bounds AS (
    SELECT ticker, MIN(date) AS first_date, MAX(date) AS last_date
    FROM prices
    GROUP BY ticker
)
SELECT
    t.ticker,
    t.sector,
    f.close AS first_close,
    l.close AS last_close,
    ROUND(l.close / f.close - 1, 4) AS total_return,
    (SELECT MIN(close) FROM prices p WHERE p.ticker = t.ticker) AS min_close,
    (SELECT MAX(close) FROM prices p WHERE p.ticker = t.ticker) AS max_close,
    (SELECT ROUND(AVG(volume)) FROM prices p WHERE p.ticker = t.ticker) AS avg_volume
FROM tickers t
JOIN bounds b ON b.ticker = t.ticker
JOIN prices f ON f.ticker = b.ticker AND f.date = b.first_date
JOIN prices l ON l.ticker = b.ticker AND l.date = b.last_date
ORDER BY total_return DESC;

-- Q2: What is the monthly return of each ticker (month-end close versus the previous month-end close)?
WITH month_last AS (
    SELECT ticker, strftime('%Y-%m', date) AS month, MAX(date) AS last_date
    FROM prices
    GROUP BY ticker, strftime('%Y-%m', date)
),
month_end AS (
    SELECT m.ticker, m.month, p.close
    FROM month_last m
    JOIN prices p ON p.ticker = m.ticker AND p.date = m.last_date
),
with_return AS (
    SELECT
        ticker,
        month,
        close AS month_end_close,
        ROUND(close / LAG(close) OVER (PARTITION BY ticker ORDER BY month) - 1, 4) AS monthly_return
    FROM month_end
)
SELECT ticker, month, month_end_close, monthly_return
FROM with_return
WHERE monthly_return IS NOT NULL
ORDER BY ticker, month;

-- Q3: Which ticker-months were the most volatile (highest mean absolute daily return)?
WITH daily AS (
    SELECT
        ticker,
        date,
        close / LAG(close) OVER (PARTITION BY ticker ORDER BY date) - 1 AS ret
    FROM prices
)
SELECT
    ticker,
    strftime('%Y-%m', date) AS month,
    COUNT(*) AS trading_days,
    ROUND(AVG(ABS(ret)), 4) AS mean_abs_daily_return,
    ROUND(MIN(ret), 4) AS worst_day,
    ROUND(MAX(ret), 4) AS best_day
FROM daily
WHERE ret IS NOT NULL
GROUP BY ticker, strftime('%Y-%m', date)
ORDER BY mean_abs_daily_return DESC
LIMIT 10;

-- Q4: How do the calm and stressed regimes differ in size of moves and trading volume?
WITH daily AS (
    SELECT
        ticker,
        volatility_regime,
        volume,
        close / LAG(close) OVER (PARTITION BY ticker ORDER BY date) - 1 AS ret
    FROM prices
)
SELECT
    ticker,
    volatility_regime,
    COUNT(*) AS days,
    ROUND(AVG(ABS(ret)), 4) AS mean_abs_daily_return,
    ROUND(AVG(volume)) AS avg_volume
FROM daily
WHERE ret IS NOT NULL
GROUP BY ticker, volatility_regime
ORDER BY ticker, volatility_regime;

-- Q5: What were the 10 largest one-day price moves, and in which regime did they happen?
WITH daily AS (
    SELECT
        ticker,
        date,
        volatility_regime,
        volume,
        close / LAG(close) OVER (PARTITION BY ticker ORDER BY date) - 1 AS ret
    FROM prices
)
SELECT ticker, date, volatility_regime, ROUND(ret, 4) AS daily_return, volume
FROM daily
WHERE ret IS NOT NULL
ORDER BY ABS(ret) DESC
LIMIT 10;

-- Q6: On what share of days does each ticker close above its 20-day simple moving average?
WITH ma AS (
    SELECT
        ticker,
        date,
        close,
        AVG(close) OVER w AS sma_20,
        COUNT(*) OVER w AS window_size
    FROM prices
    WINDOW w AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
)
SELECT
    ticker,
    COUNT(*) AS days_with_sma,
    SUM(CASE WHEN close > sma_20 THEN 1 ELSE 0 END) AS days_above_sma,
    ROUND(AVG(CASE WHEN close > sma_20 THEN 1.0 ELSE 0.0 END), 4) AS share_above_sma
FROM ma
WHERE window_size = 20
GROUP BY ticker
ORDER BY share_above_sma DESC;

-- Q7: What is the maximum drawdown of each ticker (largest fall from a running peak of the close)?
WITH running AS (
    SELECT
        ticker,
        date,
        close,
        MAX(close) OVER (PARTITION BY ticker ORDER BY date) AS running_peak
    FROM prices
),
dd AS (
    SELECT ticker, date, close, running_peak, close / running_peak - 1 AS drawdown
    FROM running
)
SELECT
    ticker,
    ROUND(MIN(drawdown), 4) AS max_drawdown,
    (SELECT d2.date FROM dd d2
     WHERE d2.ticker = dd.ticker
     ORDER BY d2.drawdown ASC
     LIMIT 1) AS trough_date
FROM dd
GROUP BY ticker
ORDER BY max_drawdown;

-- Q8: Is volume higher on days with large moves (absolute return above 3 percent) than on other days?
WITH daily AS (
    SELECT
        ticker,
        volume,
        close / LAG(close) OVER (PARTITION BY ticker ORDER BY date) - 1 AS ret
    FROM prices
)
SELECT
    ticker,
    CASE WHEN ABS(ret) > 0.03 THEN 'large move' ELSE 'normal day' END AS day_type,
    COUNT(*) AS days,
    ROUND(AVG(volume)) AS avg_volume
FROM daily
WHERE ret IS NOT NULL
GROUP BY ticker, day_type
ORDER BY ticker, day_type;
