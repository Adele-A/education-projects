# Delivery Time Analysis

## Business question
Which factors drive food delivery times, and how much extra time do long distances, bad weather and a heavily loaded courier add? Can we predict the delivery time of an order well enough to give customers a realistic ETA?

## Goal
Quantify the effect of distance, weather and courier load on delivery time, and build a regression model that predicts delivery time (in minutes) for a new order.

## Plan of the work
1. **Data generation** – create a synthetic dataset of couriers and orders (`src/data_generator.py`).
2. **EDA** – check data quality, distributions and relationships between delivery time and distance, weather, courier load and time of day (`src/eda.py`).
3. **Analysis / model** – baseline and multiple linear regression (plus a comparison with a tree-based model), validated on a hold-out set (`src/analysis.py`, helpers in `src/utils.py`).
4. **SQL** – analytical queries on the order and courier tables (average delivery time by weather, by load, by vehicle, etc.).
5. **Tests** – pytest checks for the data generator and the data quality of the generated tables.

## Dataset (synthetic)
All data is **synthetic**. It is produced by `src/data_generator.py` with `numpy` and a fixed random seed (42). It does not describe any real company, courier or customer. Relationships in the data (for example, rain slows delivery down) are assumptions written into the generator, not real-world findings.

Two tables are saved to `data/`:
- `couriers.csv` – one row per courier (300 rows).
- `orders.csv` – one row per order (5000 rows).

Full column descriptions are in [data_description.md](data_description.md).

## Exploratory data analysis
`src/eda.py` loads both CSV files from `data/` and:
- prints the shape, data types, missing values and duplicate count of each table;
- prints descriptive statistics for the numeric columns;
- prints aggregations of delivery time (order count, mean, median) by weather, courier load, vehicle type, distance band and peak-hour flag, plus the correlation of numeric features with delivery time;
- saves charts to `figures/`:
  - `delivery_time_distribution.png` – histogram of the target;
  - `delivery_vs_distance.png` – delivery time vs distance, colored by weather;
  - `delivery_by_weather_and_load.png` – boxplots by weather and by courier load;
  - `delivery_by_hour.png` – mean delivery time by hour of day.

Because the data is synthetic, any patterns seen in the output reflect the assumptions coded in the generator, not real-world behavior. Findings should be read from the actual output of the script.

## Method
`src/analysis.py` joins orders with courier attributes and predicts `delivery_time_min`.

- **Features:** distance, courier load (active orders), preparation time, peak-hour flag, courier experience, courier rating, one-hot weather (reference: `clear`) and one-hot vehicle type (reference: `bike`). Feature preparation lives in `src/utils.py`.
- **Split:** 80% train / 20% test, `random_state=42`. All reported metrics are computed on the untouched test set.
- **Models compared:**
  1. mean baseline (predicts the training mean);
  2. simple linear regression on distance only;
  3. multiple linear regression on all features;
  4. random forest (200 trees, `min_samples_leaf=5`) to capture non-linear effects and interactions.
- **Metrics:** MAE and RMSE (minutes) and R². The multiple linear model also gets a 5-fold cross-validated R² on the training set.
- **Effect sizes:** an OLS fit (statsmodels) on the training set gives coefficients with 95% confidence intervals, read as extra minutes per km of distance, per additional active order, and for rain/snow compared with clear weather. Random forest feature importances are printed as a second view.
- **Outputs:**
  - console report with all metrics, labelled;
  - `reports/metrics.json` – metrics summary, OLS effects and feature importances;
  - `figures/model_coefficients.png` – OLS coefficients with confidence intervals;
  - `figures/predicted_vs_actual.png` – hold-out predictions of the best model (lowest test RMSE) vs actual values.

**Caveats.**
- The data is synthetic, so the estimated effects mostly recover assumptions written into the generator; they are not evidence about real delivery operations.
- The weather effect in the generator multiplies travel time, so it grows with distance. A purely additive linear model can only approximate this, which is one reason the tree-based model is included.
- `prep_time_min` is used as a feature; in a real ETA system it would have to be known (or estimated) at the moment the ETA is shown.

Actual numbers should be read from the script output and `reports/metrics.json`.

## How to run
From the project root:

```bash
pip install -r requirements.txt
python src/data_generator.py
python src/eda.py
python src/analysis.py
```

All scripts are non-interactive. The generator writes `data/couriers.csv` and `data/orders.csv`; the EDA script prints its report and writes PNG files to `figures/`; the analysis script prints the model metrics, writes two more charts to `figures/` and creates `reports/metrics.json`.

---

*This is a training project that uses synthetic data generated by AI.*
