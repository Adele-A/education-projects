# Fraud Transaction Detection

Anomaly detection with Isolation Forest and a supervised baseline, built on **synthetic** card transaction data.

## Business question

Can we flag suspicious card transactions early enough to stop losses, without overwhelming the fraud-review team with false alarms?

## Goal

- Explore how fraudulent transactions differ from legitimate ones (amount, time of day, channel, location, device).
- Build an unsupervised anomaly detector (Isolation Forest) that does not need labels.
- Build a supervised baseline (logistic regression) that uses the fraud label.
- Compare both approaches with metrics suited to highly imbalanced data (precision, recall, PR-AUC).

## Plan of the work

1. **Data generation**: create reproducible synthetic customers and transactions (`src/data_generator.py`).
2. **EDA**: class balance, distributions by fraud label, time patterns, categorical breakdowns (`src/eda.py`).
3. **Analysis / model**: feature engineering, Isolation Forest, supervised baseline, comparison (`src/analysis.py`, `src/utils.py`).
4. **SQL**: analytical queries on the same tables (fraud rate by segment, customer-level aggregates).
5. **Tests**: pytest checks for the helper functions (`tests/test_utils.py`); data integrity and generator tests still to be added.

## Dataset (synthetic)

All data is **synthetic** and produced by this project's own code with a fixed random seed (42). It does not describe real customers or real transactions. Two tables are saved to `data/`:

| File | Rows | Description |
|------|------|-------------|
| `data/customers.csv` | 1,000 | One row per customer: demographics, income, credit limit |
| `data/transactions.csv` | 5,000 | One row per card transaction, with an `is_fraud` label |

The fraud label is assigned by the generator, and fraudulent transactions are given different distributions (larger amounts, more night-time, more online and foreign activity, longer distances from home). Column details are in [data_description.md](data_description.md).

## Exploratory data analysis

`src/eda.py` loads both CSV files and:

- prints the shape, data types, missing values and descriptive statistics of each table;
- prints five aggregations: class balance, amount statistics by label, and fraud rate by channel, by merchant category, and by night-time / foreign flags;
- saves three charts to `figures/`:
  - `class_balance_and_amount.png`: class balance and amount distribution by label;
  - `fraud_by_hour.png`: hour-of-day profile of each class;
  - `fraud_rate_by_segment.png`: fraud rate by channel and merchant category.

All numbers appear in the script's printed output and are computed from the synthetic data. Because the fraud patterns are built in by the generator, the differences seen in the EDA reflect the simulation design, not real-world behaviour.

## Method

`src/analysis.py` compares two detectors. Reusable logic (feature building and metrics) lives in `src/utils.py`.

**Features** (`build_features`): transactions are joined to customer attributes. Features are log amount, amount relative to the credit limit, log distance from home, hour of day, a night-time flag, foreign flag, new-device flag, customer age, account age, and one-hot encoded channel and merchant category. The label is never used as a feature.

**Split**: stratified 70/30 train/test split with `random_state=42`, so both sets keep the same fraud share. All metrics are computed on the test set only.

**Models**
- *Isolation Forest* (unsupervised): fitted on the training features without labels; the anomaly score (higher = more unusual) ranks test transactions.
- *Logistic regression* (supervised baseline): standardised features, `class_weight="balanced"` to handle the 2% fraud share; the predicted fraud probability ranks test transactions.

**Metrics**: with so few frauds, accuracy is misleading, so the comparison uses
- PR-AUC (average precision) and ROC-AUC, which evaluate the full ranking;
- precision, recall and F1 at a fixed **alert budget**: the top 2% of transactions by score are flagged, which mimics a review team that can only check a limited number of cases.

**Outputs**
- `figures/pr_curves.png`: precision-recall curves of both models with the random baseline;
- `figures/logistic_coefficients.png`: largest standardised coefficients of the logistic regression;
- `reports/metrics.json`: all metrics and the run configuration.

The actual metric values are printed by the script and stored in `reports/metrics.json`; run the project to see them. Because the data is simulated with built-in fraud patterns, the results show how the methods behave on this synthetic setup and say nothing about performance on real transactions. A supervised model is expected to use the labels to its advantage, while Isolation Forest shows what is possible when no labels exist.

## Project structure

```
fraud-transaction-detection/
├── README.md
├── requirements.txt
├── data_description.md
├── data/            # created by the generator
├── figures/         # created by the EDA and analysis scripts
├── reports/         # metrics.json, created by the analysis script
├── src/
│   ├── data_generator.py
│   ├── eda.py
│   ├── utils.py
│   └── analysis.py
├── sql/
└── tests/
    └── test_utils.py
```

## How to run

From the project root:

```bash
pip install -r requirements.txt
python src/data_generator.py
python src/eda.py
python src/analysis.py
pytest
```

The generator writes `data/customers.csv` and `data/transactions.csv` and prints a short summary. Running it again reproduces the same files. The EDA script prints the analysis and writes charts to `figures/`. The analysis script trains both models, prints the test-set metrics, saves two charts and writes `reports/metrics.json`.

---

*This is a training project that uses synthetic data generated by AI.*
