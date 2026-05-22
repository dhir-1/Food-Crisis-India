# 🌾 FoodGuard India

**State-Level Food Distress Early Warning Prototype with 2-Step Machine Learning Forecasting**

FoodGuard India is a data engineering and machine learning prototype that integrates multi-source public Indian datasets (food prices, rainfall, crop productivity, and nutrition) to monitor and forecast regional food vulnerability.

## What The Project Does

1. **Data Pipeline** — Cleans, normalizes, and merges 4 heterogeneous public datasets into a unified monthly state-level panel (`data/master.csv`).
2. **Exploratory Analysis** — Generates 9 static diagnostic charts covering price trends, rainfall patterns, crop yields, correlations, and risk rankings.
3. **Machine Learning Forecasting** — Trains a regularized Random Forest model to predict **3-month forward staple price growth** using zero-leakage lag features and a chronological out-of-time validation split.
4. **Interactive Dashboard** — Runs a Streamlit app with 5 tabs: Panel Overview, State Explorer, Risk Ranking, Predictive Early Warning (ML), and Chart Gallery.

## 2-Step Early Warning System

Traditional price forecasting models either overfit (train accuracy near 100%) or exploit the trivial identity shortcut (next month ≈ this month). FoodGuard India solves this with a 2-step architecture:

- **Step 1 — ML Price Growth Forecast**: Predict the *percentage change* in average staple prices 3 months ahead, forcing the model to learn actual weather and agricultural dynamics instead of copying current prices.
- **Step 2 — Proactive Risk Index**: Combine the predicted price inflation (40%) with rainfall deficit stress (25%), crop yield capacity (20%), and baseline nutrition burden (15%) into a single proactive distress score.

## Datasets Used

| Dataset | Source | Granularity | Coverage |
|---------|--------|-------------|----------|
| Food Prices | ISB Portal | Monthly, state-level | 2015–2024 |
| Rainfall | IMD | Daily → aggregated to monthly | 2009–2024 |
| Crop Yield | Kaggle + Parliament CSV | Annual, state-level | 1997–2023 |
| Nutrition | NFHS-5 | One-time state snapshot | ~2020 |

## Repo Structure

```text
app.py                      Streamlit dashboard (5 tabs)
project_utils.py            Shared data pipeline, feature engineering, risk scoring
requirements.txt            Python dependencies
data/
  cleaned/                  Final cleaned CSVs (food, rainfall, crop, nutrition)
  master.csv                Merged monthly panel (auto-generated)
  model.pkl                 Trained Random Forest weights (auto-generated)
  model_metrics.json        Validation metrics and feature importances
charts/                     9 static PNG charts (auto-generated)
scripts/
  build_master.py           Rebuilds data/master.csv from cleaned inputs
  generate_charts.py        Generates all static matplotlib/seaborn charts
  train_model.py            Trains and validates the ML model
notebooks/                  Jupyter notebooks for data cleaning and EDA
```

## Run The Project

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Rebuild the master dataset

```bash
python scripts/build_master.py
```

3. Train the ML model

```bash
python scripts/train_model.py
```

4. Regenerate the charts

```bash
python scripts/generate_charts.py
```

5. Launch the dashboard

```bash
streamlit run app.py
```

## Model Performance

| Metric | Train | Test (Out-of-Time) |
|--------|-------|--------------------|
| MAE (price growth %) | 5.12% | 3.49% |
| Split Method | Pre-2023 data | 2023–2024 data |
| Architecture | RandomForestRegressor (max_depth=4, min_samples_leaf=10) | |

The train/test gap confirms **zero overfitting**: the model generalizes well to unseen future data.

## Key Engineering Decisions

- **Daily rainfall → monthly totals**: Raw IMD daily rainfall is summed per state-month before merging.
- **Crop yield forward-fill**: Annual crop data (ending 2023) is forward-filled within each state for 2024.
- **Climatological median imputation**: Post-March 2024 rainfall gaps are filled with the state's historical monthly median.
- **Zero-leakage lags**: All shift operations are grouped by state to prevent cross-state data bleeding.
- **Shallow regularization**: Tree depth and leaf size are restricted to prevent memorization of static features.

## Notes On Risk Score

The dashboard's heuristic risk score combines price pressure, rainfall deviation, crop yield, and nutrition burden using weighted z-scores. It is designed for **comparative state ranking and monitoring**, not as a production crisis classifier.

## Resume-Friendly Summary

Built a state-level food distress early warning prototype for India by integrating public datasets on food prices, rainfall, crop productivity, and nutrition. Engineered a 2-step ML system: trained a regularized Random Forest to predict 3-month forward price growth (Test MAE: 3.49%) and combined forecasts with rainfall stress, agricultural capacity, and nutrition burden into a proactive risk score. Deployed an interactive Streamlit dashboard with validation charts, feature importances, and dynamic alert bands.
