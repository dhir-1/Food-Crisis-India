# FoodGuard India

FoodGuard India is a state-level food distress monitoring and early warning prototype built from public Indian datasets on food prices, rainfall, crop productivity, and nutrition.

## What The Project Does

- Cleans and standardizes multi-source public data
- Merges the cleaned datasets into one monthly state-level master table
- Generates 8 exploratory charts for trends, relationships, and vulnerability
- Runs a Streamlit dashboard for interactive analysis

## Final Project Scope

This repo is intentionally locked to a **state-level** version of the project. It focuses on:

- food affordability pressure through staple prices
- weather stress through rainfall
- agricultural strength through crop productivity
- baseline vulnerability through nutrition indicators

It does **not** claim to be a district-level famine or IPC classifier.

## Datasets Used

- Food prices: state-wise retail prices of essential commodities
- Rainfall: state-wise monthly rainfall
- Crop productivity: annual state crop yield / production
- Nutrition: NFHS state-level stunting, underweight, and anaemia indicators

## Repo Structure

```text
charts/                  Generated chart outputs
data/cleaned/            Final cleaned CSVs used to build the master table
notebooks/               Cleaning and EDA notebooks
scripts/build_master.py  Rebuilds data/master.csv from cleaned inputs
scripts/generate_charts.py
app.py                   Streamlit dashboard
project_utils.py         Shared data-loading and risk-scoring helpers
requirements.txt
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

3. Regenerate the charts

```bash
python scripts/generate_charts.py
```

4. Launch the dashboard

```bash
streamlit run app.py
```

## Output Files

- `data/master.csv`
- `charts/01_food_price_trends.png`
- `charts/02_avg_annual_rainfall.png`
- `charts/03_annual_rainfall_trends.png`
- `charts/04_rainfall_vs_yield.png`
- `charts/05_rainfall_vs_price.png`
- `charts/06_rice_price_distribution.png`
- `charts/07_correlation_heatmap.png`
- `charts/08_food_security_risk_ranking.png`

## Notes On Risk Score

The dashboard uses an **exploratory heuristic risk score** that combines:

- higher food price pressure
- lower rainfall exposure
- lower crop productivity
- higher nutrition burden

It is meant for monitoring and ranking, not as a production crisis label.

## Resume-Friendly Summary

Built a state-level food distress early warning prototype for India by integrating public datasets on food prices, rainfall, crop productivity, and nutrition; developed a cleaned monthly panel, exploratory charts, and an interactive Streamlit dashboard for vulnerability monitoring.
