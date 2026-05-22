from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CLEANED_DIR = DATA_DIR / "cleaned"
CHARTS_DIR = ROOT_DIR / "charts"
MASTER_PATH = DATA_DIR / "master.csv"

PRICE_COLUMNS = ["rice_price", "wheat_price", "dal_price", "oil_price"]

STATE_ALIASES = {
    "andaman & nicobar islands": "Andaman And Nicobar Islands",
    "andaman and nicobar islands": "Andaman And Nicobar Islands",
    "dadra & nagar haveli": "Dadra And Nagar Haveli And Daman And Diu",
    "dadra and nagar haveli": "Dadra And Nagar Haveli And Daman And Diu",
    "daman & diu": "Dadra And Nagar Haveli And Daman And Diu",
    "jammu & kashmir": "Jammu And Kashmir",
    "orissa": "Odisha",
    "pondicherry": "Puducherry",
    "uttaranchal": "Uttarakhand",
    "maharastra": "Maharashtra",
}


def normalize_state(value: str) -> str:
    if pd.isna(value):
        return value
    cleaned = " ".join(str(value).strip().replace("&", "And").split())
    lowered = cleaned.lower()
    return STATE_ALIASES.get(lowered, cleaned.title())


def _safe_zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    std = numeric.std(skipna=True)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (numeric - numeric.mean(skipna=True)) / std


def load_cleaned_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    food = pd.read_csv(CLEANED_DIR / "food_prices_cleaned.csv")
    rainfall = pd.read_csv(CLEANED_DIR / "rainfall_cleaned.csv")
    crop = pd.read_csv(CLEANED_DIR / "crop_yield_cleaned.csv")
    nutrition = pd.read_csv(CLEANED_DIR / "nutrition_cleaned.csv")

    food["date"] = pd.to_datetime(food["date"])
    rainfall["date"] = pd.to_datetime(rainfall["date"])
    crop["year"] = pd.to_numeric(crop["year"], errors="coerce").astype("Int64")

    for frame in [food, rainfall, crop, nutrition]:
        frame["state"] = frame["state"].map(normalize_state)

    return food, rainfall, crop, nutrition


def add_derived_columns(master: pd.DataFrame) -> pd.DataFrame:
    df = master.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["avg_food_price"] = df[PRICE_COLUMNS].mean(axis=1, skipna=True)
    df["nutrition_burden"] = df[["stunting_rate", "underweight_rate", "anaemia_rate"]].mean(axis=1, skipna=True)

    # Sort chronologically within state to perform correct group-by lag calculations
    df = df.sort_values(["state", "date"]).reset_index(drop=True)

    # 1. Handle Missing Crop Yields (Annual capacity ffill + overall median fallback)
    df["yield_ton_per_hectare"] = df.groupby("state")["yield_ton_per_hectare"].ffill()
    df["production_tonnes"] = df.groupby("state")["production_tonnes"].ffill()
    df["yield_ton_per_hectare"] = df["yield_ton_per_hectare"].fillna(df["yield_ton_per_hectare"].median())
    df["production_tonnes"] = df["production_tonnes"].fillna(df["production_tonnes"].median())

    # 2. Handle Rainfall gaps in 2024 using state-specific monthly historical medians (climatological baseline)
    state_month_medians = df.groupby(["state", "month"])["rainfall_mm"].transform("median")
    df["rainfall_mm"] = df["rainfall_mm"].fillna(state_month_medians)
    df["rainfall_mm"] = df["rainfall_mm"].fillna(df["rainfall_mm"].median())  # Final fallback

    # 3. Zero-Leakage Lag Features (Grouped STRICTLY by state before shift)
    df["price_lag_1"] = df.groupby("state")["avg_food_price"].shift(1)
    df["price_lag_2"] = df.groupby("state")["avg_food_price"].shift(2)
    df["price_lag_3"] = df.groupby("state")["avg_food_price"].shift(3)

    df["rain_lag_1"] = df.groupby("state")["rainfall_mm"].shift(1)
    df["rain_lag_2"] = df.groupby("state")["rainfall_mm"].shift(2)

    # Lags for individual commodities
    df["rice_lag_1"] = df.groupby("state")["rice_price"].shift(1)
    df["wheat_lag_1"] = df.groupby("state")["wheat_price"].shift(1)
    df["dal_lag_1"] = df.groupby("state")["dal_price"].shift(1)
    df["oil_lag_1"] = df.groupby("state")["oil_price"].shift(1)

    # Growth indicators
    df["price_growth_1m"] = df["avg_food_price"] / df["price_lag_1"] - 1
    df["price_growth_2m"] = df["price_lag_1"] / df["price_lag_2"] - 1

    # 4. Target Variable: 3-month forward price growth (t+3 / t - 1)
    df["target_growth_3m"] = df.groupby("state")["avg_food_price"].shift(-3) / df["avg_food_price"] - 1

    # 5. Base Heuristic Risk Score (Static fallback before ML predictions are loaded)
    df["price_zscore"] = _safe_zscore(df["avg_food_price"])
    df["rainfall_zscore"] = _safe_zscore(df["rainfall_mm"])
    df["yield_zscore"] = _safe_zscore(df["yield_ton_per_hectare"])
    df["nutrition_zscore"] = _safe_zscore(df["nutrition_burden"])

    df["risk_score"] = (
        df["price_zscore"].fillna(0) * 0.40
        - df["rainfall_zscore"].fillna(0) * 0.25
        - df["yield_zscore"].fillna(0) * 0.20
        + df["nutrition_zscore"].fillna(0) * 0.15
    )

    return df.sort_values(["state", "date"]).reset_index(drop=True)


def build_master_dataframe() -> pd.DataFrame:
    food, rainfall, crop, nutrition = load_cleaned_data()

    # Aggregate daily rainfall to monthly totals (summing rainfall_mm per state-month)
    rainfall["month_date"] = rainfall["date"].dt.to_period("M").dt.to_timestamp()
    monthly_rainfall = (
        rainfall.groupby(["state", "month_date"], as_index=False)["rainfall_mm"]
        .sum(min_count=1)
        .rename(columns={"month_date": "date"})
    )

    food["year"] = food["date"].dt.year
    master = food.merge(monthly_rainfall, on=["state", "date"], how="left")
    master = master.merge(crop, on=["state", "year"], how="left")
    master = master.merge(nutrition, on="state", how="left")

    return add_derived_columns(master)


def save_master_dataframe() -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    master = build_master_dataframe()
    master.to_csv(MASTER_PATH, index=False)
    return master


def load_master_dataframe() -> pd.DataFrame:
    if MASTER_PATH.exists():
        df = pd.read_csv(MASTER_PATH, parse_dates=["date"])
        # Ensure we have our newly engineered features
        required = {"price_lag_1", "target_growth_3m", "rainfall_mm"}
        if required.issubset(df.columns):
            return df.sort_values(["state", "date"]).reset_index(drop=True)
        return add_derived_columns(df)
    return save_master_dataframe()


def compute_proactive_risk_score(df: pd.DataFrame, predicted_growth: pd.Series) -> pd.Series:
    """
    Computes a forward-looking 2-step Food Distress Risk Score:
    - 40% predicted 3-month forward price pressure (growth)
    - 25% climatological monthly rainfall stress (deficit relative to median)
    - 20% agricultural crop yield capacity (negative z-score)
    - 15% baseline NFHS-5 nutrition burden
    """
    # 1. Predicted Growth z-score
    price_z = _safe_zscore(predicted_growth)

    # 2. Rainfall Stress: Deficit relative to the state's historical median for that calendar month
    month = pd.to_datetime(df["date"]).dt.month
    state_medians = df.groupby(["state", month])["rainfall_mm"].transform("median")
    rain_deficit = state_medians - df["rainfall_mm"]
    rain_z = _safe_zscore(rain_deficit)

    # 3. Crop Yield capacity z-score
    yield_z = _safe_zscore(df["yield_ton_per_hectare"])

    # 4. Nutrition Burden z-score
    nutrition_z = _safe_zscore(df["nutrition_burden"])

    # Combined proactive weighted risk index
    proactive_risk = (
        price_z.fillna(0) * 0.40
        + rain_z.fillna(0) * 0.25
        - yield_z.fillna(0) * 0.20
        + nutrition_z.fillna(0) * 0.15
    )
    return proactive_risk


def latest_state_ranking(master: pd.DataFrame, target_date: str = "2024-03-01") -> pd.DataFrame:
    """
    Returns the food distress risk ranking for all states anchored on a specific date.
    Defaults to 2024-03-01, the latest month with fully complete active rainfall metrics.
    """
    df = master.copy()
    df["date"] = pd.to_datetime(df["date"])
    target_dt = pd.to_datetime(target_date)

    # Filter to target date
    dated_df = df[df["date"] == target_dt].copy()

    # If date is out of range, fall back to the maximum date available in the dataset
    if dated_df.empty:
        max_dt = df["date"].max()
        dated_df = df[df["date"] == max_dt].copy()

    # Sort by risk score (descending)
    ranking = dated_df.sort_values("risk_score", ascending=False).reset_index(drop=True)

    # Assign risk bands
    if not ranking.empty:
        ranking["risk_band"] = pd.qcut(
            ranking["risk_score"].rank(method="first"),
            q=3,
            labels=["Low", "Medium", "High"],
        )
    else:
        ranking["risk_band"] = []

    return ranking

