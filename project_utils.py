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

    food["year"] = food["date"].dt.year
    master = food.merge(rainfall, on=["state", "date"], how="left")
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
        required = {"avg_food_price", "risk_score", "nutrition_burden"}
        if required.issubset(df.columns):
            return df.sort_values(["state", "date"]).reset_index(drop=True)
        return add_derived_columns(df)
    return save_master_dataframe()


def latest_state_ranking(master: pd.DataFrame) -> pd.DataFrame:
    latest = (
        master.sort_values(["state", "date"])
        .groupby("state", as_index=False)
        .tail(1)
        .copy()
    )
    latest = latest.sort_values("risk_score", ascending=False).reset_index(drop=True)

    latest["risk_band"] = pd.qcut(
        latest["risk_score"].rank(method="first"),
        q=3,
        labels=["Low", "Medium", "High"],
    )
    return latest
