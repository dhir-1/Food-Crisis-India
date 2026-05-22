from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from project_utils import CHARTS_DIR, load_master_dataframe, latest_state_ranking


sns.set_theme(style="whitegrid")


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    CHARTS_DIR.mkdir(exist_ok=True)
    master = load_master_dataframe()
    ranking = latest_state_ranking(master)
    top_states = ranking.head(5)["state"].tolist()

    state_trends = master[master["state"].isin(top_states)]
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=state_trends, x="date", y="avg_food_price", hue="state")
    plt.title("Food Price Trends In Higher-Risk States")
    plt.xlabel("Date")
    plt.ylabel("Average Food Price")
    savefig(CHARTS_DIR / "01_food_price_trends.png")

    annual_rainfall = (
        master.groupby("state", as_index=False)["rainfall_mm"]
        .mean()
        .sort_values("rainfall_mm", ascending=False)
    )
    plt.figure(figsize=(12, 8))
    sns.barplot(data=annual_rainfall.head(15), y="state", x="rainfall_mm", palette="Blues_r")
    plt.title("States With Highest Average Rainfall")
    plt.xlabel("Average Monthly Rainfall (mm)")
    plt.ylabel("State")
    savefig(CHARTS_DIR / "02_avg_annual_rainfall.png")

    national_rainfall = master.groupby("year", as_index=False)["rainfall_mm"].mean()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=national_rainfall, x="year", y="rainfall_mm", marker="o", color="#0f766e")
    plt.title("Average Annual Rainfall Trend")
    plt.xlabel("Year")
    plt.ylabel("Average Rainfall (mm)")
    savefig(CHARTS_DIR / "03_annual_rainfall_trends.png")

    crop_scatter = master.dropna(subset=["rainfall_mm", "yield_ton_per_hectare"])
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=crop_scatter,
        x="rainfall_mm",
        y="yield_ton_per_hectare",
        hue="risk_score",
        palette="coolwarm",
        alpha=0.6,
    )
    plt.title("Rainfall vs Crop Yield")
    plt.xlabel("Rainfall (mm)")
    plt.ylabel("Yield (ton/hectare)")
    savefig(CHARTS_DIR / "04_rainfall_vs_yield.png")

    price_scatter = master.dropna(subset=["rainfall_mm", "avg_food_price"])
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=price_scatter,
        x="rainfall_mm",
        y="avg_food_price",
        hue="risk_score",
        palette="viridis",
        alpha=0.6,
    )
    plt.title("Rainfall vs Food Price Pressure")
    plt.xlabel("Rainfall (mm)")
    plt.ylabel("Average Food Price")
    savefig(CHARTS_DIR / "05_rainfall_vs_price.png")

    rice_distribution = (
        master.groupby("state", as_index=False)["rice_price"].mean()
        .sort_values("rice_price", ascending=False)
        .head(10)["state"]
        .tolist()
    )
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=master[master["state"].isin(rice_distribution)], x="state", y="rice_price", palette="Set2")
    plt.title("Rice Price Distribution Across Selected States")
    plt.xlabel("State")
    plt.ylabel("Rice Price")
    plt.xticks(rotation=45, ha="right")
    savefig(CHARTS_DIR / "06_rice_price_distribution.png")

    corr_cols = [
        "rice_price",
        "wheat_price",
        "dal_price",
        "oil_price",
        "rainfall_mm",
        "yield_ton_per_hectare",
        "stunting_rate",
        "underweight_rate",
        "anaemia_rate",
        "risk_score",
    ]
    plt.figure(figsize=(10, 8))
    sns.heatmap(master[corr_cols].corr(numeric_only=True), cmap="RdBu_r", center=0, annot=True, fmt=".2f")
    plt.title("Correlation Heatmap")
    savefig(CHARTS_DIR / "07_correlation_heatmap.png")

    plt.figure(figsize=(12, 8))
    sns.barplot(data=ranking.head(15), y="state", x="risk_score", palette="Reds_r")
    plt.title("Latest Food Security Risk Ranking")
    plt.xlabel("Exploratory Risk Score")
    plt.ylabel("State")
    savefig(CHARTS_DIR / "08_food_security_risk_ranking.png")

    print(f"Saved charts to: {CHARTS_DIR}")


if __name__ == "__main__":
    main()
