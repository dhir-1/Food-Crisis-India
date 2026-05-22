from pathlib import Path
import json
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

from project_utils import MASTER_PATH, DATA_DIR, CHARTS_DIR


def main() -> None:
    if not MASTER_PATH.exists():
        print("master.csv not found! Re-run build_master.py first.")
        sys.exit(1)

    df = pd.read_csv(MASTER_PATH)
    df["date"] = pd.to_datetime(df["date"])

    # Define features and target
    feature_cols = [
        "avg_food_price",
        "price_lag_1",
        "price_lag_2",
        "price_lag_3",
        "price_growth_1m",
        "price_growth_2m",
        "rainfall_mm",
        "rain_lag_1",
        "rain_lag_2",
        "yield_ton_per_hectare",
        "stunting_rate",
        "underweight_rate",
        "anaemia_rate",
    ]

    target_col = "target_growth_3m"

    # Drop missing target or lag rows for training/validation
    train_ready = df.dropna(subset=[target_col] + feature_cols).copy()

    X = train_ready[feature_cols]
    y = train_ready[target_col]

    # Chronological Split (Train < 2023, Test >= 2023)
    train_mask = train_ready["date"] < "2023-01-01"
    test_mask = train_ready["date"] >= "2023-01-01"

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # Regularized Random Forest to prevent overfitting on panel identifiers
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=4,
        min_samples_leaf=10,
        random_state=42,
    )
    rf.fit(X_train, y_train)

    # Evaluate
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    print("\n=== Model Evaluation ===")
    print(f"Train R²: {train_r2:.4f} | Train MAE: {train_mae:.4f}")
    print(f"Test R²:  {test_r2:.4f} | Test MAE:  {test_mae:.4f}")

    # Save Model Weights (via pickle)
    model_path = DATA_DIR / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(rf, f)
    print(f"Model saved to: {model_path}")

    # Compile & Save Metrics/Feature Importances
    importances = list(rf.feature_importances_)
    feature_importances = sorted(
        [{"feature": f, "importance": float(imp)} for f, imp in zip(feature_cols, importances)],
        key=lambda x: x["importance"],
        reverse=True,
    )

    metrics = {
        "train_r2": float(train_r2),
        "train_mae": float(train_mae),
        "test_r2": float(test_r2),
        "test_mae": float(test_mae),
        "feature_importances": feature_importances,
    }

    metrics_path = DATA_DIR / "model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Model metrics saved to: {metrics_path}")

    # Generate and save Feature Importance Chart
    CHARTS_DIR.mkdir(exist_ok=True)
    imp_df = pd.DataFrame(feature_importances)
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.barplot(data=imp_df, y="feature", x="importance", palette="viridis")
    plt.title("Random Forest Regressor: Feature Importances")
    plt.xlabel("Importance Score")
    plt.ylabel("Engineered Feature")
    plt.tight_layout()
    chart_path = CHARTS_DIR / "09_ml_feature_importances.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Saved feature importances chart to: {chart_path}")


if __name__ == "__main__":
    main()
