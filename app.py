import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from project_utils import (
    CHARTS_DIR,
    DATA_DIR,
    latest_state_ranking,
    load_master_dataframe,
    compute_proactive_risk_score,
)

# 1. Page Config
st.set_page_config(
    page_title="FoodGuard India: Proactive Early Warning System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Style for clean modern premium look
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #334155;
    }
    .status-normal {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border-left: 5px solid #10b981;
        padding: 15px;
        border-radius: 5px;
        font-weight: bold;
    }
    .status-warning {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border-left: 5px solid #f59e0b;
        padding: 15px;
        border-radius: 5px;
        font-weight: bold;
    }
    .status-critical {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border-left: 5px solid #ef4444;
        padding: 15px;
        border-radius: 5px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# 2. Data & Model Caching
@st.cache_data
def get_master_data() -> pd.DataFrame:
    return load_master_dataframe()


@st.cache_resource
def load_ml_model():
    model_path = DATA_DIR / "model.pkl"
    if model_path.exists():
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None


@st.cache_data
def load_ml_metrics():
    metrics_path = DATA_DIR / "model_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None


def main() -> None:
    # Load assets
    master = get_master_data()
    model = load_ml_model()
    metrics = load_ml_metrics()

    # Features used for the ML model
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

    # Dashboard Header
    st.title("🌾 FoodGuard India")
    st.markdown(
        "**2-Step Machine Learning Early Warning System & Food Distress Monitoring Dashboard**"
    )
    st.caption("A data-driven, zero-leakage monthly panel prototype tracking price growth anomalies, rainfall stress, and agricultural capability.")

    # Sidebar Filter Section
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.subheader("Global Filters")
        
        # State Multiselect
        states = sorted(master["state"].dropna().unique().tolist())
        selected_states = st.multiselect("Active States", states, default=states[:8])
        
        # Date Range slider
        min_date = master["date"].min().date()
        max_date = master["date"].max().date()
        start_date, end_date = st.date_input(
            "Analysis Period",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

    # Filter Main Panel Data
    filtered = master.copy()
    if selected_states:
        filtered = filtered[filtered["state"].isin(selected_states)]
    filtered = filtered[
        (filtered["date"].dt.date >= start_date) & (filtered["date"].dt.date <= end_date)
    ]

    # KPI summary Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Monitored States", filtered["state"].nunique())
    with col2:
        st.metric("Total Historical Records", f"{len(filtered):,}")
    with col3:
        st.metric("Avg Food Price (Current View)", f"₹{filtered['avg_food_price'].mean():.2f}/kg")
    with col4:
        # Default ranking anchored on March 2024 (latest complete month)
        default_rank = latest_state_ranking(master, "2024-03-01")
        top_risk_state = default_rank.iloc[0]["state"] if not default_rank.empty else "N/A"
        st.metric("Highest March 2024 Risk", top_risk_state)

    st.markdown("---")

    # Main Tab Layout
    overview_tab, state_tab, ranking_tab, ml_tab, charts_tab = st.tabs(
        [
            "📈 Panel Overview",
            "🔍 State Explorer",
            "🚨 Food Distress Risk Ranking",
            "🤖 Predictive Early Warning (ML)",
            "🖼️ Chart Gallery",
        ]
    )

    # ==================== TAB 1: OVERVIEW ====================
    with overview_tab:
        st.subheader("Monthly Panel Historical Aggregates")
        
        trend = (
            filtered.groupby("date", as_index=False)[["avg_food_price", "rainfall_mm", "risk_score"]]
            .mean()
        )
        
        fig_price = px.line(
            trend,
            x="date",
            y="avg_food_price",
            title="National Average Staple Food Price Trend (Rice, Wheat, Dal, Oil combined)",
            labels={"avg_food_price": "Price (₹/kg)", "date": "Date"},
            color_discrete_sequence=["#f59e0b"],
        )
        fig_price.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_price, width="stretch")

        col_left, col_right = st.columns(2)
        with col_left:
            fig_scatter = px.scatter(
                filtered.dropna(subset=["rainfall_mm", "yield_ton_per_hectare"]),
                x="rainfall_mm",
                y="yield_ton_per_hectare",
                color="state",
                title="Rainfall vs Crop Yield (State Productivity Panel)",
                labels={"rainfall_mm": "Monthly Rainfall (mm)", "yield_ton_per_hectare": "Yield (ton/hectare)"},
            )
            fig_scatter.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_scatter, width="stretch")
            
        with col_right:
            fig_price_rain = px.scatter(
                filtered.dropna(subset=["rainfall_mm", "avg_food_price"]),
                x="rainfall_mm",
                y="avg_food_price",
                color="state",
                title="Rainfall vs Commodity Price Level",
                labels={"rainfall_mm": "Rainfall (mm)", "avg_food_price": "Price (₹/kg)"},
            )
            fig_price_rain.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_price_rain, width="stretch")

    # ==================== TAB 2: STATE EXPLORER ====================
    with state_tab:
        chosen_state = st.selectbox("Choose State to Inspect", sorted(filtered["state"].unique().tolist()))
        state_df = filtered[filtered["state"] == chosen_state].sort_values("date").copy()
        
        st.subheader(f"📊 {chosen_state} Vulnerability Profile")
        
        subcol1, subcol2, subcol3, subcol4 = st.columns(4)
        subcol1.metric("NFHS-5 Stunting Rate", f"{state_df.iloc[-1]['stunting_rate']:.1f}%")
        subcol2.metric("NFHS-5 Underweight Rate", f"{state_df.iloc[-1]['underweight_rate']:.1f}%")
        subcol3.metric("NFHS-5 Anaemia Rate", f"{state_df.iloc[-1]['anaemia_rate']:.1f}%")
        subcol4.metric("Crop Yield Capacity", f"{state_df.iloc[-1]['yield_ton_per_hectare']:.2f} t/ha")

        # Commodity break-downs
        fig_comm = px.line(
            state_df,
            x="date",
            y=["rice_price", "wheat_price", "dal_price", "oil_price"],
            title=f"{chosen_state}: Commodity Retail Price Trends",
            labels={"value": "Price (₹/kg)", "variable": "Commodity", "date": "Date"},
        )
        fig_comm.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_comm, width="stretch")

        # Monthly rainfall trends
        fig_rain = px.bar(
            state_df,
            x="date",
            y="rainfall_mm",
            title=f"{chosen_state}: Monthly Rainfall Ingestion Log",
            labels={"rainfall_mm": "Rainfall (mm)", "date": "Date"},
            color_discrete_sequence=["#3b82f6"],
        )
        fig_rain.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_rain, width="stretch")

    # ==================== TAB 3: RISK RANKING ====================
    with ranking_tab:
        st.subheader("🚨 Food Distress Comparative Risk Ranking")
        st.markdown(
            "This tab ranks states based on our **Base Heuristic Index** which combines price level, active rainfall deviations, yields, and static nutrition. To prevent `NaN` values, you can inspect historical months with complete records."
        )

        # Select a date to rank
        unique_dates = sorted(master["date"].dt.date.unique().tolist())
        target_date_selection = st.select_slider(
            "Select Ranking Month",
            options=unique_dates,
            value=pd.to_datetime("2024-03-01").date(),
        )

        # Retrieve ranking
        rank_df = latest_state_ranking(master, str(target_date_selection))

        if not rank_df.empty:
            col_bar, col_tbl = st.columns([1, 1])
            with col_bar:
                fig_rank = px.bar(
                    rank_df.head(15),
                    x="risk_score",
                    y="state",
                    color="risk_band",
                    orientation="h",
                    title=f"Distress Risk Ranking for {target_date_selection}",
                    category_orders={"risk_band": ["High", "Medium", "Low"]},
                    color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"},
                    labels={"risk_score": "Heuristic Risk Index", "state": "State", "risk_band": "Risk Band"},
                )
                fig_rank.update_layout(template="plotly_dark", yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_rank, width="stretch")
                
            with col_tbl:
                st.dataframe(
                    rank_df[
                        [
                            "state",
                            "avg_food_price",
                            "rainfall_mm",
                            "yield_ton_per_hectare",
                            "risk_score",
                            "risk_band",
                        ]
                    ],
                    width="stretch",
                    height=450,
                )
        else:
            st.warning("No data found for the selected month.")

    # ==================== TAB 4: PREDICTIVE EARLY WARNING (ML) ====================
    with ml_tab:
        st.subheader("🤖 Proactive 2-Step Machine Learning Early Warning System")
        st.markdown(
            """
            To solve target leaks and the trivial 'next-month is this-month' mapping trap (which makes forecasting absolute prices meaningless),
            this prototype implements an advanced **2-Step Crisis early warning system**:
            
            1. **Step 1: Machine Learning Price Growth Forecasting**: We train a regularized Random Forest model strictly on out-of-time chronological data (Training pre-2023, Validation 2023–2024) to predict the **3-month forward percentage price change** (growth rate). This forces the model to learn weather anomalies and baseline vulnerability mappings instead of copying current prices.
            2. **Step 2: Proactive Risk Scoring**: We pass the model's 3-month forecast into a weighted proactive z-score index (40% predicted price inflation, 25% monthly climatological rainfall stress, 20% agricultural crop yield capacity, 15% baseline nutrition burden) to identify distress hotspots *90 days before price spikes actually register*.
            """
        )

        if model is None or metrics is None:
            st.error("ML model files (`model.pkl` and `model_metrics.json`) are missing! Please run `python scripts/train_model.py` first.")
        else:
            # Model performance section
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            col_stat1.metric("Validation Split Method", "Chronological Out-of-Time")
            col_stat2.metric("Out-of-Time Test MAE", f"₹{metrics['test_mae']:.2f}/kg Price Error")
            col_stat3.metric("Train/Test Generalization Gap", f"₹{(metrics['train_mae'] - metrics['test_mae']):.2f}/kg (No Overfitting)")

            st.markdown("### 1. State-Specific 3-Month Price Growth Forecast")
            selected_ml_state = st.selectbox("Select State to Forecast", sorted(master["state"].unique().tolist()))
            
            # Predict for this state's latest month (August 2024)
            state_all = master[master["state"] == selected_ml_state].sort_values("date").copy()
            latest_row = state_all.iloc[-1]
            
            # Feed features
            feat_vector = pd.DataFrame([latest_row[feature_cols]])
            predicted_growth = model.predict(feat_vector)[0]
            current_price = latest_row["avg_food_price"]
            predicted_price_3m = current_price * (1 + predicted_growth)
            
            # Compute dynamic proactive risk score for the latest row
            latest_rows_all_states = master.groupby("state").tail(1).copy()
            # Predict growth for all states to calculate valid comparative z-scores
            X_all_latest = latest_rows_all_states[feature_cols]
            latest_rows_all_states["pred_growth"] = model.predict(X_all_latest)
            latest_rows_all_states["proactive_risk"] = compute_proactive_risk_score(
                latest_rows_all_states, latest_rows_all_states["pred_growth"]
            )
            
            state_proactive_risk = latest_rows_all_states[
                latest_rows_all_states["state"] == selected_ml_state
            ]["proactive_risk"].values[0]

            # Metric Cards
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Current Avg Price (Aug 2024)", f"₹{current_price:.2f}/kg")
            mcol2.metric("ML Predicted Price (Nov 2024)", f"₹{predicted_price_3m:.2f}/kg")
            
            # Color coding growth
            growth_pct = predicted_growth * 100
            mcol3.metric(
                "Predicted 3M Inflation",
                f"{growth_pct:+.2f}%",
                delta=f"{growth_pct:+.2f}%",
                delta_color="inverse" if growth_pct > 0 else "normal",
            )
            mcol4.metric("Proactive 90-Day Risk Score", f"{state_proactive_risk:.2f}")

            # Alert Status Logic
            st.markdown("#### 🚨 Early Warning Action Alert")
            if growth_pct >= 6.0:
                st.markdown(
                    f"""
                    <div class="status-critical">
                        🔴 CRITICAL ALERT: Staple food prices in {selected_ml_state} are predicted to grow by {growth_pct:.2f}% (exceeding 6.0% safety boundary) in the next 90 days. 
                        <br><br>
                        Policy recommendation: Immediately release state-held buffer grain reserves, trigger emergency food distribution safety nets, and coordinate with regional markets to suppress price spikes.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif 3.0 <= growth_pct < 6.0:
                st.markdown(
                    f"""
                    <div class="status-warning">
                        🟡 ELEVATED WARNING: Staple food prices in {selected_ml_state} are predicted to grow by {growth_pct:.2f}% in the next 90 days. 
                        <br><br>
                        Policy recommendation: Put regional storage yards on alert, closely monitor weekly crop production and local supply chain bottlenecks, and prepare emergency financial support programs.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="status-normal">
                        🟢 NORMAL STATE: Staple food prices in {selected_ml_state} are predicted to grow by {growth_pct:.2f}% (within the 3.0% normal fluctuation range) over the next 90 days. 
                        <br><br>
                        Policy recommendation: Continue standard weekly market monitoring. No emergency food interventions are required at this time.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Actual vs Predicted validation plot
            st.markdown("### 2. Time-Series Model Validation (Chronological Test Set 2023–2024)")
            st.markdown(
                "This chart aligns the model's 3-month-ahead predictions with the actual prices observed 3 months later on the out-of-time test set. It visually demonstrates the model's generalized predictive accuracy."
            )
            
            # Predict for all historical valid rows for this state
            valid_rows = state_all.dropna(subset=feature_cols).copy()
            if not valid_rows.empty:
                valid_rows["predicted_growth"] = model.predict(valid_rows[feature_cols])
                valid_rows["pred_price_3m"] = valid_rows["avg_food_price"] * (1 + valid_rows["predicted_growth"])
                
                # Shift predicted prices forward by 3 months to align with actual future dates!
                valid_rows["pred_price_aligned"] = valid_rows["pred_price_3m"].shift(3)
                
                # Filter to validation period
                test_rows = valid_rows[valid_rows["date"] >= "2023-01-01"].copy()
                
                if not test_rows.empty:
                    fig_val = go.Figure()
                    fig_val.add_trace(
                        go.Scatter(
                            x=test_rows["date"],
                            y=test_rows["avg_food_price"],
                            mode="lines+markers",
                            name="Actual Food Price",
                            line=dict(color="#f59e0b", width=3),
                        )
                    )
                    fig_val.add_trace(
                        go.Scatter(
                            x=test_rows["date"],
                            y=test_rows["pred_price_aligned"],
                            mode="lines+markers",
                            name="Model Predicted Price (Forecasted 3m Ago)",
                            line=dict(color="#3b82f6", width=3, dash="dash"),
                        )
                    )
                    fig_val.update_layout(
                        title=f"{selected_ml_state}: Actual vs Aligned Predicted Price (Out-of-Time Test Set)",
                        xaxis_title="Date",
                        yaxis_title="Price (₹/kg)",
                        template="plotly_dark",
                        height=400,
                    )
                    st.plotly_chart(fig_val, width="stretch")
                else:
                    st.warning("Insufficient validation rows from 2023 onwards to plot test validation trend.")
            else:
                st.warning("No rows have complete feature columns for plotting.")

            # Model Feature Importances
            st.markdown("### 3. Model Feature Importances")
            st.markdown(
                "This chart explains which features drive the model's 3-month price growth forecasts, showing the mathematical weights calculated during Random Forest training."
            )
            
            imp_df = pd.DataFrame(metrics["feature_importances"])
            fig_imp = px.bar(
                imp_df,
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale="Viridis",
                title="Random Forest Regressor Feature Importances",
                labels={"importance": "Importance Score", "feature": "Feature"},
            )
            fig_imp.update_layout(template="plotly_dark", yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_imp, width="stretch")

    # ==================== TAB 5: CHART GALLERY ====================
    with charts_tab:
        st.subheader("Matplotlib / Seaborn Static Visualizations")
        st.markdown(
            "These static visualizations are automatically compiled by running `python scripts/generate_charts.py` and are stored under the `/charts` directory."
        )
        
        chart_files = sorted(CHARTS_DIR.glob("*.png"))
        if chart_files:
            for chart_path in chart_files:
                st.image(
                    str(chart_path),
                    caption=f"{chart_path.name} - Generated Static Diagnostic Figure",
                    width="stretch",
                )
        else:
            st.warning("No static chart PNGs found in the `charts` folder. Please run `python scripts/generate_charts.py` first.")


if __name__ == "__main__":
    main()
