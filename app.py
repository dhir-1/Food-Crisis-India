import pandas as pd
import plotly.express as px
import streamlit as st

from project_utils import CHARTS_DIR, load_master_dataframe, latest_state_ranking


st.set_page_config(
    page_title="FoodGuard India",
    page_icon="??",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    return load_master_dataframe()


def main() -> None:
    master = load_data()
    ranking = latest_state_ranking(master)

    st.title("FoodGuard India")
    st.caption("State-level food distress monitoring and early warning prototype")

    with st.sidebar:
        st.header("Filters")
        states = sorted(master["state"].dropna().unique().tolist())
        selected_states = st.multiselect("States", states, default=states[:8])
        min_date = master["date"].min().date()
        max_date = master["date"].max().date()
        start_date, end_date = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    filtered = master.copy()
    if selected_states:
        filtered = filtered[filtered["state"].isin(selected_states)]
    filtered = filtered[(filtered["date"].dt.date >= start_date) & (filtered["date"].dt.date <= end_date)]

    latest_filtered = ranking[ranking["state"].isin(filtered["state"].unique())].copy()
    latest_top_state = latest_filtered.iloc[0]["state"] if not latest_filtered.empty else "N/A"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("States In View", filtered["state"].nunique())
    col2.metric("Rows", f"{len(filtered):,}")
    col3.metric("Average Food Price", f"{filtered['avg_food_price'].mean():.2f}")
    col4.metric("Highest Current Risk", latest_top_state)

    overview_tab, state_tab, ranking_tab, charts_tab = st.tabs(
        ["Overview", "State Explorer", "Risk Ranking", "Chart Gallery"]
    )

    with overview_tab:
        trend = (
            filtered.groupby("date", as_index=False)[["avg_food_price", "rainfall_mm", "risk_score"]]
            .mean()
        )
        price_fig = px.line(
            trend,
            x="date",
            y="avg_food_price",
            title="Average Food Price Trend",
            labels={"avg_food_price": "Average Food Price", "date": "Date"},
        )
        st.plotly_chart(price_fig, use_container_width=True)

        scatter_fig = px.scatter(
            filtered.dropna(subset=["rainfall_mm", "yield_ton_per_hectare"]),
            x="rainfall_mm",
            y="yield_ton_per_hectare",
            color="state",
            title="Rainfall vs Crop Yield",
            labels={"rainfall_mm": "Rainfall (mm)", "yield_ton_per_hectare": "Yield (ton/hectare)"},
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

    with state_tab:
        chosen_state = st.selectbox("Choose one state", sorted(filtered["state"].unique().tolist()))
        state_df = filtered[filtered["state"] == chosen_state].copy()

        st.subheader(f"{chosen_state} Snapshot")
        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("Latest Risk Score", f"{state_df.iloc[-1]['risk_score']:.2f}")
        metric_b.metric("Latest Avg Food Price", f"{state_df.iloc[-1]['avg_food_price']:.2f}")
        metric_c.metric("Latest Rainfall", f"{state_df.iloc[-1]['rainfall_mm']:.2f}" if pd.notna(state_df.iloc[-1]["rainfall_mm"]) else "N/A")

        state_price = px.line(
            state_df,
            x="date",
            y=["rice_price", "wheat_price", "dal_price", "oil_price"],
            title=f"{chosen_state}: Commodity Price Trends",
        )
        st.plotly_chart(state_price, use_container_width=True)

        state_risk = px.line(
            state_df,
            x="date",
            y="risk_score",
            title=f"{chosen_state}: Exploratory Risk Score",
            labels={"risk_score": "Risk Score", "date": "Date"},
        )
        st.plotly_chart(state_risk, use_container_width=True)

    with ranking_tab:
        st.subheader("Latest State Ranking")
        top_n = st.slider("Top states to show", min_value=5, max_value=20, value=10)
        latest_rank_subset = latest_filtered.head(top_n)
        rank_fig = px.bar(
            latest_rank_subset,
            x="risk_score",
            y="state",
            color="risk_band",
            orientation="h",
            title="Current Food Distress Risk Ranking",
            category_orders={"risk_band": ["High", "Medium", "Low"]},
        )
        rank_fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(rank_fig, use_container_width=True)
        st.dataframe(
            latest_filtered[["state", "date", "avg_food_price", "rainfall_mm", "yield_ton_per_hectare", "risk_score", "risk_band"]],
            use_container_width=True,
        )

    with charts_tab:
        st.subheader("Generated Chart Files")
        chart_files = sorted(CHARTS_DIR.glob("*.png"))
        for chart_path in chart_files:
            st.image(str(chart_path), caption=chart_path.name, use_container_width=True)


if __name__ == "__main__":
    main()
