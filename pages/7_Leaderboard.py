import os
import sys
import streamlit as st
import pandas as pd
import plotly.graph_objs as go

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(layout="wide")
st.title("🏆 Model Performance Leaderboard")

st.markdown("📊 Compare model performance across all cryptos using RMSE & MAE.")

leaderboard_file = "outputs/model_leaderboard.csv"

try:
    df = pd.read_csv(leaderboard_file)

    tab1, tab2 = st.tabs(["📋 Leaderboard Table", "📊 RMSE Comparison"])

    with tab1:
        st.dataframe(df.style.highlight_min(subset=["RMSE", "MAE"], color="lightgreen", axis=0))

    with tab2:
        fig = go.Figure()
        for model in df["Model"].unique():
            model_df = df[df["Model"] == model]
            fig.add_trace(go.Bar(
                x=model_df["Symbol"],
                y=model_df["RMSE"],
                name=f"{model} - RMSE"
            ))
        fig.update_layout(
            barmode="group",
            title="📉 RMSE by Model & Symbol",
            xaxis_title="Symbol",
            yaxis_title="RMSE"
        )
        st.plotly_chart(fig, use_container_width=True)

except FileNotFoundError:
    st.error("🚫 Leaderboard file not found. Please run the backtest first.")
except Exception as e:
    st.error(f"⚠️ Error loading leaderboard: {e}")
