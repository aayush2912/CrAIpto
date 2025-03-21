import os
import sys
import streamlit as st
import pandas as pd
import plotly.graph_objs as go

# Fix imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.backtest_model import backtest

st.set_page_config(layout="wide")
st.title("📉 Model Backtesting")

symbol = st.selectbox("📌 Select Symbol for Backtest", ["BTCUSDT", "ETHUSDT", "BNBUSDT"], index=0)

if "backtest_metrics" not in st.session_state:
    st.session_state.backtest_metrics = None

if st.button("🔁 Run Backtest for Selected Symbol"):
    try:
        rmse_lstm, mae_lstm, _, _ = backtest(symbol, model_type="LSTM")
        rmse_xgb, mae_xgb, _, _ = backtest(symbol, model_type="XGBOOST")
        st.session_state.backtest_metrics = pd.DataFrame({
            "Model": ["LSTM", "XGBoost"],
            "RMSE": [rmse_lstm, rmse_xgb],
            "MAE": [mae_lstm, mae_xgb]
        })
    except Exception as e:
        st.error(f"⚠️ Backtest error: {e}")

if st.session_state.backtest_metrics is not None:
    tab1, tab2 = st.tabs(["📋 Metrics", "📊 Comparison"])

    with tab1:
        st.dataframe(st.session_state.backtest_metrics)

    with tab2:
        fig = go.Figure(data=[
            go.Bar(name='RMSE', x=st.session_state.backtest_metrics["Model"], y=st.session_state.backtest_metrics["RMSE"]),
            go.Bar(name='MAE', x=st.session_state.backtest_metrics["Model"], y=st.session_state.backtest_metrics["MAE"])
        ])
        fig.update_layout(barmode='group', title=f"Backtest: LSTM vs XGBoost ({symbol})",
                          yaxis_title="Error", xaxis_title="Model")
        st.plotly_chart(fig, use_container_width=True)
