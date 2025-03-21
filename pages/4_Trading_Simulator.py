import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.multi_step_predict import predict_next_n_steps
from config.db_config import engine

st.set_page_config(layout="wide")
st.title("🤖 Smart Auto-Trading Simulator")

symbol = st.selectbox("📌 Select Symbol", ["BTCUSDT", "ETHUSDT", "BNBUSDT"], index=0)
initial_capital = st.number_input("💰 Starting Capital ($)", value=10000, step=500)
model_type = st.radio("🧠 Prediction Model", ["LSTM", "XGBoost", "HYBRID"], horizontal=True)
forecast_horizon = st.slider("⏱️ Forecast Steps Ahead", 1, 10, 5)

if st.button("🚀 Simulate Trades"):
    try:
        pred, lower, upper, actuals = predict_next_n_steps(symbol, forecast_horizon, model_type=model_type)
        df = pd.DataFrame({
            "Step": [f"T+{i+1}" for i in range(forecast_horizon)],
            "Predicted": pred
        })

        buys, sells = [], []
        capital = initial_capital
        holdings = 0

        for i in range(1, len(pred)):
            if pred[i] > pred[i - 1]:  # price expected to rise
                if capital > 0:
                    amount_to_buy = capital / pred[i]
                    holdings += amount_to_buy
                    capital = 0
                    buys.append((i, pred[i]))
            elif pred[i] < pred[i - 1]:  # price expected to drop
                if holdings > 0:
                    capital += holdings * pred[i]
                    holdings = 0
                    sells.append((i, pred[i]))

        final_value = capital + holdings * pred[-1]
        step_labels = [f"T+{i+1}" for i in range(len(pred))]

        st.subheader("📍 Trade Actions")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Buys", len(buys))
        col2.metric("Total Sells", len(sells))
        col3.metric("Final Portfolio Value", f"${final_value:,.2f}")

        # Chart with buy/sell markers
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=step_labels, y=pred, mode='lines+markers', name='Prediction'))

        for i, price in buys:
            fig.add_trace(go.Scatter(x=[f"T+{i+1}"], y=[price],
                                     mode="markers", marker=dict(color='green', size=10),
                                     name="Buy"))

        for i, price in sells:
            fig.add_trace(go.Scatter(x=[f"T+{i+1}"], y=[price],
                                     mode="markers", marker=dict(color='red', size=10),
                                     name="Sell"))

        fig.update_layout(title=f"Simulated Trades ({symbol} - {model_type})", xaxis_title="Step", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Simulation Failed: {e}")
