import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Fix imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.db_config import engine
from scripts.multi_step_predict import predict_next_n_steps

st.set_page_config(layout="wide")
st.title("🔮 AI Forecasting")

# Model + Symbol Selector
model_selected = st.radio("🧠 Choose Model", ["LSTM", "XGBoost", "HYBRID"], horizontal=True)
symbol_selected = st.selectbox("📌 Select Cryptocurrency", ["BTCUSDT", "ETHUSDT", "BNBUSDT"], index=0)
forecast_horizon = st.slider("⏱️ Forecast next N steps:", 1, 10, 5)

if "forecast_df" not in st.session_state:
    st.session_state.forecast_df = {}

if st.button("📈 Predict & Save"):
    models = ["LSTM", "XGBOOST", "HYBRID"]
    for model in models:
        try:
            prediction, lower, upper, actuals = predict_next_n_steps(
                symbol_selected, steps_ahead=forecast_horizon, model_type=model
            )
            df = pd.DataFrame({
                "Step": [f"T+{i+1}" for i in range(forecast_horizon)],
                "Predicted": prediction,
                "Lower": lower,
                "Upper": upper
            })
            st.session_state.forecast_df[model] = df
            if model == model_selected.upper():
                st.session_state.rmse = round(np.sqrt(mean_squared_error(actuals[:len(prediction)], prediction)), 2)
                st.session_state.mae = round(mean_absolute_error(actuals[:len(prediction)], prediction), 2)
        except Exception as e:
            st.error(f"⚠️ Error in {model}: {e}")

# --- Forecast Display ---
if model_selected.upper() in st.session_state.forecast_df:
    selected_df = st.session_state.forecast_df[model_selected.upper()]
    
    tab1, tab2 = st.tabs(["📊 Forecast Table", "📈 Forecast Plot"])
    
    with tab1:
        st.dataframe(selected_df)

    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=selected_df["Step"], y=selected_df["Predicted"], name="Prediction", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=selected_df["Step"], y=selected_df["Upper"], mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=selected_df["Step"], y=selected_df["Lower"], mode='lines', fill='tonexty',
                                 fillcolor='rgba(0,100,80,0.2)', line=dict(width=0), name="Confidence Band"))
        fig.update_layout(title=f"{symbol_selected} - {model_selected} Forecast",
                          xaxis_title="Step", yaxis_title="Predicted Price")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"📊 **RMSE**: `{st.session_state.rmse}` | **MAE**: `{st.session_state.mae}`")

# --- Model Comparison ---
if len(st.session_state.forecast_df) == 3:
    st.subheader("📊 Compare LSTM vs XGBoost vs HYBRID")
    fig = go.Figure()
    for model in ["LSTM", "XGBOOST", "HYBRID"]:
        df = st.session_state.forecast_df.get(model)
        if df is not None:
            fig.add_trace(go.Scatter(x=df["Step"], y=df["Predicted"], mode="lines+markers", name=model))
    fig.update_layout(title="🔍 Forecast Comparison", xaxis_title="Step", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)
