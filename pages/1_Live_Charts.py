import os
import sys
import streamlit as st
import pandas as pd
import plotly.graph_objs as go

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.db_config import engine

st.set_page_config(layout="wide")
st.title("📈 Live Crypto Candlestick Charts")

symbol = st.selectbox("Choose Symbol", ["BTCUSDT", "ETHUSDT", "BNBUSDT"])

@st.cache_data(ttl=30)
def get_candle_data(symbol):
    query = f"""
    SELECT timestamp, open, high, low, close FROM crypto_ohlcv
    WHERE symbol = '{symbol}'
    ORDER BY timestamp DESC
    LIMIT 100;
    """
    df = pd.read_sql(query, engine)
    return df[::-1]  # oldest to newest

df = get_candle_data(symbol)

if not df.empty:
    fig = go.Figure(data=[go.Candlestick(
        x=df["timestamp"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Candlestick"
    )])
    
    fig.update_layout(
        title=f"{symbol} - Live Candlestick Chart",
        xaxis_title="Timestamp",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No data available.")
