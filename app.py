import streamlit as st
from PIL import Image

st.set_page_config(page_title="CrAIpto | AI-Powered Crypto Dashboard", layout="wide")

# --- Title + Banner ---
st.title("🤖 CrAIpto | AI-Powered Crypto Dashboard")
st.markdown("Welcome to your **Multi-Page AI-Powered Cryptocurrency Analysis System**.")

st.markdown("""
This dashboard uses AI models like **LSTM**, **XGBoost**, and **HYBRID** to forecast crypto prices,
simulate trades, detect data drift, and more — all with real-time insights. 🧠📊

**Navigation is available on the left sidebar 👉**
""")

# --- Dashboard Overview ---
st.subheader("📂 Available Pages")
st.markdown("""
| Page                         | Description                                                  |
|------------------------------|--------------------------------------------------------------|
| 🔮 Forecasting               | Predict future prices using AI models (LSTM/XGBoost/Hybrid) |
| 📉 Backtesting               | Evaluate model performance with RMSE and MAE                |
| 🤖 Auto-Trading Simulator    | Simulate smart Buy/Sell strategies                          |
| ⚠️ Drift Detection           | Detect data drift and retrain models                        |
| 🧠 Explainability            | Visualize model logic using SHAP for XGBoost                |
| 🏆 Leaderboard               | Multi-crypto model comparison leaderboard                   |
""")

# --- Quick Instructions ---
with st.expander("🛠️ Getting Started (Click to Expand)"):
    st.markdown("""
1. Select a page from the **sidebar**.
2. Choose a cryptocurrency symbol like `BTCUSDT`, `ETHUSDT`, or `BNBUSDT`.
3. Explore forecasting, simulation, backtesting, explainability, and more.
4. Use `Predict & Save` to generate forecasts.
5. Try `Run Backtest` to see model performance.
6. Use `Auto-Retrain` if data drift is detected.

All models and visualizations are stored inside:
- `/models/` → Saved LSTM & XGBoost models
- `/outputs/` → Forecasts, metrics, and leaderboard

Happy forecasting! 🚀
""")

image = Image.open("image.png")
st.image(image, use_column_width=True)

