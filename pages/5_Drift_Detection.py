import os
import sys
import streamlit as st
import pandas as pd
import plotly.graph_objs as go

# Fix import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.data_drift_detector import detect_drift
from scripts.drift_based_retrainer import retrain_if_drifted

st.set_page_config(layout="wide")
st.title("⚠️ Data Drift Detection & Auto-Retraining")

symbol = st.selectbox("📌 Select Symbol for Drift Detection", ["BTCUSDT", "ETHUSDT", "BNBUSDT"], index=0)
drift_threshold = st.slider("📊 Drift Threshold (for retraining)", 0.01, 1.0, 0.3, 0.01)

if st.button("🔍 Detect Drift"):
    drift_result  = detect_drift(symbol)
    drift_score = drift_result["js_div"]
    st.session_state.drift_score = drift_score
    st.success(f"📈 Drift Score (JS Divergence) for {symbol}: `{round(drift_score, 4)}`")

if "drift_score" in st.session_state:
    st.progress(min(st.session_state.drift_score, 1.0))

    if st.session_state.drift_score > drift_threshold:
        st.warning("⚠️ Drift exceeds threshold. You may want to retrain models.")
        if st.button("♻️ Retrain Models Automatically"):
            retrain_if_drifted(symbol, threshold=drift_threshold)
            st.success("✅ Retraining complete.")
    else:
        st.info("✅ Drift is within acceptable range.")

# Optional: Display log output (if implemented)
log_file = f"logs/retrain_log_{symbol}.txt"
if os.path.exists(log_file):
    with open(log_file) as f:
        logs = f.read()
    st.subheader("📜 Retraining Log")
    st.code(logs, language="bash")
