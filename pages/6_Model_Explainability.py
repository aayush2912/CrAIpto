import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.explain_xgb_model import explain_xgboost_model

st.set_page_config(layout="wide")
st.title("🧠 Model Explainability")

st.markdown("🔍 **Understand model predictions using SHAP (XGBoost only)**")

symbol = st.selectbox("📌 Select Symbol", ["BTCUSDT", "ETHUSDT", "BNBUSDT"], index=0)

try:
    shap_values, X = explain_xgboost_model(symbol)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🔝 Top 10 Feature Importances")
        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Mean SHAP Value": np.abs(shap_values.values).mean(axis=0)
        }).sort_values(by="Mean SHAP Value", ascending=False)
        st.dataframe(importance_df.head(10), use_container_width=True)

    with col2:
        st.subheader("📊 SHAP Summary Plot (Bar)")
        fig_summary, ax = plt.subplots()
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        st.pyplot(fig_summary, use_container_width=True)

except FileNotFoundError:
    st.error(f"❌ No XGBoost model found for {symbol}. Please train it first.")
except Exception as e:
    st.error(f"⚠️ Error: {e}")
