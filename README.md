
# 🤖 CrAIpto | AI-Powered Crypto Forecasting & Analytics Dashboard

**CrAIpto** is an end-to-end AI-powered platform that integrates real-time cryptocurrency data ingestion, forecasting using LSTM and XGBoost models, hybrid model blending, trading simulation, model evaluation, data drift detection, and explainability — all deployed through a fully interactive Streamlit dashboard.

---

## 🚀 Key Features

| Module | Description |
|--------|-------------|
| **📊 Real-Time Data Pipeline** | Collects live cryptocurrency prices and OHLCV data using the Binance US API and stores it in a PostgreSQL database for persistent storage and analysis. |
| **🧠 Forecasting Models** | Includes both LSTM (deep learning) and XGBoost (ensemble learning) models for future price prediction, designed for flexibility and accuracy in short-term forecasting. |
| **⚡ Hybrid Model (Auto-Tuned)** | Combines predictions from LSTM and XGBoost based on recent backtest RMSE scores using dynamic alpha weighting for robust ensemble output. |
| **📉 Model Evaluation & Backtesting** | Computes RMSE, MAE, and generates historical backtest visualizations. Allows easy performance tracking across multiple models and assets. |
| **📬 Smart Auto-Trading Simulation** | Implements trading logic that simulates buy/sell decisions based on prediction confidence bands, showcasing potential trading profitability. |
| **📡 Drift Detection & Monitoring** | Tracks market shifts using Jensen-Shannon Divergence and KS statistics. Highlights deviations in distribution and triggers model retraining. |
| **🔁 Automated Model Retraining** | When drift exceeds defined thresholds, models are automatically retrained using the latest data to ensure continued accuracy and relevance. |
| **📊 Model Explainability** | Uses SHAP (SHapley Additive exPlanations) to provide insights into XGBoost model predictions, making black-box models interpretable. |
| **📈 Interactive Charts & Indicators** | Displays live charts with candlesticks, SMA, EMA, RSI, MACD. Includes overlays of forecasts, confidence bands, and buy/sell markers. |
| **🏆 Model Leaderboard (Multi-Crypto)** | Displays comparative RMSE and MAE scores for different models (LSTM, XGBoost) across selected cryptocurrencies (BTC, ETH, BNB). |
| **🧩 Multi-Page Dashboard with Session State** | Intuitive UI using Streamlit’s session state to retain results, auto-refresh toggles, and interactive tabs for forecast, backtest, and trading. |

---

## 📁 Project Structure

```
cryptoTest/
├── app.py                         
├── config/
│   └── db_config.py               
├── models/                        
├── outputs/
│   └── model_leaderboard.csv     
├── pages/                         
│   ├── 1_📈_Live_Chart.py
│   ├── 2_🔮_Forecast.py
│   ├── 3_📉_Backtest.py
│   ├── 4_🏆_Leaderboard.py
│   ├── 5_⚠️_Drift_Detection.py
│   └── 6_🧠_Model_Explainability.py
├── scripts/                       
│   ├── fetch_crypto_prices.py
│   ├── fetch_historical_ohlcv.py
│   ├── train_lstm_model.py
│   ├── train_xgb_model.py
│   ├── evaluate_model.py
│   ├── multi_step_predict.py
│   ├── backtest_model.py
│   ├── simulate_trading.py
│   ├── data_drift_detector.py
│   ├── drift_based_retrainer.py
│   └── explain_xgb_model.py
├── README.md
└── requirements.txt
```

---

## 🧠 Forecasting Models: LSTM, XGBoost & HYBRID

- **LSTM**: Captures sequential patterns from historical time-series prices using deep learning. Best for capturing temporal dependencies.
- **XGBoost**: Tree-based boosting algorithm, great for modeling nonlinear relationships in engineered lag-based features.
- **HYBRID**: Dynamically weighted ensemble of LSTM and XGBoost, where weights are auto-adjusted based on historical model RMSEs.

---

## 📉 Evaluation Metrics & Backtesting

- **Metrics**: RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error).
- **Backtesting**: Compares predicted vs actuals on historical windows.
- **Charts**: Grouped bar charts for RMSE & MAE by model; visualized in leaderboard and evaluation sections.

---

## 📡 Data Drift Detection & Auto-Retraining

- **Statistical Tests**: 
  - **KS Test**: Distributional similarity test.
  - **Jensen-Shannon Divergence**: Measures divergence between baseline and recent distributions.
- **Trigger**: When JS Divergence > 0.2 or KS p-value < 0.05.
- **Action**: Triggers automatic retraining of both LSTM and XGBoost models using the most recent dataset.

---

## 🧠 Explainability with SHAP

- Focuses on **XGBoost** predictions.
- Shows top lag-based features with their SHAP impact scores.
- Includes SHAP bar plots and summary charts.

---

## 💹 Auto-Trading Simulation

- Based on:
  - Predicted price > upper band → Simulate Buy
  - Predicted price < lower band → Simulate Sell
- Tracks portfolio value, net P&L, and number of trades executed.

---

## 🖥️ Streamlit Dashboard Highlights

- Modular multi-page app for easy navigation.
- Session state tracks user choices, auto-refresh settings, cached results.
- Candlestick visualizations with overlays of indicators and AI forecasts.
- Forecast selector: LSTM vs XGBoost vs HYBRID
- Real-time price monitoring + Auto-refresh toggles
- Leaderboard and explainability embedded in UI

---

## 🔐 Setup Instructions

Update `config/db_config.py` with your credentials:

```python
BINANCE_API_KEY = "your_api_key_here"
POSTGRES_URI = "postgresql://user:password@localhost:5432/crypto_db"
```

Ensure PostgreSQL is running and the tables are created.

---

## ▶️ Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Start the dashboard
streamlit run app.py
```

---

## 📦 Requirements

- Python 3.9+
- TensorFlow / Keras
- XGBoost
- SHAP
- SQLAlchemy
- PostgreSQL
- Plotly
- Streamlit
- Pandas / NumPy
- scikit-learn

---

## 🧪 Supported Cryptocurrencies

- **BTCUSDT** ✅
- **ETHUSDT** ✅
- **BNBUSDT** ✅

More can be added by running training scripts on new symbols.

---

![alt text](image.png)

