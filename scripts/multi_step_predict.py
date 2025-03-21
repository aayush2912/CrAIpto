import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# Fix import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.db_config import engine
from scripts.data_utils import fetch_data  # ✅ External, clean source

def predict_next_n_steps(symbol="BTCUSDT", steps_ahead=5, model_type="LSTM", alpha=0.5):
    df = fetch_data(symbol)
    df = df.sort_values("timestamp")
    close_prices = df["close"].values.reshape(-1, 1)

    # -----------------------------
    # Load LSTM model & scaler
    # -----------------------------
    lstm_model = load_model(f"models/lstm_{symbol}.h5")
    lstm_scaler = np.load(f"models/scaler_{symbol}.npy", allow_pickle=True).item()
    with open(f"models/time_steps_{symbol}.txt", "r") as f:
        time_steps_lstm = int(f.read())

    scaled_lstm = lstm_scaler.transform(close_prices).flatten()
    X_input = scaled_lstm[-time_steps_lstm:]

    lstm_preds = []
    for _ in range(steps_ahead):
        x = np.reshape(X_input, (1, time_steps_lstm, 1))
        pred = lstm_model.predict(x, verbose=0)[0][0]
        lstm_preds.append(pred)
        X_input = np.append(X_input[1:], pred)

    lstm_preds = lstm_scaler.inverse_transform(np.array(lstm_preds).reshape(-1, 1)).flatten()

    if model_type.upper() == "LSTM":
        return lstm_preds, lstm_preds * 0.97, lstm_preds * 1.03, close_prices.flatten()

    # -----------------------------
    # Load XGBoost model & scaler
    # -----------------------------
    xgb_model = joblib.load(f"models/xgb_{symbol}.joblib")
    xgb_scaler = np.load(f"models/xgb_scaler_{symbol}.npy", allow_pickle=True).item()
    with open(f"models/xgb_time_steps_{symbol}.txt", "r") as f:
        time_steps_xgb = int(f.read())

    scaled_xgb = xgb_scaler.transform(close_prices).flatten()
    X_input = scaled_xgb[-time_steps_xgb:]

    xgb_preds = []
    for _ in range(steps_ahead):
        x = np.reshape(X_input, (1, time_steps_xgb))
        pred = xgb_model.predict(x)[0]
        xgb_preds.append(pred)
        X_input = np.append(X_input[1:], pred)

    xgb_preds = xgb_scaler.inverse_transform(np.array(xgb_preds).reshape(-1, 1)).flatten()

    if model_type.upper() == "XGBOOST":
        return xgb_preds, xgb_preds * 0.97, xgb_preds * 1.03, close_prices.flatten()

    # -----------------------------
    # HYBRID: Auto-adjust weights using RMSE
    # -----------------------------
    if model_type.upper() == "HYBRID":
        from scripts.backtest_model import backtest  # ✅ Local import avoids circular dependency
        rmse_lstm, _, _, _ = backtest(symbol=symbol, model_type="LSTM")
        rmse_xgb, _, _, _ = backtest(symbol=symbol, model_type="XGBOOST")

        if (rmse_lstm + rmse_xgb) == 0:
            alpha = 0.5
        else:
            alpha = 1 - (rmse_lstm / (rmse_lstm + rmse_xgb))

        print(f"[INFO] HYBRID Weights — LSTM: {round(alpha, 2)} | XGBoost: {round(1 - alpha, 2)}")

        hybrid_preds = alpha * lstm_preds + (1 - alpha) * xgb_preds
        lower = hybrid_preds * 0.97
        upper = hybrid_preds * 1.03
        return hybrid_preds, lower, upper, close_prices.flatten()

    raise ValueError("❌ Invalid model_type. Use: LSTM, XGBOOST, or HYBRID.")
