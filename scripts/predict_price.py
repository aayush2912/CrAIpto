import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine

# Ensure Python finds the config directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import database connection
from config.db_config import engine

# Load Model & Scaler
def load_lstm_model(symbol="BTCUSDT"):
    model_path = f"models/lstm_{symbol}.h5"
    scaler_path = f"models/scaler_{symbol}.npy"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("[ERROR] Model or scaler not found. Train the model first.")
        return None, None

    model = load_model(model_path)
    scaler = np.load(scaler_path, allow_pickle=True).item()
    return model, scaler

# Fetch latest price data
def fetch_latest_data(symbol="BTCUSDT", time_steps=50):
    query = f"""
    SELECT timestamp, close FROM crypto_ohlcv
    WHERE symbol = '{symbol}'
    ORDER BY timestamp DESC
    LIMIT {time_steps};
    """
    df = pd.read_sql(query, engine)
    return df[::-1]  # Reverse for time series order

# Predict Future Price
def predict_future_price(symbol="BTCUSDT"):
    model, scaler = load_lstm_model(symbol)
    if model is None or scaler is None:
        return None

    df = fetch_latest_data(symbol)
    if df.empty:
        print("[ERROR] No recent data found.")
        return None

    scaled_data = scaler.transform(df[['close']])
    X = np.array([scaled_data])
    prediction = model.predict(X)
    future_price = scaler.inverse_transform(prediction)[0][0]

    return future_price

if __name__ == "__main__":
    predicted_price = predict_future_price("BTCUSDT")
    if predicted_price:
        print(f"[INFO] Predicted BTC Price: ${predicted_price:.2f}")
