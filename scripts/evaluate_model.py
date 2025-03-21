import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine

# Set path to access config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.db_config import engine

def fetch_data(symbol="BTCUSDT", limit=500):
    query = f"""
    SELECT timestamp, close FROM crypto_ohlcv
    WHERE symbol = '{symbol}'
    ORDER BY timestamp ASC
    LIMIT {limit};
    """
    return pd.read_sql(query, engine)

def prepare_data(df, time_steps=50):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['close']])
    
    X, y = [], []
    for i in range(time_steps, len(scaled)):
        X.append(scaled[i-time_steps:i])
        y.append(scaled[i])
    
    return np.array(X), np.array(y), scaler

def evaluate_model(symbol="BTCUSDT"):
    model_path = f"models/lstm_{symbol}.h5"
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return

    df = fetch_data(symbol)
    X, y, scaler = prepare_data(df)
    
    model = load_model(model_path)
    y_pred_scaled = model.predict(X)
    
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n📊 [Evaluation for {symbol}]")
    print(f"✅ RMSE: {rmse:.2f}")
    print(f"✅ MAE: {mae:.2f}")
    print(f"✅ R² Score: {r2:.4f}")

    # Plot Actual vs Predicted
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="Actual Price")
    plt.plot(y_pred, label="Predicted Price")
    plt.title(f"{symbol} - Actual vs Predicted")
    plt.xlabel("Time")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_model("BTCUSDT")
