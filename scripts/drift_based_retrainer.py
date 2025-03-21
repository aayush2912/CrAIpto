import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from xgboost import XGBRegressor
from scripts.data_drift_detector import detect_drift
from config.db_config import engine

def fetch_data(symbol="BTCUSDT", limit=1000):
    query = f"""
    SELECT timestamp, close FROM crypto_ohlcv
    WHERE symbol = '{symbol}'
    ORDER BY timestamp ASC
    LIMIT {limit};
    """
    return pd.read_sql(query, engine)

def retrain_if_drifted(symbol="BTCUSDT", threshold=0.3):
    drift_score = detect_drift(symbol)
    if drift_score <= threshold:
        print(f"[INFO] No retraining needed. Drift score ({drift_score}) below threshold.")
        return

    print(f"[INFO] Drift detected! Retraining models for {symbol}...")

    # Fetch data
    df = fetch_data(symbol)
    close_prices = df["close"].values.reshape(-1, 1)

    # === LSTM Retraining ===
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)
    time_steps = 30

    X, y = [], []
    for i in range(len(scaled) - time_steps):
        X.append(scaled[i:i+time_steps])
        y.append(scaled[i+time_steps])

    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=(X.shape[1], X.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    os.makedirs("models", exist_ok=True)
    model.save(f"models/lstm_{symbol}.h5")
    np.save(f"models/scaler_{symbol}.npy", scaler)
    with open(f"models/time_steps_{symbol}.txt", "w") as f:
        f.write(str(time_steps))

    print(f"✅ LSTM model retrained and saved for {symbol}.")

    # === XGBoost Retraining ===
    X_flat = [seq.flatten() for seq in X]
    model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4)
    model_xgb.fit(X_flat, y)

    joblib.dump(model_xgb, f"models/xgb_{symbol}.joblib")
    np.save(f"models/xgb_scaler_{symbol}.npy", scaler)
    with open(f"models/xgb_time_steps_{symbol}.txt", "w") as f:
        f.write(str(time_steps))

    print(f"✅ XGBoost model retrained and saved for {symbol}.")

    # Save retraining log
    with open(f"logs/retrain_log_{symbol}.txt", "w") as f:
        f.write(f"Symbol: {symbol}\nDrift Score: {drift_score}\nRetrained: Yes\n")
