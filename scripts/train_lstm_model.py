import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Fix for relative imports

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from config.db_config import engine
from scripts.multi_step_predict import fetch_data

def train_lstm_model(symbol="BTCUSDT", time_steps=30, epochs=50):
    df = fetch_data(symbol)
    if df.empty or len(df) < time_steps + 1:
        print(f"[ERROR] Not enough data to train for {symbol}.")
        return

    df = df.sort_values("timestamp")
    close_prices = df["close"].values.reshape(-1, 1)

    # Scale prices
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)

    # Create sequences
    X, y = [], []
    for i in range(time_steps, len(scaled)):
        X.append(scaled[i - time_steps:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(time_steps, 1)))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")

    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

    # Save model + scaler + config
    os.makedirs("models", exist_ok=True)
    model.save(f"models/lstm_{symbol}.h5")
    np.save(f"models/scaler_{symbol}.npy", scaler, allow_pickle=True)
    with open(f"models/time_steps_{symbol}.txt", "w") as f:
        f.write(str(time_steps))

    print(f"[INFO] Model trained & saved for {symbol} ✅")

# ----------------------------
if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    train_lstm_model(symbol)
