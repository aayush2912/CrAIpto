import sys
import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

# Access config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.db_config import engine

def fetch_data(symbol="BTCUSDT", limit=1000):
    query = f"""
    SELECT timestamp, close FROM crypto_ohlcv
    WHERE symbol = '{symbol}'
    ORDER BY timestamp ASC
    LIMIT {limit};
    """
    return pd.read_sql(query, engine)

def create_supervised_features(data, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps].flatten())  # 👈 Flatten to 1D
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

def train_xgb_model(symbol="BTCUSDT", time_steps=10):
    df = fetch_data(symbol)
    close_prices = df["close"].values.reshape(-1, 1)

    # Normalize
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)

    X, y = create_supervised_features(scaled, time_steps)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4)
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/xgb_{symbol}.joblib")
    np.save(f"models/xgb_scaler_{symbol}.npy", scaler)
    with open(f"models/xgb_time_steps_{symbol}.txt", "w") as f:
        f.write(str(time_steps))

    print(f"✅ XGBoost model trained and saved for {symbol}.")

if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    train_xgb_model(symbol, time_steps=10)

