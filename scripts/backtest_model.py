import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.db_config import engine
from scripts.data_utils import fetch_data

import joblib
from tensorflow.keras.models import load_model

def backtest(symbol="BTCUSDT", model_type="LSTM", window_stride=1):
    df = fetch_data(symbol)
    df = df.sort_values("timestamp")
    prices = df["close"].values.reshape(-1, 1)

    # Load model + scaler + time_steps
    if model_type.upper() == "LSTM":
        model = load_model(f"models/lstm_{symbol}.h5")
        scaler = np.load(f"models/scaler_{symbol}.npy", allow_pickle=True).item()
        with open(f"models/time_steps_{symbol}.txt", "r") as f:
            time_steps = int(f.read())

    elif model_type.upper() == "XGBOOST":
        model = joblib.load(f"models/xgb_{symbol}.joblib")
        scaler = np.load(f"models/xgb_scaler_{symbol}.npy", allow_pickle=True).item()
        with open(f"models/xgb_time_steps_{symbol}.txt", "r") as f:
            time_steps = int(f.read())

    else:
        raise ValueError("Invalid model type. Use 'LSTM' or 'XGBOOST'.")

    # Scale data
    scaled = scaler.transform(prices).flatten()
    preds, actuals = [], []

    for i in range(0, len(scaled) - time_steps - 1, window_stride):
        input_seq = scaled[i:i + time_steps]
        true_val = scaled[i + time_steps]

        if model_type.upper() == "LSTM":
            x = np.reshape(input_seq, (1, time_steps, 1))
        else:
            x = np.reshape(input_seq, (1, time_steps))

        pred = model.predict(x)[0]
        preds.append(float(pred))
        actuals.append(true_val)

    # Inverse transform to original price scale
    pred_prices = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    true_prices = scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(true_prices, pred_prices))
    mae = mean_absolute_error(true_prices, pred_prices)

    return rmse, mae, pred_prices, true_prices

# ----------------------------------------------
# 🔁 Multi-Crypto Backtest + CSV Leaderboard
# ----------------------------------------------
if __name__ == "__main__":
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    models = ["LSTM", "XGBOOST"]

    results = []

    for symbol in symbols:
        for model in models:
            try:
                print(f"🔁 Backtesting {model} on {symbol}...")
                rmse, mae, _, _ = backtest(symbol=symbol, model_type=model)
                results.append({
                    "Symbol": symbol,
                    "Model": model,
                    "RMSE": round(rmse, 2),
                    "MAE": round(mae, 2)
                })
            except Exception as e:
                print(f"❌ Failed for {symbol} - {model}: {e}")

    df = pd.DataFrame(results)
    print("\n🏆 Model Leaderboard:\n", df)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/model_leaderboard.csv", index=False)
