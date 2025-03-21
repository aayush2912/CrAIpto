import numpy as np
import pandas as pd
from scripts.multi_step_predict import predict_next_n_steps
from config.db_config import engine
from scripts.data_utils import fetch_data

def simulate_trading(symbol="BTCUSDT", model_type="HYBRID", steps=30, threshold=0.01):
    df = fetch_data(symbol)
    df = df.sort_values("timestamp")
    prices = df["close"].values[-steps:]  # last N steps
    timestamps = df["timestamp"].values[-steps:]

    pred, lower, upper, _ = predict_next_n_steps(symbol, steps_ahead=steps, model_type=model_type)

    actions = []
    cash = 10000  # Starting balance
    position = 0
    portfolio = []

    for i in range(steps):
        predicted_price = pred[i]
        actual_price = prices[i]
        time = timestamps[i]

        signal = "HOLD"
        if predicted_price > actual_price * (1 + threshold) and cash >= actual_price:
            # Buy
            position += 1
            cash -= actual_price
            signal = "BUY"
        elif predicted_price < actual_price * (1 - threshold) and position > 0:
            # Sell
            position -= 1
            cash += actual_price
            signal = "SELL"

        total_value = cash + position * actual_price
        actions.append({
            "timestamp": time,
            "actual_price": actual_price,
            "predicted_price": predicted_price,
            "action": signal,
            "position": position,
            "cash": round(cash, 2),
            "total_value": round(total_value, 2)
        })

    return pd.DataFrame(actions)
