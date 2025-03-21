import shap
import joblib
import numpy as np
import pandas as pd
from config.db_config import engine
from scripts.data_utils import fetch_data

def explain_xgboost_model(symbol="BTCUSDT"):
    # Load model and scaler
    model = joblib.load(f"models/xgb_{symbol}.joblib")
    scaler = np.load(f"models/xgb_scaler_{symbol}.npy", allow_pickle=True).item()

    # Load data
    df = fetch_data(symbol)
    close_prices = df["close"].values.reshape(-1, 1)
    
    # Read time steps
    with open(f"models/xgb_time_steps_{symbol}.txt", "r") as f:
        time_steps = int(f.read())

    # Prepare features
    scaled = scaler.transform(close_prices).flatten()
    X = np.array([scaled[i:i + time_steps] for i in range(len(scaled) - time_steps)])
    X = X[-500:]  # Limit to recent data for performance

    # Convert to DataFrame with feature names
    feature_names = [f"lag_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)

    # SHAP Explanation
    explainer = shap.Explainer(model)
    shap_values = explainer(X_df)

    return shap_values, X_df
