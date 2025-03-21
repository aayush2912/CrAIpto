import sys
import os
import pandas as pd
import numpy as np
import ta
from sqlalchemy import create_engine

# Fix ModuleNotFoundError
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import database connection
from config.db_config import engine

def fetch_ohlcv_data(symbol="BTCUSDT", limit=100):
    """Fetch OHLCV data from PostgreSQL"""
    query = f"""
    SELECT * FROM crypto_ohlcv
    WHERE symbol = '{symbol}'
    ORDER BY timestamp DESC
    LIMIT {limit}
    """
    df = pd.read_sql(query, engine)
    return df

def compute_indicators(df):
    """Compute technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands"""
    
    # Ensure data is sorted (Oldest to Newest)
    df = df.sort_values(by="timestamp")

    # Simple Moving Average (SMA)
    df["SMA_20"] = ta.trend.sma_indicator(df["close"], window=20)

    # Exponential Moving Average (EMA)
    df["EMA_20"] = ta.trend.ema_indicator(df["close"], window=20)

    # Relative Strength Index (RSI)
    df["RSI_14"] = ta.momentum.rsi(df["close"], window=14)

    # Moving Average Convergence Divergence (MACD)
    df["MACD"] = ta.trend.macd(df["close"])
    df["MACD_Signal"] = ta.trend.macd_signal(df["close"])
    df["MACD_Hist"] = ta.trend.macd_diff(df["close"])

    # Bollinger Bands
    df["Bollinger_High"] = ta.volatility.bollinger_hband(df["close"], window=20)
    df["Bollinger_Low"] = ta.volatility.bollinger_lband(df["close"], window=20)

    return df

def store_indicators(df):
    """Store computed indicators in PostgreSQL"""
    df.to_sql("crypto_indicators", engine, if_exists="replace", index=False)
    print(f"[INFO] Stored {len(df)} technical indicators in PostgreSQL.")

def main():
    """Main function to fetch, compute, and store indicators"""
    symbol = "BTCUSDT"
    df = fetch_ohlcv_data(symbol)
    df = compute_indicators(df)
    store_indicators(df)

if __name__ == "__main__":
    main()
