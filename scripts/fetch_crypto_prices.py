import sys
import os
import time
import requests
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine

# Ensure Python finds the config directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.db_config import engine

BINANCE_US_API_URL = "https://api.binance.us/api/v3/ticker/price"

def fetch_and_store_crypto_prices():
    """Fetches real-time crypto prices and stores only new entries in PostgreSQL."""
    while True:
        response = requests.get(BINANCE_US_API_URL)

        if response.status_code == 200:
            data = response.json()
            prices = []

            for entry in data:
                symbol = entry["symbol"]
                price = float(entry["price"])
                timestamp = datetime.now()

                prices.append((symbol, price, timestamp))

            df = pd.DataFrame(prices, columns=["symbol", "price", "timestamp"])

            # Prevent duplicate entries by checking the latest timestamp
            query = """
            SELECT MAX(timestamp) FROM crypto_prices;
            """
            latest_timestamp = pd.read_sql(query, engine).iloc[0, 0]

            if latest_timestamp:
                df = df[df["timestamp"] > latest_timestamp]

            if not df.empty:
                df.to_sql("crypto_prices", engine, if_exists="append", index=False)
                print(f"[INFO] Stored {len(df)} new prices in PostgreSQL.")
            else:
                print("[INFO] No new prices to insert.")

        else:
            print(f"[ERROR] Failed to fetch data: {response.status_code}")

        time.sleep(10)  # Fetch every 10 seconds

if __name__ == "__main__":
    fetch_and_store_crypto_prices()
