import sys
import os

# Ensure Python finds the `config` directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.db_config import engine, BINANCE_API_KEY
import requests
import pandas as pd
from datetime import datetime

# Binance.US API URL
BINANCE_US_API_URL = "https://api.binance.us/api/v3/ticker/price"
HEADERS = {"X-MBX-APIKEY": BINANCE_API_KEY}

# Fetch prices manually
response = requests.get(BINANCE_US_API_URL, headers=HEADERS)

if response.status_code == 200:
    data = response.json()
    prices = []

    for entry in data:
        symbol = entry["symbol"]
        price = float(entry["price"])
        timestamp = datetime.now()
        prices.append((symbol, price, timestamp))

    # Convert to DataFrame
    df = pd.DataFrame(prices, columns=["symbol", "price", "timestamp"])

    # Store in PostgreSQL
    df.to_sql("crypto_prices", engine, if_exists="append", index=False)
    print(f"[INFO] Stored {len(df)} real-time prices in PostgreSQL.")

else:
    print(f"[ERROR] Failed to fetch data: {response.status_code}")
