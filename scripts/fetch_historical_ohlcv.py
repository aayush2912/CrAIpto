import sys
import os
import requests
import pandas as pd
from datetime import datetime

# Add project root directory to sys.path (Fixes ModuleNotFoundError)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import database connection & API Keys
from config.db_config import engine, BINANCE_API_KEY

# ✅ Binance.US API URL (instead of Binance.com)
BINANCE_US_API_URL = "https://api.binance.us/api/v3/klines"

# ✅ Set API headers
HEADERS = {"X-MBX-APIKEY": BINANCE_API_KEY}

def fetch_historical_ohlcv(symbol="BTCUSDT", interval="1h", limit=100):
    """
    Fetches historical OHLCV (Open, High, Low, Close, Volume) data from Binance.US API
    and stores it in PostgreSQL.
    """
    url = f"{BINANCE_US_API_URL}?symbol={symbol}&interval={interval}&limit={limit}"
    
    response = requests.get(url, headers=HEADERS)

    if response.status_code == 200:
        data = response.json()

        ohlcv_data = []
        for entry in data:
            timestamp = datetime.fromtimestamp(entry[0] / 1000)  # Convert from ms to seconds
            open_price, high, low, close, volume = map(float, entry[1:6])

            ohlcv_data.append((symbol, open_price, high, low, close, volume, timestamp))

        # Convert to DataFrame
        df = pd.DataFrame(ohlcv_data, columns=["symbol", "open", "high", "low", "close", "volume", "timestamp"])

        # ✅ Store in PostgreSQL (Ensure correct schema permissions)
        df.to_sql("crypto_ohlcv", engine, if_exists="append", index=False)
        print(f"[INFO] Stored {len(df)} OHLCV records for {symbol} in PostgreSQL.")

    else:
        print(f"[ERROR] Failed to fetch OHLCV data: {response.status_code} - {response.text}")

if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    fetch_historical_ohlcv(symbol)

