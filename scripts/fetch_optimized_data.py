import sys
import os
import pandas as pd
from sqlalchemy import create_engine

# Fix ModuleNotFoundError
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import database connection
from config.db_config import engine

def fetch_optimized_indicators(symbol="BTCUSDT", limit=5):
    """Fetch optimized technical indicators using indexed queries"""
    query = f"""
    SELECT * FROM crypto_indicators
    WHERE symbol = '{symbol}'
    ORDER BY timestamp DESC
    LIMIT {limit}
    """
    df = pd.read_sql(query, engine)
    return df

if __name__ == "__main__":
    df = fetch_optimized_indicators("BTCUSDT", 5)
    print(df)
