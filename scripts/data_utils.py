import pandas as pd
from config.db_config import engine

def fetch_data(symbol="BTCUSDT"):
    query = f"""
    SELECT timestamp, close
    FROM crypto_ohlcv
    WHERE symbol = '{symbol}'
    ORDER BY timestamp ASC
    """
    return pd.read_sql(query, engine)
