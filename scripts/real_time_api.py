import sys
import os

# Add project root directory to sys.path (Fixes ModuleNotFoundError)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, jsonify
import pandas as pd
from config.db_config import engine  # Import Database Connection

app = Flask(__name__)

@app.route('/latest_prices', methods=['GET'])
def get_latest_prices():
    """Fetches the latest crypto prices from PostgreSQL."""
    query = """
    SELECT * FROM crypto_prices 
    ORDER BY timestamp DESC 
    LIMIT 10
    """
    df = pd.read_sql(query, engine)
    
    # Convert to JSON
    data = df.to_dict(orient="records")
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
