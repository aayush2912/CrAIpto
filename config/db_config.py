from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv("config/.env")

# PostgreSQL Database Credentials
DB_USER = "crypto_user"
DB_PASS = "securepassword"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "crypto_db"

# Binance API Keys
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

# Create PostgreSQL Engine
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
