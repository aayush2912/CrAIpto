import sys
import os
import keras_tuner as kt
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine

# Setup path to access config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.db_config import engine

def fetch_data(symbol="BTCUSDT", limit=1000):
    query = f"""
    SELECT timestamp, close FROM crypto_ohlcv
    WHERE symbol = '{symbol}'
    ORDER BY timestamp ASC
    LIMIT {limit};
    """
    return pd.read_sql(query, engine)

def prepare_data(df, time_steps):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['close']])

    X, y = [], []
    for i in range(time_steps, len(scaled)):
        X.append(scaled[i - time_steps:i])
        y.append(scaled[i])

    return np.array(X), np.array(y), scaler

# Now accepts `time_steps` from the tuning loop
def build_model(hp, time_steps):
    model = Sequential()

    units = hp.Choice('units', values=[32, 64, 128])

    model.add(LSTM(units=units, return_sequences=True, input_shape=(time_steps, 1)))
    model.add(Dropout(hp.Float('dropout_1', 0.1, 0.4, step=0.1)))

    model.add(LSTM(units=units))
    model.add(Dropout(hp.Float('dropout_2', 0.1, 0.4, step=0.1)))

    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        ),
        loss='mean_squared_error'
    )

    return model

def run_tuning(symbol="BTCUSDT"):
    df = fetch_data(symbol)
    os.makedirs("models", exist_ok=True)

    # Try different time_steps manually
    for time_steps in [30, 50, 70, 90]:
        print(f"\n🚀 Tuning with time_steps = {time_steps}")

        # Prepare data
        X, y, scaler = prepare_data(df, time_steps)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Create tuner for each time step size
        tuner = kt.RandomSearch(
            lambda hp: build_model(hp, time_steps),
            objective='val_loss',
            max_trials=10,
            directory='tuning',
            project_name=f'{symbol}_lstm_ts_{time_steps}'
        )

        tuner.search(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

        best_hps = tuner.get_best_hyperparameters(1)[0]
        print("✅ Best hyperparameters found:")
        print(f"  • Units: {best_hps.get('units')}")
        print(f"  • Dropout 1: {best_hps.get('dropout_1')}")
        print(f"  • Dropout 2: {best_hps.get('dropout_2')}")
        print(f"  • Learning Rate: {best_hps.get('learning_rate')}")

        print(f"\n📦 Retraining best model for time_steps={time_steps}...")
        best_model = tuner.hypermodel.build(best_hps)
        best_model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2, verbose=1)

        # Save model, scaler, and time_steps
        best_model.save(f"models/lstm_{symbol}.h5")
        np.save(f"models/scaler_{symbol}.npy", scaler)
        with open(f"models/time_steps_{symbol}.txt", "w") as f:
            f.write(str(time_steps))

        print(f"🎉 Model saved with time_steps={time_steps}.\n")
        break  # Remove this break to try other time_steps too

if __name__ == "__main__":
    run_tuning("BTCUSDT")
