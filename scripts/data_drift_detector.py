import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, entropy
from config.db_config import engine
from scripts.data_utils import fetch_data
from sklearn.preprocessing import MinMaxScaler

def detect_drift(symbol="BTCUSDT", window_size=100):
    df = fetch_data(symbol)
    df = df.sort_values("timestamp")

    close_prices = df["close"].values.reshape(-1, 1)

    # Define baseline (earliest N points) vs recent window
    baseline = close_prices[:window_size].flatten()
    recent = close_prices[-window_size:].flatten()

    # Normalize both
    scaler = MinMaxScaler()
    baseline_scaled = scaler.fit_transform(baseline.reshape(-1, 1)).flatten()
    recent_scaled = scaler.transform(recent.reshape(-1, 1)).flatten()

    # Compute statistics
    ks_stat, ks_pval = ks_2samp(baseline_scaled, recent_scaled)

    # Jensen-Shannon Divergence
    baseline_hist, _ = np.histogram(baseline_scaled, bins=20, range=(0, 1), density=True)
    recent_hist, _ = np.histogram(recent_scaled, bins=20, range=(0, 1), density=True)

    js_div = 0.5 * (entropy(baseline_hist, qk=(baseline_hist + recent_hist)/2) +
                    entropy(recent_hist, qk=(baseline_hist + recent_hist)/2))

    return {
        "ks_stat": ks_stat,
        "ks_pval": ks_pval,
        "js_div": js_div,
        "baseline_mean": np.mean(baseline),
        "recent_mean": np.mean(recent),
        "baseline_std": np.std(baseline),
        "recent_std": np.std(recent)
    }
