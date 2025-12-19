
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_data(tickers, start_date, end_date):
    """
    Fetch historical data for given tickers.
    """
    logger.info(f"Fetching data for {tickers} from {start_date} to {end_date}")
    try:
        data = yf.download(tickers, start=start_date, end=end_date)
        
        # Check for empty data
        if data is None or data.empty:
             return None

        # yfinance return structure varies. 
        # Usually it is MultiIndex columns with levels (Price, Ticker)
        # We want 'Adj Close' or 'Close'
        
        target_col = 'Adj Close'
        # Check if 'Adj Close' is in the top level of columns
        if isinstance(data.columns, pd.MultiIndex):
             if 'Adj Close' not in data.columns.get_level_values(0):
                 target_col = 'Close'
        else:
             if 'Adj Close' not in data.columns:
                 target_col = 'Close'

        logger.info(f"Using column: {target_col}")
        
        if isinstance(data.columns, pd.MultiIndex):
            # If we simply select data[target_col], we get a DF with tickers as columns
            data = data[target_col]
        else:
             # Single level, might provide Open, High, Low, Close, etc.
             data = data[[target_col]]
        
        return data
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None

def create_sequences(data, seq_length):
    """
    Create sequences for LSTM: (samples, time_steps, features)
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def preprocess_data(df, seq_length=60, train_split=0.8):
    """
    Normalize data (using Log Returns) and create sequences.
    Returns: X_train, y_train, X_test, y_test, scaler, last_prices
    """
    results = {}
    scalers = {}
    last_prices = {} # Store last known prices to reconstruct later

    for ticker in df.columns:
        # Calculate Log Returns
        # P_t / P_{t-1} -> log -> diff of logs
        prices = df[[ticker]].values
        # Store last price for reconstruction reference (of the whole series or split?)
        # We need the price just before the sequence starts to reconstruct?
        # Simpler: Just store the original prices series relative to the split.
        
        # Log returns
        # Add small epsilon to avoid log(0) if any, though prices shouldn't be 0
        eps = 1e-8
        returns = np.diff(np.log(prices + eps), axis=0) # Shape (N-1, 1)
        
        # Scale Returns (returns are small, usually -0.1 to 0.1)
        # MinMax is fine, or StandardScaler
        scaler = MinMaxScaler(feature_range=(-1, 1)) # Returns can be negative
        scaled_data = scaler.fit_transform(returns)
        
        X, y = create_sequences(scaled_data, seq_length)
        
        req_len = int(len(X) * train_split)
        
        X_train, y_train = X[:req_len], y[:req_len]
        X_test, y_test = X[req_len:], y[req_len:]
        
        # We need to know the "Base Price" for the test set to reconstruct it.
        # The test set starts at index `req_len`. 
        # The sequences start at `i`. `y` is at `i + seq_length`.
        # The returns correspond to `returns`.
        # `returns[0]` is change from price[0] to price[1].
        # `y[0]` (which is from X[0]) corresponds to `returns[seq_length]`.
        # `returns[seq_length]` is change `price[seq_length] -> price[seq_length+1]`.
        
        # So to reconstruct y_test, we need Price at `req_len + seq_length`.
        # Actually, let's just pass the full price series for easy reconstruction in app.py
        
        results[ticker] = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'prices': prices # Raw prices
        }
        scalers[ticker] = scaler
        
    return results, scalers
