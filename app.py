
from flask import Flask, render_template, jsonify, request
import sys
import os
import datetime
import numpy as np
import pandas as pd
import torch
import logging

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_pull import fetch_data, preprocess_data, fetch_intraday_data
from forecast_model import LSTMModel, train_model, predict

from flask_apscheduler import APScheduler

app = Flask(__name__) # Moved up or already there, merging context
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Scheduler Config
class Config:
    SCHEDULER_API_ENABLED = True

app.config.from_object(Config())

scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()

# Global Cache
DATA_CACHE = {}

from scipy.signal import argrelextrema

def calculate_technicals(prices):
    """
    Calculate Moving Averages and Trend Inflection Points.
    prices: numpy array of prices
    """
    df = pd.DataFrame(prices, columns=['price'])
    
    # Moving Averages
    df['ma15'] = df['price'].rolling(window=15).mean()
    df['ma30'] = df['price'].rolling(window=30).mean()
    df['ma60'] = df['price'].rolling(window=60).mean()
    df['ma180'] = df['price'].rolling(window=180).mean()
    
    # Trend Points (Local Minima/Maxima)
    # Using order=10 (look 10 days ahead/behind)
    p = df['price'].values
    max_idx = argrelextrema(p, np.greater, order=10)[0]
    min_idx = argrelextrema(p, np.less, order=10)[0]
    
    trend_points = []
    for i in max_idx:
        trend_points.append({'index': int(i), 'value': float(p[i]), 'type': 'peark'}) # 'peak' typo intentional? No, let's fix. 'peak'
    for i in min_idx:
        trend_points.append({'index': int(i), 'value': float(p[i]), 'type': 'valley'})
        
    # Replace NaN with None for JSON
    mas = {
        'ma15': df['ma15'].where(pd.notnull(df['ma15']), None).tolist(),
        'ma30': df['ma30'].where(pd.notnull(df['ma30']), None).tolist(),
        'ma60': df['ma60'].where(pd.notnull(df['ma60']), None).tolist(),
        'ma180': df['ma180'].where(pd.notnull(df['ma180']), None).tolist()
    }
    
    return mas, trend_points

import math

def sanitize_for_json(obj):
    """
    Recursively replace NaN/Infinity with None for valid JSON serialization.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(x) for x in obj]
    elif isinstance(obj, np.generic):
         # Handle numpy scalars
         if np.isnan(obj) or np.isinf(obj):
             return None
         return obj.item()
    return obj

def generate_forecast(ticker):
    """
    Core logic to fetch, train, and predict for a single ticker.
    Returns the response dict or None on error.
    """
    days = 365 * 5 # Increased for long-term ma180
    epochs = 15
    
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days)
    
    try:
        logger.info(f"Generating forecast for {ticker}...")
        df = fetch_data([ticker], start_date, end_date)
        
        if df is None or df.empty:
            logger.error(f"No data for {ticker}")
            return None
            
        if hasattr(df, 'columns') and isinstance(df.columns, pd.MultiIndex):
             if ticker in df.columns.levels[1]:
                  df = df.xs(ticker, axis=1, level=1)
        
        if len(df.columns) == 1:
            df.columns = [ticker]
            
        processed_data, scalers = preprocess_data(df, seq_length=60)
        
        if ticker not in processed_data:
             logger.error(f"Insufficient data for {ticker}")
             return None
             
        data = processed_data[ticker]
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        raw_prices = data['prices'] 
        scaler = scalers[ticker] # raw_prices is (N, 1)

        model = LSTMModel()
        model = train_model(model, X_train, y_train, epochs=epochs)
        
        preds = predict(model, X_test)
        
        future_days = 10
        last_seq = X_test[-1].reshape(1, 60, 1)
        future_preds = []
        curr_seq = torch.FloatTensor(last_seq)
        
        # import torch
        model.eval()
        with torch.no_grad():
            for _ in range(future_days):
                pred = model(curr_seq)
                future_preds.append(pred.item())
                new_val = pred.unsqueeze(1)
                curr_seq = torch.cat((curr_seq[:, 1:, :], new_val), dim=1)

        y_test_inv = scaler.inverse_transform(y_test)
        preds_inv = scaler.inverse_transform(preds)
        future_preds_inv = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

        # Reconstruct Prices
        seq_length = 60
        returns_len = len(raw_prices) - 1
        X_len = returns_len - seq_length
        train_len = int(X_len * 0.8)
        base_price_test_index = train_len + seq_length
        test_base_price = float(raw_prices[base_price_test_index])
        
        actual_path = [test_base_price]
        pred_path   = [test_base_price]
        
        for r in y_test_inv.flatten():
            actual_path.append(actual_path[-1] * np.exp(r))
        for r in preds_inv.flatten():
            pred_path.append(pred_path[-1] * np.exp(r))
            
        last_actual_price = float(raw_prices[-1])
        future_path = [last_actual_price]
        for r in future_preds_inv.flatten():
            future_path.append(future_path[-1] * np.exp(r))

        actual_path = [float(x) for x in actual_path]
        pred_path = [float(x) for x in pred_path]
        future_path = [float(x) for x in future_path]

        y_true = np.array(actual_path[1:])
        y_pred = np.array(pred_path[1:])
        mse = np.mean((y_true - y_pred) ** 2)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        full_history_prices = raw_prices.flatten()
        mas, trend_points = calculate_technicals(full_history_prices)
        
        # Intraday Data for High-Res View
        intraday_data = fetch_intraday_data(ticker, period="1mo", interval="60m")
        
        # Dates
        # df.index usually contains the dates. 
        # Check if index is datetime
        try:
            history_dates = df.index.strftime('%Y-%m-%d').tolist()
        except:
            # Fallback if index is not datetime (e.g. integer)
            history_dates = [f"D{i}" for i in range(len(full_history_prices))]

        # Generate Future Dates (Business Days)
        last_date_str = history_dates[-1]
        try:
            last_date = datetime.datetime.strptime(last_date_str, '%Y-%m-%d').date()
        except:
             last_date = datetime.date.today()
             
        future_dates = []
        curr = last_date
        for _ in range(len(future_path) - 1): # First point of future path is last actual
            curr += datetime.timedelta(days=1)
            # Simple skip weekends check
            while curr.weekday() >= 5:
                curr += datetime.timedelta(days=1)
            future_dates.append(curr.strftime('%Y-%m-%d'))
            
        # Full Dates = History + Future
        full_dates = history_dates + future_dates

        result = {
            'ticker': ticker,
            'metrics': { 'mse': float(mse), 'mape': float(mape) },
            'backtest': { 'actual': actual_path[1:], 'predicted': pred_path[1:] },
            'forecast': future_path[1:],
            'full_history': [float(x) for x in full_history_prices],
            'dates': full_dates,
            'technicals': {
                'mas': mas,
                'trend_points': trend_points
            },
            'intraday': intraday_data,
            'last_updated': datetime.datetime.now().isoformat()
        }
        return sanitize_for_json(result)
        
    except Exception as e:
        logger.error(f"Error generating forecast for {ticker}: {e}")
        return None

# --- Scheduled Job ---
def scheduled_update_job():
    logger.info("Starting scheduled update for Top 20 tickers...")
    tickers = get_top20_list()
    for ticker in tickers:
        res = generate_forecast(ticker)
        if res:
            DATA_CACHE[ticker] = res
            logger.info(f"Updated cache for {ticker}")
    logger.info("Scheduled update completed.")

@scheduler.task('interval', id='do_update', minutes=30)
def job1():
    with app.app_context():
        scheduled_update_job()

def get_top20_list():
    return [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "COST", "PEP",
        "ADBE", "CSCO", "NFLX", "AMD", "TMUS", "INTC", "QCOM", "TXN", "HON", "AMGN"
    ]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict/<ticker>')
def get_prediction(ticker):
    # Check Cache
    if ticker in DATA_CACHE:
        logger.info(f"Serving {ticker} from cache.")
        return jsonify(DATA_CACHE[ticker])

    # If not in cache, generate on demand
    res = generate_forecast(ticker)
    if res:
        DATA_CACHE[ticker] = res # Cache it for next time
        return jsonify(res)
    else:
        return jsonify({'error': 'Failed to generate forecast'}), 500

@app.route('/api/top10')
def get_top10():
    return jsonify(get_top20_list())

if __name__ == '__main__':
    app.run(debug=True, port=5000)
