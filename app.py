
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

from data_pull import fetch_data, preprocess_data
from forecast_model import LSTMModel, train_model, predict

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache models in memory for demo purposes (simple cache)
model_cache = {}
scaler_cache = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict/<ticker>')
def get_prediction(ticker):
    days = 365 * 2
    epochs = 15 # Fast training for demo
    
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days)
    
    try:
        # DATA FETCH
        logger.info(f"Fetching data for {ticker}...")
        df = fetch_data([ticker], start_date, end_date)
        
        if df is None or df.empty:
            return jsonify({'error': 'No data found'}), 404
            
        # Ensure correct column name handling
        if hasattr(df, 'columns') and isinstance(df.columns, pd.MultiIndex):
             if ticker in df.columns.levels[1]:
                  df = df.xs(ticker, axis=1, level=1)
        
        if len(df.columns) == 1:
            df.columns = [ticker]
            
        # PREPROCESS (Now uses Log Returns)
        processed_data, scalers = preprocess_data(df, seq_length=60)
        
        if ticker not in processed_data:
             return jsonify({'error': 'Insufficient data after processing'}), 400
             
        data = processed_data[ticker]
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        raw_prices = data['prices'] # Shape (N, 1)
        
        # Calculate split point to know where test starts
        # X length = len(returns) - seq_len
        # Train split = 0.8
        seq_length = 60
        returns_len = len(raw_prices) - 1
        X_len = returns_len - seq_length
        train_len = int(X_len * 0.8)
        
        # Test sequences start at index `train_len` relative to X
        # This corresponds to `returns` index `train_len`.
        # The target `y` corresponds to `returns` index `train_len + seq_length`.
        # So the first predicted return is the change from `Price[train_len + seq_length]` to `Price[train_len + seq_length + 1]`
        
        base_price_test_index = train_len + seq_length
        
        scaler = scalers[ticker]

        # MODEL TRAIN
        model = LSTMModel()
        model = train_model(model, X_train, y_train, epochs=epochs)
        
        # PREDICT
        # Predict on Test (Backtest)
        preds = predict(model, X_test) # (M, 1) scaled returns
        
        # Predict Future
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

        # INVERSE TRANSFORM (Returns)
        y_test_inv = scaler.inverse_transform(y_test) # Actual Returns
        preds_inv = scaler.inverse_transform(preds)   # Predicted Returns
        future_preds_inv = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

        # RECONSTRUCT PRICES
        # Price_t = Price_{t-1} * exp(Return_t)
        # We process cumulatively.
        
        # Backtest Reconstruction
        # We need the price just before the first test prediction.
        # That is raw_prices[base_price_test_index]
        
        test_base_price = float(raw_prices[base_price_test_index])
        
        # We want to reconstruct the path.
        # Path_Actual = base * cumprod(exp(act_ret))
        # Path_Pred   = base * cumprod(exp(pred_ret))
        
        actual_path = [test_base_price]
        pred_path   = [test_base_price]
        
        for r in y_test_inv.flatten():
            actual_path.append(actual_path[-1] * np.exp(r))
            
        for r in preds_inv.flatten():
            pred_path.append(pred_path[-1] * np.exp(r))
            
        # Future Reconstruction applies to the LAST ACTUAL Price
        last_actual_price = float(raw_prices[-1])
        future_path = [last_actual_price]
        
        for r in future_preds_inv.flatten():
            future_path.append(future_path[-1] * np.exp(r))

        # Exclude the base seeds from list to align with time steps if needed, 
        # or keep them. Let's keep them and let frontend handle visual overlap.
        # But wait, y_test_inv length is N. actual_path length is N+1.
        # The first point is "known history".
        # Let's return the computed path excluding the base seed for "predicted" part?
        # Actually better to return specific points.
        
        # Convert numpy types to native floats for JSON serialization
        actual_path = [float(x) for x in actual_path]
        pred_path = [float(x) for x in pred_path]
        future_path = [float(x) for x in future_path]

        # METRICS CALCULATION (Backtest)
        # Exclude the first element (seed price)
        y_true = np.array(actual_path[1:])
        y_pred = np.array(pred_path[1:])
        
        mse = np.mean((y_true - y_pred) ** 2)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        response = {
            'ticker': ticker,
            'metrics': {
                'mse': float(mse),
                'mape': float(mape)
            },
            'backtest': {
                'actual': actual_path[1:], 
                'predicted': pred_path[1:]
            },
            'forecast': future_path[1:]
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/top10')
def get_top10():
    # Hardcoded top 20 Nasdaq/Tech/Growth for demo
    tickers = [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "COST", "PEP",
        "ADBE", "CSCO", "NFLX", "AMD", "TMUS", "INTC", "QCOM", "TXN", "HON", "AMGN"
    ]
    return jsonify(tickers)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
