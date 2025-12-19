
import argparse
import sys
import os
import datetime
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_pull import fetch_data, preprocess_data
from forecast_model import LSTMModel, train_model, predict
from visualization import plot_compact_prediction

def main():
    parser = argparse.ArgumentParser(description='Stock Forecast Engine')
    parser.add_argument('--tickers', type=str, nargs='+', default=['AAPL', 'GOOGL', 'MSFT'], help='List of tickers')
    parser.add_argument('--days', type=int, default=365*2, help='Days of history to fetch')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--output', type=str, default='output_plots', help='Output directory for plots')
    
    args = parser.parse_args()
    
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=args.days)
    
    print(f"Fetching data for {args.tickers}...")
    df = fetch_data(args.tickers, start_date, end_date)
    
    if df is None or df.empty:
        print("No data found.")
        return

    # Handle single ticker case where DF columns are not MultiIndex
    # yfinance returns different shapes. If multiple tickers, columns are tickers.
    # If single ticker 'Adj Close' is the series/col.
    if len(args.tickers) == 1 and isinstance(df, pd.DataFrame):
         df.columns = args.tickers

    print("Preprocessing data...")
    # seq_length = 60 days
    processed_data, scalers = preprocess_data(df, seq_length=60)
    
    for ticker in args.tickers:
        print(f"\nProcessing {ticker}...")
        if ticker not in processed_data:
            print(f"Skipping {ticker}, no data.")
            continue
            
        data = processed_data[ticker]
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        
        if len(X_train) == 0 or len(X_test) == 0:
            print(f"Not enough data for {ticker}")
            continue
            
        print(f"Training model for {ticker} with {len(X_train)} samples...")
        model = LSTMModel()
        model = train_model(model, X_train, y_train, epochs=args.epochs)
        
        print(f"Predicting...")
        preds = predict(model, X_test)
        
        # Inverse transform
        scaler = scalers[ticker]
        # scaler was fitted on shape (n, 1)
        # y_test and preds are (n, 1)
        y_test_inv = scaler.inverse_transform(y_test)
        preds_inv = scaler.inverse_transform(preds)
        
        print(f"Generating plot...")
        plot_compact_prediction(ticker, None, y_test_inv, preds_inv, output_dir=args.output)
        
    print("\nDone! Check output directory.")

if __name__ == "__main__":
    main()
