
import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.getcwd()))

from app import generate_forecast, app

# Mock context if needed (scheduler might auto-start)
logging.basicConfig(level=logging.INFO)

import json
import numpy as np

# JSON encoder to simulate Flask jsonify behavior (which usually fails on basic numpy types)
# We want to see if it fails without helper

tickers = ["GOOGL", "TSLA", "META", "AMZN", "NVDA"]

for t in tickers:
    print(f"\nTesting {t}...")
    try:
        res = generate_forecast(t)
        if res:
            # Try to serialize
            # Flask's jsonify uses simplejson or json, but default json.dumps fails on numpy types 
            # unless a custom encoder is used. Flask doesn't have a custom encoder by default 
            # that handles ALL numpy types automatically (it handles some).
            # But let's see if json.dumps fails.
            json_str = json.dumps(res) 
            print(f"Success! Metrics: {res['metrics']}")
        else:
            print("Returned None")
    except Exception as e:
        print(f"Exception: {e}")
        # import traceback
        # traceback.print_exc()
