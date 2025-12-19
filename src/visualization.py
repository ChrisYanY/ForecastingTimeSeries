
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_compact_prediction(ticker, history, true_future, prediction, output_dir="plots"):
    """
    Create a compact 'sparkline' style plot.
    history: previous prices
    true_future: actual future prices (test set)
    prediction: model prediction
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(4, 2))  # Small size for compact view
    
    # Concatenate history and validation for a continuous line
    # Note: history is just the last sequence from test set for visualization logic simplicity
    # or we can plot the whole test series.
    # Let's plot the test series vs prediction
    
    plt.plot(np.arange(len(true_future)), true_future, label='Actual', color='gray', linewidth=1)
    plt.plot(np.arange(len(prediction)), prediction, label='Predicted', color='green', linewidth=1)
    
    plt.title(f"{ticker} Forecast", fontsize=10)
    plt.axis('off') # Remove axis for sparkline look
    
    # Add a small legend or just text
    # plt.legend(fontsize=6)
    
    filename = os.path.join(output_dir, f"{ticker}_sparkline.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()
    print(f"Saved plot to {filename}")
