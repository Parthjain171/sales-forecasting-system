import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from models.lstm_model import LSTMForecaster
from model_training import time_series_split

def visualize_loss():
    # Use standard paths
    data_path = "data/features_data.csv"
    if not os.path.exists(data_path):
        print("Data not found at any path.")
        return

    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Use California as it has the most data
    state = "California"
    state_df = df[df['State'] == state].copy()
    train_df, val_df = time_series_split(state_df)
    train_series = train_df.set_index('Date')['Total']
    
    print(f"Starting LSTM training for {state} to capture loss history...")
    
    # We modify the fit logic slightly here to capture history
    forecaster = LSTMForecaster()
    
    # Scale data (mirroring fit() logic)
    values = train_series.values.reshape(-1, 1)
    scaled_values = forecaster.scaler.fit_transform(values).flatten()
    X, y = forecaster.prepare_sequences(scaled_values)
    
    # Fit and capture history
    history = forecaster.model.fit(X, y, epochs=50, batch_size=32, verbose=0, validation_split=0.1)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2, color='#3b82f6')
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='#ef4444')
    plt.title(f'LSTM Training Loss Curve - {state}', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Style the plot like the dashboard
    plt.gca().set_facecolor('#f8fafc')
    plt.gcf().set_facecolor('#f8fafc')
    
    output_path = "charts/lstm_loss_curve.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Loss graph saved to: {output_path}")

if __name__ == "__main__":
    visualize_loss()
