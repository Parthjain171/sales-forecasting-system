import pandas as pd
from model_training import time_series_split
from models.lstm_model import LSTMForecaster
import os

def evaluate_lstm_on_pipeline():
    # Load feature data
    data_path = "data/features_data.csv"
    if not os.path.exists(data_path):
        print("Please run feature_engineering.py first.")
        return
        
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Evaluate on California
    state = "California"
    state_df = df[df['State'] == state].copy()
    
    print(f"\n--- Evaluating LSTM for {state} ---")
    
    # Split data (80/20)
    train_df, val_df = time_series_split(state_df)
    
    # Initialize Forecaster (default n_steps is now 8)
    forecaster = LSTMForecaster()
    
    # Train for 50 epochs (Phase 4 requirement)
    train_series = train_df.set_index('Date')['Total']
    forecaster.fit(train_series, epochs=50)
    
    # Predict on validation set
    # We use the end of training data as the starting point for predictions
    steps = len(val_df)
    print(f"Predicting for {steps} weeks...")
    predictions = forecaster.predict(train_series, n_future_steps=steps)
    
    # Compare
    val_df = val_df.set_index('Date')
    comparison = pd.DataFrame({
        'Actual': val_df['Total'],
        'Predicted': predictions
    })
    
    print("\nComparison (First 5 weeks of validation):")
    print(comparison.head())
    
    # Save results
    output_path = f"data/lstm_predictions_{state}.csv"
    comparison.to_csv(output_path)
    print(f"\nPredictions saved to {output_path}")

if __name__ == "__main__":
    evaluate_lstm_on_pipeline()
