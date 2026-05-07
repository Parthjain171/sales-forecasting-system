import pandas as pd
from model_training import time_series_split
from models.prophet_model import ProphetForecaster, prepare_prophet_data
import os

def evaluate_prophet_on_pipeline():
    # Load feature data
    data_path = "data/features_data.csv"
    if not os.path.exists(data_path):
        print("Please run feature_engineering.py first.")
        return
        
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # For simplicity, let's evaluate on a single state (e.g., California)
    # as Prophet is typically fit per time series.
    state = "California"
    state_df = df[df['State'] == state].copy()
    
    print(f"\n--- Evaluating Prophet for {state} ---")
    
    # Split data
    train_df, val_df = time_series_split(state_df)
    
    # Initialize and fit
    forecaster = ProphetForecaster()
    # Prophet handles its own ds/y conversion in our fit method
    # but we need to pass a Series with Date Index to fit our wrapper's current API
    train_series = train_df.set_index('Date')['Total']
    forecaster.fit(train_series)
    
    # Predict for the length of validation set
    steps = len(val_df)
    print(f"Predicting for {steps} weeks...")
    predictions = forecaster.predict(steps=steps)
    
    # Compare
    val_df = val_df.set_index('Date')
    comparison = pd.DataFrame({
        'Actual': val_df['Total'],
        'Predicted': predictions
    })
    
    print("\nComparison (First 5 weeks of validation):")
    print(comparison.head())
    
    # Save results
    output_path = f"data/prophet_predictions_{state}.csv"
    comparison.to_csv(output_path)
    print(f"\nPredictions saved to {output_path}")

if __name__ == "__main__":
    evaluate_prophet_on_pipeline()
