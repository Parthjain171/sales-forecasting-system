import pandas as pd
from model_training import time_series_split
from models.xgboost_model import XGBoostForecaster
import os

def evaluate_xgboost_on_pipeline():
    # Load feature data
    data_path = "data/features_data.csv"
    if not os.path.exists(data_path):
        print("Please run feature_engineering.py first.")
        return
        
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    print("\n--- Evaluating XGBoost using Phase 3 Engineered Features ---")
    
    # Split data (80/20)
    train_df, val_df = time_series_split(df)
    
    # Initialize Forecaster
    forecaster = XGBoostForecaster(n_estimators=500, learning_rate=0.1)
    
    # Fit model (uses lags, rolling stats, holidays, etc. as inputs)
    forecaster.fit(train_df, val_df)
    
    # Predict on validation set
    predictions = forecaster.predict(val_df)
    
    # Analyze Feature Importance
    feature_names = train_df.drop(columns=['Total', 'Date', 'State']).columns
    importance = forecaster.get_feature_importance(feature_names)
    print("\nTop 5 Most Important Features:")
    print(importance.head(5))
    
    # Compare results
    comparison = pd.DataFrame({
        'Date': val_df['Date'],
        'State': val_df['State'],
        'Actual': val_df['Total'],
        'Predicted': predictions
    })
    
    print("\nComparison (Sample Rows):")
    print(comparison.head())
    
    # Save results
    output_path = "data/xgboost_predictions.csv"
    comparison.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")

if __name__ == "__main__":
    evaluate_xgboost_on_pipeline()
