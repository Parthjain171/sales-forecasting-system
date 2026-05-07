import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from model_training import time_series_split
from models.sarima_model import fit_sarima_model, forecast_on_validation
from models.prophet_model import ProphetForecaster
from models.xgboost_model import XGBoostForecaster
from models.lstm_model import LSTMForecaster
import os
import joblib

def calculate_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    return rmse, mae

def compare_models(state="California"):
    # Load data
    data_path = "data/features_data.csv"
    if not os.path.exists(data_path):
        print("Please run feature_engineering.py first.")
        return
        
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    state_df = df[df['State'] == state].copy()
    
    print(f"\n=== Comparing All Models for {state} ===")
    train_df, val_df = time_series_split(state_df)
    train_series = train_df.set_index('Date')['Total']
    val_series = val_df.set_index('Date')['Total']
    steps = len(val_series)
    
    results = []

    # 1. SARIMA
    print("\n[1/4] Evaluating SARIMA...")
    sarima_res = fit_sarima_model(train_series, order=(1,1,1), seasonal_order=(0,0,0,0))
    sarima_preds = forecast_on_validation(sarima_res, steps)
    results.append(("SARIMA", *calculate_metrics(val_series, sarima_preds), sarima_res))

    # 2. Prophet
    print("\n[2/4] Evaluating Prophet...")
    prophet = ProphetForecaster()
    prophet.fit(train_series)
    prophet_preds = prophet.predict(steps)
    results.append(("Prophet", *calculate_metrics(val_series, prophet_preds), prophet))

    # 3. XGBoost
    print("\n[3/4] Evaluating XGBoost...")
    xgboost = XGBoostForecaster()
    xgboost.fit(train_df, val_df)
    xgboost_preds = xgboost.predict(val_df)
    results.append(("XGBoost", *calculate_metrics(val_series, xgboost_preds), xgboost))

    # 4. LSTM
    print("\n[4/4] Evaluating LSTM...")
    lstm = LSTMForecaster()
    lstm.fit(train_series, epochs=20) 
    lstm_preds = lstm.predict(train_series, steps)
    results.append(("LSTM", *calculate_metrics(val_series, lstm_preds), lstm))

    # Summary
    summary_df = pd.DataFrame(results, columns=['Model', 'RMSE', 'MAE', 'ModelObject'])
    
    # Logic to automatically pick the model with lowest MAE
    summary_df = summary_df.sort_values('MAE', ascending=True)
    
    print("\n=== Model Comparison Summary ===")
    print(summary_df[['Model', 'RMSE', 'MAE']])
    
    best_model_name = summary_df.iloc[0]['Model']
    best_mae = summary_df.iloc[0]['MAE']
    best_model_obj = summary_df.iloc[0]['ModelObject']
    
    print(f"\nAUTO-SELECTOR: The best model for {state} is '{best_model_name}' with an MAE of {best_mae:,.2f}.")
    return best_model_name, summary_df, best_model_obj

if __name__ == "__main__":
    compare_models()
