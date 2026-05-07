import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_mae(actual, predicted):
    """
    Calculate Mean Absolute Error between actual and predicted values.
    """
    return mean_absolute_error(actual, predicted)

def calculate_rmse(actual, predicted):
    """
    Calculate Root Mean Squared Error between actual and predicted values.
    """
    return np.sqrt(mean_squared_error(actual, predicted))

def print_model_report(model_name, actual, predicted):
    """
    Print a formatted report for a model's performance.
    """
    mae = calculate_mae(actual, predicted)
    rmse = calculate_rmse(actual, predicted)
    print(f"\n--- {model_name} Performance Report ---")
    print(f"MAE  : {mae:,.2f}")
    print(f"RMSE : {rmse:,.2f}")
    return mae, rmse
