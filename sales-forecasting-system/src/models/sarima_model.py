import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

# Suppress ConvergenceWarnings from statsmodels
warnings.filterwarnings("ignore")

def fit_sarima_model(series: pd.Series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)):
    """
    Function to fit a SARIMA model using statsmodels.
    """
    print(f"Fitting SARIMA{order}x{seasonal_order}...")
    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    print("Model fit complete.")
    return results

def forecast_on_validation(results, steps: int):
    """
    Predict on the validation period and return predictions.
    """
    print(f"Generating forecast for {steps} steps...")
    forecast = results.get_forecast(steps=steps)
    return forecast.predicted_mean

class SARIMAForecaster:
    """
    A wrapper for the SARIMA model from statsmodels.
    """
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_result = None

    def fit(self, series: pd.Series):
        """
        Fit the SARIMA model to a time series.
        """
        print(f"Fitting SARIMA{self.order}x{self.seasonal_order}...")
        model = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.model_result = model.fit(disp=False)
        print("Model fit complete.")
        return self.model_result

    def predict(self, steps: int):
        """
        Predict the next 'steps' values.
        """
        if self.model_result is None:
            raise ValueError("Model must be fitted before calling predict.")
        
        forecast = self.model_result.get_forecast(steps=steps)
        return forecast.predicted_mean

if __name__ == "__main__":
    # Test on a dummy series
    import numpy as np
    dummy_data = pd.Series(np.random.randn(100).cumsum())
    
    # Using a smaller seasonal order for testing to speed up fit
    forecaster = SARIMAForecaster(order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
    forecaster.fit(dummy_data)
    forecast = forecaster.predict(steps=10)
    print("\nSample Forecast:")
    print(forecast)
