import pandas as pd
from prophet import Prophet
import logging

# Suppress Prophet's verbose logging
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

def prepare_prophet_data(df: pd.DataFrame, date_col: str = 'Date', target_col: str = 'Total'):
    """
    Rename columns to 'ds' and 'y' as required by Prophet.
    """
    prophet_df = df[[date_col, target_col]].copy()
    prophet_df.columns = ['ds', 'y']
    return prophet_df

class ProphetForecaster:
    """
    A wrapper for the Facebook Prophet model.
    """
    def __init__(self, weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False):
        self.model = Prophet(
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            daily_seasonality=daily_seasonality
        )
        self.is_fitted = False

    def _prepare_data(self, series: pd.Series) -> pd.DataFrame:
        """
        Convert a pandas Series (with datetime index) to Prophet's ds/y format.
        """
        df = series.reset_index()
        df.columns = ['ds', 'y']
        return df

    def fit(self, series: pd.Series):
        """
        Fit the Prophet model.
        """
        print("Fitting Prophet model...")
        df = self._prepare_data(series)
        self.model.fit(df)
        self.is_fitted = True
        print("Model fit complete.")

    def predict(self, steps: int, freq: str = 'W'):
        """
        Predict the next 'steps' values.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict.")
        
        future = self.model.make_future_dataframe(periods=steps, freq=freq, include_history=False)
        forecast = self.model.predict(future)
        return forecast[['ds', 'yhat']].set_index('ds')['yhat']

def fit_prophet_model(series: pd.Series):
    """
    Standalone function to fit a Prophet model.
    """
    forecaster = ProphetForecaster()
    forecaster.fit(series)
    return forecaster

if __name__ == "__main__":
    # Test on a dummy series
    import numpy as np
    dates = pd.date_range(start='2020-01-01', periods=100, freq='W')
    dummy_data = pd.Series(np.random.randn(100).cumsum(), index=dates)
    
    forecaster = fit_prophet_model(dummy_data)
    forecast = forecaster.predict(steps=10)
    print("\nSample Forecast:")
    print(forecast.head())
