import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

class LSTMForecaster:
    """
    A wrapper for an LSTM model using TensorFlow/Keras.
    """
    def __init__(self, n_steps=8, n_features=1):
        self.n_steps = n_steps
        self.n_features = n_features
        self.model = self._build_model()
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def _build_model(self):
        """
        Build a simple LSTM architecture (1 LSTM layer + 1 Dense layer).
        """
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(self.n_steps, self.n_features)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def prepare_sequences(self, data: np.array):
        """
        Transform a 1D array into sequences for LSTM.
        Input shape: [samples]
        Output shape: [samples - n_steps, n_steps, n_features]
        """
        X, y = [], []
        for i in range(len(data)):
            end_ix = i + self.n_steps
            if end_ix > len(data) - 1:
                break
            seq_x, seq_y = data[i:end_ix], data[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        
        X = np.array(X).reshape((-1, self.n_steps, self.n_features))
        y = np.array(y)
        return X, y

    def fit(self, series: pd.Series, epochs=20, batch_size=32):
        """
        Scale and fit the LSTM model.
        """
        print("Scaling data and preparing sequences...")
        values = series.values.reshape(-1, 1)
        scaled_values = self.scaler.fit_transform(values).flatten()
        
        X, y = self.prepare_sequences(scaled_values)
        
        print(f"Fitting LSTM on {len(X)} sequences...")
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        print("Model fit complete.")

    def predict(self, series: pd.Series, n_future_steps: int):
        """
        Predict the next steps.
        """
        values = series.values.reshape(-1, 1)
        scaled_values = self.scaler.transform(values).flatten()
        
        # Start with the last known sequence
        current_batch = scaled_values[-self.n_steps:].reshape((1, self.n_steps, self.n_features))
        
        predictions = []
        for _ in range(n_future_steps):
            pred = self.model.predict(current_batch, verbose=0)[0]
            predictions.append(pred)
            # Update batch: drop first, append prediction
            new_val = pred.reshape((1, 1, self.n_features))
            current_batch = np.append(current_batch[:, 1:, :], new_val, axis=1)
            
        # Inverse scale
        return self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

if __name__ == "__main__":
    # Test on dummy series
    dummy_data = pd.Series(np.sin(np.linspace(0, 50, 100)) + np.random.normal(0, 0.1, 100))
    
    # 8-week lookback window (Phase 4 requirement)
    forecaster = LSTMForecaster(n_steps=8, n_features=1)
    forecaster.fit(dummy_data, epochs=5)
    forecast = forecaster.predict(dummy_data, n_future_steps=10)
    
    print("\nSample LSTM Forecast:")
    print(forecast)
