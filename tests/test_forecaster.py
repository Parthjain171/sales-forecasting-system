import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os
from pathlib import Path

# Add src to path for tests
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / 'src'))

from forecaster import ForecasterEngine

@pytest.fixture
def sample_history():
    dates = pd.date_range(start='2023-01-01', periods=40, freq='W')
    data = {
        'Date': dates,
        'State': ['California'] * 40,
        'Total': np.random.rand(40) * 1000,
    }
    return pd.DataFrame(data)

def test_predict_future_shape(sample_history):
    with patch('forecaster.ForecasterEngine._load_mapping') as mock_mapping:
        mock_mapping.return_value = {'California': 'XGBoost'}
        
        with patch('forecaster.ForecasterEngine.load_model_for_state') as mock_load:
            # Mock the model's predict method
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([500.0])
            mock_load.return_value = mock_model
            
            # Use a dummy path that won't be used anyway due to mocks
            engine = ForecasterEngine(mapping_path="dummy.json")
            steps = 8
            forecast = engine.predict_future('California', sample_history, steps=steps)
            
            assert len(forecast) == steps
            assert 'Date' in forecast.columns
            assert 'Predicted_Sales' in forecast.columns
            assert forecast['Predicted_Sales'].iloc[0] == 500.0

def test_forecaster_engine_initialization_error():
    with pytest.raises(FileNotFoundError):
        ForecasterEngine(mapping_path="non_existent.json")
