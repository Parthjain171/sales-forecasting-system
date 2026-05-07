import os
import json
import joblib
import pandas as pd

def get_best_model(state: str, mapping_path="data/best_models.json", model_dir="data/models"):
    """
    Load the saved best model object for a given state.
    """
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Model mapping not found at {mapping_path}")
        
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
        
    if state not in mapping:
        raise ValueError(f"No model found for state: {state}")
        
    model_name = mapping[state]
    model_path = os.path.join(model_dir, f"{state}_{model_name}.joblib")
    
    print(f"Loading {model_name} for {state}...")
    return joblib.load(model_path)

class ForecasterEngine:
    """
    High-level engine to load the best model for a state and generate forecasts.
    """
    def __init__(self, mapping_path="data/best_models.json", model_dir="data/models", preload=False):
        self.mapping_path = mapping_path
        self.model_dir = model_dir
        self.best_models = self._load_mapping()
        self.loaded_models = {}
        
        if preload:
            self.load_all_models()

    def _load_mapping(self):
        """
        Load the {state: model_name} mapping.
        """
        if not os.path.exists(self.mapping_path):
            raise FileNotFoundError(f"Model mapping not found at {self.mapping_path}")
        with open(self.mapping_path, 'r') as f:
            return json.load(f)

    def load_model_for_state(self, state: str):
        """
        Load the specific serialized model object for a given state.
        Checks cache first.
        """
        if state in self.loaded_models:
            return self.loaded_models[state]
            
        if state not in self.best_models:
            raise ValueError(f"No model found for state: {state}")
            
        model_name = self.best_models[state]
        model_path = os.path.join(self.model_dir, f"{state}_{model_name}.joblib")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        print(f"Loading {model_name} for {state}...")
        model = joblib.load(model_path)
        self.loaded_models[state] = model
        return model

    def load_all_models(self):
        """
        Pre-load all models into memory.
        """
        print(f"Pre-loading {len(self.best_models)} models...")
        for state in self.best_models.keys():
            try:
                self.load_model_for_state(state)
            except Exception as e:
                print(f"Failed to pre-load model for {state}: {e}")
        print("All available models loaded into memory.")

    def forecast(self, state: str, input_data: pd.DataFrame):
        """
        Generate a forecast using the best model for the state.
        """
        model = self.load_model_for_state(state)
        predictions = model.predict(input_data)
        return predictions

    def predict_future(self, state: str, history_df: pd.DataFrame, steps: int = 8):
        """
        Predict future steps recursively (handling lags).
        """
        model = self.load_model_for_state(state)
        current_data = history_df.copy().tail(30) # Need at least 30 weeks for Lag_30
        
        future_preds = []
        last_date = pd.to_datetime(current_data['Date'].max())
        
        print(f"Generating recursive forecast for {steps} weeks...")
        
        for i in range(steps):
            next_date = last_date + pd.Timedelta(weeks=i+1)
            
            # 1. Prepare features for the next step
            # Note: In a real scenario, we'd need a more robust feature builder here
            # For this demo, we'll manually set the lags based on current_data
            new_row = {
                'Date': next_date,
                'Year': next_date.year,
                'Month': next_date.month,
                'Week': int(next_date.isocalendar().week),
                'Quarter': next_date.quarter,
                'Lag_1': current_data['Total'].iloc[-1],
                'Lag_7': current_data['Total'].iloc[-7] if len(current_data) >= 7 else 0,
                'Lag_30': current_data['Total'].iloc[-30] if len(current_data) >= 30 else 0,
                'Rolling_Mean_4': current_data['Total'].tail(4).mean(),
                'Rolling_Std_4': current_data['Total'].tail(4).std() if len(current_data) >= 4 else 0,
                'Is_US_Holiday': 0, 
                'Is_Indian_Holiday': 0,
                'Total': 0 # Dummy value for the feature processor
            }
            
            # 2. Predict
            feat_df = pd.DataFrame([new_row])
            
            # Ensure feature order matches the training data exactly
            feature_order = ['Year', 'Month', 'Week', 'Quarter', 'Is_US_Holiday', 'Is_Indian_Holiday', 
                             'Lag_1', 'Lag_7', 'Lag_30', 'Rolling_Mean_4', 'Rolling_Std_4']
            X = feat_df[feature_order]
            
            pred = model.predict(X)[0]
            
            # 3. Update history for the next iteration
            new_row['Total'] = pred
            new_row['State'] = state
            current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
            future_preds.append({'Date': next_date, 'Predicted_Sales': pred})
            
        return pd.DataFrame(future_preds)

if __name__ == "__main__":
    # Quick test
    try:
        engine = ForecasterEngine()
        # Load the features data to use as history
        df = pd.read_csv("data/features_data.csv")
        sample_state = "California"
        history = df[df['State'] == sample_state]
        
        # Predict 8 future weeks recursively
        future_forecast = engine.predict_future(sample_state, history, steps=8)
        print(f"\n8-Week Future Forecast for {sample_state}:")
        print(future_forecast)
    except Exception as e:
        print(f"Forecaster Engine test failed: {e}")
