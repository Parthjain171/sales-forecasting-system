from fastapi import FastAPI, HTTPException # Reloaded to sync state list

from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import json
import joblib
import sys

# Add src to sys.path to resolve model dependencies
# Moving up from: sales-forecasting-system/api/main.py -> sales-forecasting-system
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

# Standard ForecasterEngine implementation (Self-contained)
class ForecasterEngine:
    def __init__(self, mapping_path, model_dir, preload=False):
        self.mapping_path = mapping_path
        self.model_dir = model_dir
        self.best_models = self._load_mapping()
        self.loaded_models = {}
        if preload:
            self.load_all_models()

    def _load_mapping(self):
        if not os.path.exists(self.mapping_path):
            raise FileNotFoundError(f"Model mapping not found at {self.mapping_path}")
        with open(self.mapping_path, 'r') as f:
            return json.load(f)

    def load_model_for_state(self, state: str):
        if state in self.loaded_models:
            return self.loaded_models[state]
        if state not in self.best_models:
            raise ValueError(f"No model found for state: {state}")
        model_name = self.best_models[state]
        model_path = os.path.join(self.model_dir, f"{state}_{model_name}.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model = joblib.load(model_path)
        self.loaded_models[state] = model
        return model

    def load_all_models(self):
        print(f"Pre-loading {len(self.best_models)} models...")
        for state in self.best_models.keys():
            try:
                self.load_model_for_state(state)
            except:
                pass

    def predict_future(self, state, history_df, steps=8):
        model = self.load_model_for_state(state)
        current_data = history_df.copy().tail(30)
        future_preds = []
        last_date = pd.to_datetime(current_data['Date'].max())
        for i in range(steps):
            next_date = last_date + pd.Timedelta(weeks=i+1)
            new_row = {
                'Date': next_date, 'Year': next_date.year, 'Month': next_date.month,
                'Week': int(next_date.isocalendar().week), 'Quarter': next_date.quarter,
                'Lag_1': current_data['Total'].iloc[-1],
                'Lag_7': current_data['Total'].iloc[-7] if len(current_data) >= 7 else 0,
                'Lag_30': current_data['Total'].iloc[-30] if len(current_data) >= 30 else 0,
                'Rolling_Mean_4': current_data['Total'].tail(4).mean(),
                'Rolling_Std_4': current_data['Total'].tail(4).std() if len(current_data) >= 4 else 0,
                'Is_US_Holiday': 0, 'Is_Indian_Holiday': 0, 'Total': 0
            }
            feat_df = pd.DataFrame([new_row])
            feature_order = ['Year', 'Month', 'Week', 'Quarter', 'Is_US_Holiday', 'Is_Indian_Holiday', 
                             'Lag_1', 'Lag_7', 'Lag_30', 'Rolling_Mean_4', 'Rolling_Std_4']
            X = feat_df[feature_order]
            pred = model.predict(X)[0]
            new_row['Total'] = pred
            current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
            future_preds.append({'Date': next_date, 'Predicted_Sales': pred})
        return pd.DataFrame(future_preds)

app = FastAPI(
    title='Sales Forecasting API',
    description='8-week ML forecast per Indian state',
    version='1.0.0'
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Setup paths relative to api/ directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPPING_PATH = os.path.join(BASE_DIR, "data", "best_models.json")
MODEL_DIR = os.path.join(BASE_DIR, "data", "models")
HISTORY_PATH = os.path.join(BASE_DIR, "data", "features_data.csv")
FORECAST_PATH = os.path.join(BASE_DIR, "data", "forecast_output.csv")
LEADERBOARD_PATH = os.path.join(BASE_DIR, "data", "global_leaderboard.csv")
COMPARISON_PATH = os.path.join(BASE_DIR, "data", "full_model_comparison.csv")

# Initialize Engine
engine = None
history_df = pd.DataFrame()
leaderboard_df = pd.DataFrame()
precalculated_forecasts = pd.DataFrame()
full_comparison_df = pd.DataFrame()

@app.on_event("startup")
async def startup_event():
    global engine, history_df, leaderboard_df, precalculated_forecasts, full_comparison_df
    try:
        print("Starting API engine...")
        engine = ForecasterEngine(MAPPING_PATH, MODEL_DIR, preload=False)
        print(f"Loading history data from {HISTORY_PATH}...")
        history_df = pd.read_csv(HISTORY_PATH)
        print(f"Loading pre-calculated forecasts from {FORECAST_PATH}...")
        if os.path.exists(FORECAST_PATH):
            precalculated_forecasts = pd.read_csv(FORECAST_PATH)
        print(f"Loading comparison data from {COMPARISON_PATH}...")
        if os.path.exists(COMPARISON_PATH):
            full_comparison_df = pd.read_csv(COMPARISON_PATH)
        print(f"Loading leaderboard from {LEADERBOARD_PATH}...")
        leaderboard_df = pd.read_csv(LEADERBOARD_PATH)
        print("API initialized successfully with new paths.")
    except Exception as e:
        print(f"Startup error: {e}")

# Root Endpoint: Returns welcome message and available routes
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Astra Sales Forecasting API",
        "endpoints": {
            "root": "/",
            "states": "/states",
            "model_details": "/models/{state}",
            "forecast": "/forecast/{state}",
            "documentation": "/docs"
        }
    }

# Summary Endpoint: Returns system-wide metrics (total states, avg MAE, best overall model)
@app.get("/summary")
def get_summary():
    if leaderboard_df.empty: raise HTTPException(status_code=500, detail="Leaderboard not loaded")
    
    total_states = len(leaderboard_df)
    avg_mae = float(leaderboard_df['MAE'].mean())
    best_overall = leaderboard_df['Best_Model'].mode()[0]
    
    return {
        "total_states": total_states,
        "best_overall_model": best_overall,
        "average_mae": avg_mae
    }

# States Endpoint: Returns an alphabetically sorted list of all valid state names
@app.get("/states")
def get_states():
    if not engine: raise HTTPException(status_code=500, detail="Engine not ready")
    return {"states": sorted(list(engine.best_models.keys()))}

# Comparison Endpoint: Returns side-by-side MAE comparison for all 4 models for a specific state
@app.get("/models/compare/{state}")
def compare_models_endpoint(state: str):
    state = state.strip('"').strip()
    if full_comparison_df.empty: raise HTTPException(status_code=500, detail="Comparison data not loaded")
    state_results = full_comparison_df[full_comparison_df['State'].str.lower() == state.lower()]
    if state_results.empty:
        raise HTTPException(status_code=404, detail="State not found. Call /states to see valid names")
    
    comparisons = []
    for _, row in state_results.iterrows():
        comparisons.append({
            "model": row['Model'],
            "mae": float(row['MAE'])
        })
    
    return {
        "state": state_results.iloc[0]['State'],
        "comparisons": comparisons
    }

# Model Details Endpoint: Returns the best model and its MAE score for a specific state
@app.get("/models/{state}")
def get_model_details(state: str):
    state = state.strip('"').strip()
    if leaderboard_df.empty: raise HTTPException(status_code=500, detail="Data not loaded")
    state_info = leaderboard_df[leaderboard_df['State'].str.lower() == state.lower()]
    if state_info.empty:
        raise HTTPException(status_code=404, detail="State not found. Call /states to see valid names")
    row = state_info.iloc[0]
    return {"state": row['State'], "best_model": row['Best_Model'], "mae": float(row['MAE'])}

# Forecast Endpoint: Returns 8-week predictions, model type, and accuracy metrics for a specific state
@app.get("/forecast/{state}")
def get_forecast(state: str):
    state = state.strip('"').strip()
    if not engine: raise HTTPException(status_code=500, detail="Engine not ready")
    
    # Get MAE score from leaderboard
    state_info = leaderboard_df[leaderboard_df['State'].str.lower() == state.lower()]
    if state_info.empty:
        raise HTTPException(status_code=404, detail="State not found. Call /states to see valid names")
        
    mae_score = float(state_info.iloc[0]['MAE'])
    actual_state_name = state_info.iloc[0]['State']
    
    # 1. Prepare history and check cache
    state_history = history_df[history_df['State'].str.lower() == state.lower()]
    if state_history.empty: raise HTTPException(status_code=404, detail="No history data")
    
    preds = []
    source = "precalculated_cache"
    
    if not precalculated_forecasts.empty:
        state_forecast = precalculated_forecasts[precalculated_forecasts['State'].str.lower() == state.lower()]
        if not state_forecast.empty:
            for i, (_, row) in enumerate(state_forecast.iterrows()):
                preds.append({"week": i + 1, "date": row['Date'], "predicted_sales": float(row['Predicted_Sales'])})
    
    if not preds:
        source = "real_time_inference"
        actual_name = next((s for s in engine.best_models.keys() if s.lower() == state.lower()), actual_state_name)
        try:
            forecast = engine.predict_future(actual_name, state_history, steps=8)
            for i, (_, row) in enumerate(forecast.iterrows()):
                preds.append({"week": i + 1, "date": row['Date'].strftime('%Y-%m-%d'), "predicted_sales": float(row['Predicted_Sales'])})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # 2. Get history subset (last 12 weeks)
    history_subset = state_history.tail(12)[['Date', 'Total']].to_dict('records')
    
    return {
        "state": actual_state_name, 
        "best_model": engine.best_models.get(actual_state_name, "XGBoost"), 
        "mae_score": mae_score,
        "history": history_subset,
        "predictions": preds,
        "source": source
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
