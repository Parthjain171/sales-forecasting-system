from pathlib import Path
import pandas as pd
import os
import sys

# Get the project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / 'src'))

from forecaster import ForecasterEngine

def generate_forecast_output():
    data_path = ROOT_DIR / "data" / "features_data.csv"
    if not os.path.exists(data_path):
        print("Data not found.")
        return
        
    df = pd.read_csv(data_path)
    states = df['State'].unique()
    
    print(f"Generating 8-week forecast for {len(states)} states...")
    
    engine = ForecasterEngine()
    all_forecasts = []
    
    for state in states:
        try:
            print(f"Forecasting {state}...")
            state_history = df[df['State'] == state]
            forecast_df = engine.predict_future(state, state_history, steps=8)
            forecast_df['State'] = state
            all_forecasts.append(forecast_df)
        except Exception as e:
            print(f"Error forecasting {state}: {e}")
            
    final_df = pd.concat(all_forecasts, ignore_index=True)
    # Reorder columns
    final_df = final_df[['State', 'Date', 'Predicted_Sales']]
    
    output_path = ROOT_DIR / "data" / "forecast_output.csv"
    final_df.to_csv(output_path, index=False)
    print(f"\nFinal forecast output saved to {output_path}")
    
    # Verification
    print(f"Total rows: {len(final_df)}")
    print(f"Rows per state: {len(final_df) / len(states)}")

if __name__ == "__main__":
    generate_forecast_output()
