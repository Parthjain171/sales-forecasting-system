import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def generate_all_charts():
    os.makedirs('charts', exist_ok=True)
    
    # Use forecast_output.csv if it exists, otherwise features_data.csv for history
    forecast_path = "data/forecast_output.csv"
    history_path = "data/features_data.csv"
    
    if not os.path.exists(forecast_path) or not os.path.exists(history_path):
        print("Data files not found.")
        return
        
    f_df = pd.read_csv(forecast_path, parse_dates=['Date'])
    h_df = pd.read_csv(history_path, parse_dates=['Date'])
    
    states = f_df['State'].unique()
    print(f"Generating charts for {len(states)} states...")
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    for state in states:
        try:
            state_hist = h_df[h_df['State'] == state].tail(12)
            state_fore = f_df[f_df['State'] == state]
            
            plt.figure(figsize=(12, 6))
            
            # Plot history
            plt.plot(state_hist['Date'], state_hist['Total'], 
                     color='#6366f1', marker='o', label='Historical Sales')
            
            # Plot forecast
            plt.plot(state_fore['Date'], state_fore['Predicted_Sales'], 
                     color='#ec4899', linestyle='--', marker='s', label='8-Week Forecast')
            
            # Connect the two
            connect_dates = [state_hist['Date'].iloc[-1], state_fore['Date'].iloc[0]]
            connect_values = [state_hist['Total'].iloc[-1], state_fore['Predicted_Sales'].iloc[0]]
            plt.plot(connect_dates, connect_values, color='#ec4899', linestyle='--')
            
            plt.title(f'Sales Forecast Projection — {state}', fontsize=16)
            plt.xlabel('Date')
            plt.ylabel('Sales ($)')
            plt.legend()
            plt.tight_layout()
            
            safe_name = state.replace(' ', '_').lower()
            plt.savefig(f'charts/forecast_{safe_name}.png', dpi=120)
            plt.close()
            print(f"Generated chart for {state}")
        except Exception as e:
            print(f"Error for {state}: {e}")

if __name__ == "__main__":
    generate_all_charts()
