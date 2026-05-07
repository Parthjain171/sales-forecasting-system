import pandas as pd
import os
import json
import joblib
from model_selector import compare_models

def run_leaderboard_for_all_states():
    # Load data
    data_path = "data/features_data.csv"
    if not os.path.exists(data_path):
        print("Please run feature_engineering.py first.")
        return
        
    df = pd.read_csv(data_path)
    states = df['State'].unique()
    
    # Create directory for model objects
    model_dir = "data/models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    print(f"Starting comparison for {len(states)} states...")
    print("WARNING: This process may take a while as it trains 4 models per state.")
    
    leaderboard = []
    
    # For demonstration, let's process the first 5 states to save time, 
    # but the logic works for all.
    # To run for ALL states, change 'states[:5]' to 'states'
    for state in states:
        try:
            best_model, summary, model_obj = compare_models(state=state)
            best_mae = summary.iloc[0]['MAE']
            leaderboard.append({
                'State': state,
                'Best_Model': best_model,
                'MAE': best_mae
            })
            
            # Save the winning model object
            model_path = os.path.join(model_dir, f"{state}_{best_model}.joblib")
            joblib.dump(model_obj, model_path)
            print(f"Saved best model for {state} to {model_path}")
            
            # Save incremental results
            pd.DataFrame(leaderboard).to_csv("data/global_leaderboard.csv", index=False)
            best_models_map = {item['State']: item['Best_Model'] for item in leaderboard}
            with open("data/best_models.json", 'w') as f:
                json.dump(best_models_map, f, indent=4)
            
        except Exception as e:
            print(f"Error processing {state}: {e}")
            
    leaderboard_df = pd.DataFrame(leaderboard)
    print("\n=== GLOBAL LEADERBOARD (Top 5 States Sample) ===")
    print(leaderboard_df)
    
    # Save results as CSV
    leaderboard_df.to_csv("data/global_leaderboard.csv", index=False)
    
    # Save the {state: best_model_name} mapping to a JSON file
    best_models_map = {item['State']: item['Best_Model'] for item in leaderboard}
    json_path = "data/best_models.json"
    with open(json_path, 'w') as f:
        json.dump(best_models_map, f, indent=4)
        
    print(f"\nBest models mapping saved to {json_path}")
    print(f"Leaderboard saved to data/global_leaderboard.csv")

if __name__ == "__main__":
    run_leaderboard_for_all_states()
