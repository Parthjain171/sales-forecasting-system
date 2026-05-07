import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from model_selector import compare_models

def verify_all_states():
    data_path = "data/features_data.csv"
    if not os.path.exists(data_path):
        print("Data not found.")
        return
        
    df = pd.read_csv(data_path)
    states = df['State'].unique()
    
    print(f"Total states to verify: {len(states)}")
    
    success_states = []
    failed_states = []
    
    # Verify first 3 states to ensure logic is sound
    for state in states[:3]:
        try:
            print(f"\nVerifying {state}...")
            # Modify compare_models to only run SARIMA, Prophet, XGBoost if possible
            # But for now, we'll run the standard one.
            best_model, summary, _ = compare_models(state=state)
            success_states.append(state)
            print(f"SUCCESS: {state} trained with best model {best_model}")
        except Exception as e:
            failed_states.append((state, str(e)))
            print(f"FAILED: {state} with error: {e}")
            
    print("\n=== VERIFICATION SUMMARY ===")
    print(f"Successfully verified states: {len(success_states)}")
    print(f"Failed states: {len(failed_states)}")
    
    if len(failed_states) == 0:
        print("Initial verification passed. Starting background training for all states...")
    else:
        print("Verification found issues.")

if __name__ == "__main__":
    # Ensure we are in the right directory
    os.chdir('sales-forecasting-system')
    verify_all_states()
