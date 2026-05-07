import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from model_selector import compare_models

def generate_full_comparison_report():
    os.chdir('sales-forecasting-system')
    data_path = "data/features_data.csv"
    if not os.path.exists(data_path):
        print("Data not found.")
        return
        
    df = pd.read_csv(data_path)
    states = df['State'].unique()
    
    print(f"Generating full comparison for {len(states)} states...")
    
    all_results = []
    
    for state in states:
        try:
            print(f"Processing {state}...")
            # compare_models returns: best_model_name, summary_df, best_model_obj
            _, summary_df, _ = compare_models(state=state)
            
            # summary_df columns: ['Model', 'RMSE', 'MAE', 'ModelObject']
            for _, row in summary_df.iterrows():
                all_results.append({
                    'State': state,
                    'Model': row['Model'],
                    'RMSE': row['RMSE'],
                    'MAE': row['MAE']
                })
        except Exception as e:
            print(f"Error processing {state}: {e}")
            
    full_report_df = pd.DataFrame(all_results)
    full_report_df.to_csv("data/full_model_comparison.csv", index=False)
    print("\nFull model comparison report saved to data/full_model_comparison.csv")
    
    # Print sample for user
    print("\n=== SAMPLE: Full Comparison Table (Alabama) ===")
    print(full_report_df[full_report_df['State'] == 'Alabama'])

if __name__ == "__main__":
    generate_full_comparison_report()
