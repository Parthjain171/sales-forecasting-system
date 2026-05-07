import pandas as pd
from data_loader import DataLoader
import os

def preprocess_pipeline():
    # Load data
    loader = DataLoader(data_dir="data")
    df = loader.load_excel("Forecasting Case- Study.xlsx")
    
    # Basic preprocessing from previous steps
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.ffill()
    
    print("\n--- 28. Unique States ---")
    unique_states = df['State'].unique()
    print(f"Number of unique states: {len(unique_states)}")
    print(unique_states)
    
    print("\n--- 29 & 30. Grouping and Resampling to Weekly ('W') ---")
    # Grouping by State and Category (since there are multiple categories per state)
    # The user asked for "by state", but let's see if grouping just by state aggregates categories.
    # We will resample 'Total' to weekly sum.
    
    # We use groupby(['State', 'Category']) to ensure we don't mix products, 
    # but the prompt asked for "each state has its own time series".
    # I'll do it by State and see the result.
    
    cleaned_df = df.groupby('State').resample('W')['Total'].sum().reset_index()
    
    print("\n--- 31. Sample Output for One State (e.g., Alabama) ---")
    alabama_sample = cleaned_df[cleaned_df['State'] == 'Alabama'].head()
    print(alabama_sample)
    
    print("\n--- 32. Saving Cleaned Data ---")
    output_path = os.path.join("data", "cleaned_data.csv")
    cleaned_df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    
    return cleaned_df

if __name__ == "__main__":
    preprocess_pipeline()
