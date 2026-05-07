import pandas as pd
from data_loader import DataLoader

def explore_and_preprocess():
    # Initialize DataLoader and load data
    loader = DataLoader(data_dir="data")
    df = loader.load_excel("Forecasting Case- Study.xlsx")
    
    print("\n--- 22. DataFrame Info ---")
    df.info()
    
    print("\n--- 23. Column Analysis ---")
    # Based on previous head() output:
    # 'Date' is the date column
    # 'Total' is the sales/target value column
    print(f"Date Column: 'Date'")
    print(f"Sales/Target Column: 'Total'")
    
    print("\n--- 24. Converting 'Date' to datetime ---")
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"Date column type after conversion: {df['Date'].dtype}")
    
    print("\n--- 25. Setting 'Date' as Index ---")
    df.set_index('Date', inplace=True)
    print("New Index Head:")
    print(df.head())
    
    print("\n--- 26. Checking for Missing Values ---")
    missing_values = df.isnull().sum()
    print(missing_values)
    
    return df

if __name__ == "__main__":
    explore_and_preprocess()
