import pandas as pd
import os

def time_series_split(df: pd.DataFrame, train_size: float = 0.8):
    """
    Split a dataframe into train and validation sets based on time.
    Ensures NO shuffling to preserve the time order.
    """
    # Ensure data is sorted by Date
    df = df.sort_values('Date')
    
    split_idx = int(len(df) * train_size)
    
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    print(f"Total Rows: {len(df)}")
    print(f"Training Rows (80%): {len(train_df)}")
    print(f"Validation Rows (20%): {len(val_df)}")
    print(f"Date Range: {train_df['Date'].min()} to {val_df['Date'].max()}")
    
    return train_df, val_df

if __name__ == "__main__":
    # Test with features data
    data_path = "data/features_data.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        train, val = time_series_split(df)
    else:
        print(f"Data not found at {data_path}. Please run feature_engineering.py first.")
