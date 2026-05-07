import pandas as pd
import holidays
import os

def create_state_lags(state_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply lag features to a single state's time series.
    """
    state_df = state_df.sort_values('Date')
    
    # 35. Add lag feature: sales 1 week ago
    state_df['Lag_1'] = state_df['Total'].shift(1)
    
    # 36. Add lag feature: sales 7 weeks ago
    state_df['Lag_7'] = state_df['Total'].shift(7)
    
    # 37. Add lag feature: sales 30 weeks ago
    state_df['Lag_30'] = state_df['Total'].shift(30)
    
    return state_df

def add_features(input_path="data/cleaned_data.csv", output_path="data/features_data.csv"):
    """
    Load cleaned data and add time-series features.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Cleaned data not found at {input_path}")
        
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    print("Adding time-based features...")
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    # 42. Add quarter column
    df['Quarter'] = df['Date'].dt.quarter
    
    print("Adding Holiday flags...")
    us_holidays = holidays.US()
    # 43. Add Indian holiday flag
    india_holidays = holidays.India()
    
    df['Is_US_Holiday'] = df['Date'].apply(lambda x: 1 if x in us_holidays else 0)
    df['Is_Indian_Holiday'] = df['Date'].apply(lambda x: 1 if x in india_holidays else 0)
    
    print("Adding Lag features (per State)...")
    df['Lag_1'] = df.groupby('State')['Total'].shift(1)
    df['Lag_7'] = df.groupby('State')['Total'].shift(7)
    df['Lag_30'] = df.groupby('State')['Total'].shift(30)
    
    print("Adding Rolling features (per State)...")
    df['Rolling_Mean_4'] = df.groupby('State')['Total'].transform(lambda x: x.rolling(window=4).mean())
    df['Rolling_Std_4'] = df.groupby('State')['Total'].transform(lambda x: x.rolling(window=4).std())
    
    # 44. Drop rows with NaN values created by lag features
    print("Dropping rows with NaN values...")
    df = df.dropna()
    
    print(f"Features added. Saving to {output_path}...")
    df.to_csv(output_path, index=False)
    
    # 45. Test the function on one state and print output to verify
    sample_state = df['State'].unique()[0]
    print(f"\nVerification for {sample_state}:")
    cols = ['Date', 'Total', 'Quarter', 'Is_Indian_Holiday', 'Lag_1', 'Rolling_Mean_4']
    print(df[df['State'] == sample_state][cols].head(10))
    
    return df

if __name__ == "__main__":
    add_features()
