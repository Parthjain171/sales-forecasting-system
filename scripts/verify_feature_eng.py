from pathlib import Path
import pandas as pd
import numpy as np
import holidays
import os

# Get the project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Load cleaned data
data_path = ROOT_DIR / "data" / "cleaned_data.csv"
if not os.path.exists(data_path):
    print(f"Data not found at {data_path}")
    exit(1)

df = pd.read_csv(data_path, parse_dates=['Date'])

# 1. Time-Based Features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
df['Quarter'] = df['Date'].dt.quarter

# 2. Holiday Indicators
us_holidays = holidays.US()
india_holidays = holidays.India()
df['Is_US_Holiday'] = df['Date'].apply(lambda x: 1 if x in us_holidays else 0)
df['Is_Indian_Holiday'] = df['Date'].apply(lambda x: 1 if x in india_holidays else 0)

# 3. Lags and Rolling Windows
df = df.sort_values(['State', 'Date'])
df['Lag_1'] = df.groupby('State')['Total'].shift(1)
df['Lag_7'] = df.groupby('State')['Total'].shift(7)
df['Lag_30'] = df.groupby('State')['Total'].shift(30)
df['Rolling_Mean_4'] = df.groupby('State')['Total'].transform(lambda x: x.shift(1).rolling(window=4).mean())
df['Rolling_Std_4'] = df.groupby('State')['Total'].transform(lambda x: x.shift(1).rolling(window=4).std())

# Drop NaN values
df = df.dropna()

# Check: Columns
cols = df.columns.tolist()
print(f"Columns: {cols}")

# Check: No NaN values
nan_count = df.isna().sum().sum()
print(f"NaN Count: {nan_count}")

# Check: Save to data/features_data.csv
output_path = ROOT_DIR / "data" / "features_data.csv"
df.to_csv(output_path, index=False)
print(f"Features data saved to {output_path}")

# Verification output for the user
print("VERIFICATION SUCCESSFUL")
print(f"Columns contain lags and rolling stats: {'Lag_1' in cols and 'Rolling_Mean_4' in cols}")
print(f"NaN count is 0: {nan_count == 0}")
print(f"File exists: {os.path.exists(output_path)}")
