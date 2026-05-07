import pandas as pd
import os
from typing import Optional

class DataLoader:
    """
    A utility class to load forecasting data from various formats.
    """
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir

    def load_excel(self, filename: str, sheet_name: Optional[str] = 0) -> pd.DataFrame:
        """
        Load data from an Excel file.
        
        Args:
            filename: The name of the file in the data directory.
            sheet_name: The name or index of the sheet to load.
            
        Returns:
            pd.DataFrame: The loaded data.
        """
        file_path = os.path.join(self.data_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        print(f"Loading data from {file_path}...")
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
            print(f"Successfully loaded {len(df)} rows.")
            return df
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            raise

if __name__ == "__main__":
    # Quick test
    loader = DataLoader()
    try:
        # Assuming the file we just moved is the one to load
        data = loader.load_excel("Forecasting Case- Study.xlsx")
        print(data.head())
    except Exception as e:
        print(f"Test load failed: {e}")
