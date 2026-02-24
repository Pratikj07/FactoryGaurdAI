# src/data_load.py
import pandas as pd

def load_nasa_data(filepath: str) -> pd.DataFrame:
    """
    Load NASA C-MAPSS FD001 dataset and return a structured DataFrame.
    
    Args:
        filepath (str): Path to train_FD001.txt

    Returns:
        pd.DataFrame: Loaded DataFrame with proper column names.
    """
    
    # Define columns
    op_settings = [f'op_setting_{i}' for i in range(1, 4)]
    sensors = [f'sensor_{i}' for i in range(1, 22)]
    columns = ['engine_id', 'cycle'] + op_settings + sensors
    
    # Load CSV (space-separated)
    df = pd.read_csv(filepath, sep=r"\s+", header=None)
    df.columns = columns
    
    # Optional: convert cycle to simulated hourly timestamp
    df['timestamp'] = df.groupby('engine_id').cumcount()
    
    return df