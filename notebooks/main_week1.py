# main_week1.py
import sys
import os

# Ensure project root is in Python path so src/ can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_load import load_nasa_data
from src.feature_engineering import (
    create_rul,
    create_binary_target,
    create_lag_features,
    create_rolling_features
)

import pandas as pd

# Paths
RAW_PATH = "data/raw/train_FD001.txt"
PROCESSED_PATH = "data/processed/factoryguard_week1.csv"

def main():
    print("Loading data...")
    df = load_nasa_data(RAW_PATH)
    
    print("Creating RUL...")
    df = create_rul(df)
    
    print("Creating binary failure target (24-hour ahead)...")
    df = create_binary_target(df, horizon=24)
    
    # Choose sensors (all or a subset)
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    
    print("Creating lag features...")
    df = create_lag_features(df, sensor_cols)
    
    print("Creating rolling window features...")
    df = create_rolling_features(df, sensor_cols)
    
    print("Dropping NaNs...")
    df = df.dropna()
    
    # Quick leakage sanity checks
    assert df.isna().sum().sum() == 0
    assert df.groupby('engine_id')['cycle'].is_monotonic_increasing.all()
    
    print(f"Saving processed dataset to {PROCESSED_PATH}...")
    df.to_csv(PROCESSED_PATH, index=False)
    

if __name__ == "__main__":
    main()