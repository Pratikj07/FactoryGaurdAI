import pandas as pd

def create_rul(df: pd.DataFrame) -> pd.DataFrame:
    max_cycle = df.groupby('engine_id')['cycle'].max()
    df = df.merge(max_cycle.rename('max_cycle'), on='engine_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    return df


def create_binary_target(df: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    df['failure_24h'] = (df['RUL'] <= horizon).astype(int)
    return df


def create_lag_features(df: pd.DataFrame, sensor_cols, lags=[1,2,24]) -> pd.DataFrame:
    for col in sensor_cols:
        for lag in lags:
            df[f'{col}_lag{lag}'] = df.groupby('engine_id')[col].shift(lag)
    return df


def create_rolling_features(df: pd.DataFrame, sensor_cols, windows=[4,8,24]) -> pd.DataFrame:
    for col in sensor_cols:
        for w in windows:
            df[f'{col}_roll_mean_{w}'] = (
                df.groupby('engine_id')[col]
                .rolling(window=w)
                .mean()
                .shift(1)
                .reset_index(level=0, drop=True)
            )
            
            df[f'{col}_roll_std_{w}'] = (
                df.groupby('engine_id')[col]
                .rolling(window=w)
                .std()
                .shift(1)
                .reset_index(level=0, drop=True)
            )
    return df