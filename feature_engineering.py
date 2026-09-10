"""
Feature Engineering Pipeline for Predictive Cloud Resource Optimization
Constructs:
- Time-lag features (t-1, t-2, t-3)
- Moving window rolling statistics (mean, std dev, volatility)
- Rate-of-change (velocity / acceleration)
- Diurnal cyclical encodings (sin/cos of hour)
- Cross-resource interaction terms
- Target: Next-interval CPU Utilization (t+1) for Proactive Lookahead
"""

import numpy as np
import pandas as pd

def build_features(df_raw):
    df = df_raw.copy()
    
    # Sort chronologically to maintain causality
    df = df.sort_values("time_step").reset_index(drop=True)
    
    # 1. Target: Future CPU utilization (t+1)
    df["target_cpu_next"] = df["cpu_util_percent"].shift(-1)
    
    # 2. Lag features (t-1, t-2, t-3)
    df["cpu_lag_1"] = df["cpu_util_percent"].shift(1)
    df["cpu_lag_2"] = df["cpu_util_percent"].shift(2)
    df["cpu_lag_3"] = df["cpu_util_percent"].shift(3)
    
    df["mem_lag_1"] = df["mem_util_percent"].shift(1)
    df["net_in_lag_1"] = df["net_in_mbps"].shift(1)
    
    # 3. Rolling Statistics
    df["cpu_roll_mean_3"] = df["cpu_util_percent"].rolling(window=3).mean()
    df["cpu_roll_mean_6"] = df["cpu_util_percent"].rolling(window=6).mean()
    df["cpu_roll_std_6"] = df["cpu_util_percent"].rolling(window=6).std()
    
    # 4. Workload Momentum / Velocity (Rate of Change)
    df["cpu_velocity"] = df["cpu_util_percent"] - df["cpu_lag_1"]
    df["cpu_acceleration"] = (df["cpu_util_percent"] - df["cpu_lag_1"]) - (df["cpu_lag_1"] - df["cpu_lag_2"])
    
    # 5. Cyclical Time Encodings (smooth 24-hour cycle)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24.0)
    
    # 6. Interaction Terms
    df["network_total_mbps"] = df["net_in_mbps"] + df["net_out_mbps"]
    df["load_intensity"] = (df["cpu_util_percent"] * df["mem_util_percent"]) / 100.0
    
    # Drop boundary NaNs from shifts & rolling windows
    df_clean = df.dropna().reset_index(drop=True)
    
    feature_columns = [
        "cpu_util_percent",
        "mem_util_percent",
        "net_in_mbps",
        "net_out_mbps",
        "disk_io_percent",
        "cpu_lag_1",
        "cpu_lag_2",
        "cpu_lag_3",
        "mem_lag_1",
        "net_in_lag_1",
        "cpu_roll_mean_3",
        "cpu_roll_mean_6",
        "cpu_roll_std_6",
        "cpu_velocity",
        "cpu_acceleration",
        "hour_sin",
        "hour_cos",
        "network_total_mbps",
        "load_intensity"
    ]
    
    return df_clean, feature_columns

def prepare_train_test_split(df_clean, feature_columns, test_ratio=0.2):
    # Time-series chronological split (never random split to prevent data leakage)
    split_idx = int(len(df_clean) * (1 - test_ratio))
    
    train_df = df_clean.iloc[:split_idx]
    test_df = df_clean.iloc[split_idx:]
    
    X_train = train_df[feature_columns]
    y_train = train_df["target_cpu_next"]
    
    X_test = test_df[feature_columns]
    y_test = test_df["target_cpu_next"]
    
    return X_train, X_test, y_train, y_test, train_df, test_df

if __name__ == "__main__":
    df_raw = pd.read_csv("data/cloud_workload_trace.csv")
    df_features, feature_cols = build_features(df_raw)
    X_train, X_test, y_train, y_test, _, _ = prepare_train_test_split(df_features, feature_cols)
    print(f"Feature engineering complete. Total valid samples: {len(df_features)}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
