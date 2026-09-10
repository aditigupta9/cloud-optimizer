"""
Machine Learning Inference Engine for Cloud Resource Optimization
Loads serialized production model and performs multivariate time-series forecasting.
Includes rolling window state buffer for live dynamic feature engineering.
"""

import os
import time
from datetime import datetime
import json
import joblib
import numpy as np
import pandas as pd
import psutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "models", "model_metrics.json")
DATA_PATH = os.path.join(BASE_DIR, "data", "cloud_workload_trace.csv")

# Global state
_model = None
_scaler = None
_metrics = None
_trace_df = None
_feature_cols = None

def load_inference_artifacts():
    global _model, _scaler, _metrics, _trace_df, _feature_cols
    
    if _model is None and os.path.exists(MODEL_PATH):
        try:
            _model = joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"[ML WARNING] Could not unpickle model ({e}). Retraining in-place...")
            try:
                from sklearn.ensemble import GradientBoostingRegressor
                from feature_engineering import build_feature_pipeline
                from generate_dataset import generate_cloud_trace
                if not os.path.exists(DATA_PATH):
                    generate_cloud_trace(DATA_PATH, n_samples=4000)
                X_train, X_test, y_train, y_test, feat_cols = build_feature_pipeline(DATA_PATH)
                _feature_cols = feat_cols
                gbr = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42)
                gbr.fit(X_train, y_train)
                _model = gbr
                try:
                    joblib.dump(_model, MODEL_PATH)
                except:
                    pass
                print("[ML SUCCESS] Dynamic model re-trained successfully in-place!")
            except Exception as re_err:
                print(f"[ML ERROR] Failed to retrain model: {re_err}")
        
    if _scaler is None and os.path.exists(SCALER_PATH):
        _scaler = joblib.load(SCALER_PATH)
        
    if _metrics is None and os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            _metrics = json.load(f)
            _feature_cols = _metrics.get("feature_columns", [])
            
    if _trace_df is None and os.path.exists(DATA_PATH):
        _trace_df = pd.read_csv(DATA_PATH)

# Load artifacts on import
load_inference_artifacts()

class TelemetryHistoryBuffer:
    """Maintains a rolling window of recent machine readings to compute live lag/rolling features."""
    def __init__(self, maxlen=15):
        self.maxlen = maxlen
        self.buffer = []
        
        # Initialize with realistic baseline readings if empty
        if _trace_df is not None and len(_trace_df) > 20:
            for idx in range(10):
                row = _trace_df.iloc[idx].to_dict()
                self.buffer.append(row)
                
    def add_reading(self, reading):
        self.buffer.append(reading)
        if len(self.buffer) > self.maxlen:
            self.buffer.pop(0)
            
    def compute_feature_vector(self, current_reading=None):
        if current_reading:
            self.add_reading(current_reading)
            
        if len(self.buffer) < 4:
            raise ValueError("Buffer must have at least 4 historical samples to compute lags.")
            
        recent = self.buffer[-1]
        lag_1 = self.buffer[-2]
        lag_2 = self.buffer[-3]
        lag_3 = self.buffer[-4]
        
        cpu_series = [b["cpu_util_percent"] for b in self.buffer]
        
        cpu_curr = float(recent["cpu_util_percent"])
        mem_curr = float(recent.get("mem_util_percent", 50.0))
        net_in_curr = float(recent.get("net_in_mbps", 50.0))
        net_out_curr = float(recent.get("net_out_mbps", 80.0))
        disk_io_curr = float(recent.get("disk_io_percent", 20.0))
        hour_curr = float(recent.get("hour_of_day", 12.0))
        
        cpu_lag_1 = float(lag_1["cpu_util_percent"])
        cpu_lag_2 = float(lag_2["cpu_util_percent"])
        cpu_lag_3 = float(lag_3["cpu_util_percent"])
        
        mem_lag_1 = float(lag_1.get("mem_util_percent", mem_curr))
        net_in_lag_1 = float(lag_1.get("net_in_mbps", net_in_curr))
        
        # Rolling stats
        roll_3 = np.mean(cpu_series[-3:])
        roll_6 = np.mean(cpu_series[-6:]) if len(cpu_series) >= 6 else roll_3
        roll_std_6 = np.std(cpu_series[-6:]) if len(cpu_series) >= 6 else 2.0
        if np.isnan(roll_std_6) or roll_std_6 == 0:
            roll_std_6 = 1.5
            
        velocity = cpu_curr - cpu_lag_1
        acceleration = velocity - (cpu_lag_1 - cpu_lag_2)
        
        hour_sin = np.sin(2 * np.pi * hour_curr / 24.0)
        hour_cos = np.cos(2 * np.pi * hour_curr / 24.0)
        
        net_total = net_in_curr + net_out_curr
        load_intensity = (cpu_curr * mem_curr) / 100.0
        
        features = {
            "cpu_util_percent": cpu_curr,
            "mem_util_percent": mem_curr,
            "net_in_mbps": net_in_curr,
            "net_out_mbps": net_out_curr,
            "disk_io_percent": disk_io_curr,
            "cpu_lag_1": cpu_lag_1,
            "cpu_lag_2": cpu_lag_2,
            "cpu_lag_3": cpu_lag_3,
            "mem_lag_1": mem_lag_1,
            "net_in_lag_1": net_in_lag_1,
            "cpu_roll_mean_3": roll_3,
            "cpu_roll_mean_6": roll_6,
            "cpu_roll_std_6": roll_std_6,
            "cpu_velocity": velocity,
            "cpu_acceleration": acceleration,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "network_total_mbps": net_total,
            "load_intensity": load_intensity
        }
        return features

_telemetry_buffer = TelemetryHistoryBuffer()

def predict_future_cpu(features_dict=None):
    """
    Predicts next-step CPU utilization given current and historical feature metrics.
    Returns: dict with predicted_cpu, confidence interval, and current_metrics.
    """
    load_inference_artifacts()
    
    if _model is None:
        raise RuntimeError("Trained machine learning model not found in models/best_model.joblib. Please run train_models.py first.")
        
    if features_dict is None:
        features_dict = _telemetry_buffer.compute_feature_vector()
        
    # Order features according to training feature columns
    col_order = _feature_cols or list(features_dict.keys())
    input_vector = [features_dict.get(c, 0.0) for c in col_order]
    
    X_input = pd.DataFrame([input_vector], columns=col_order)
    raw_pred = _model.predict(X_input)[0]
    
    # Clip to valid operational range [5%, 99.5%]
    predicted_cpu = float(np.clip(raw_pred, 5.0, 99.5))
    
    # Confidence bounds based on model RMSE (~5.8%)
    rmse_val = 5.8
    if _metrics and "metrics_summary" in _metrics:
        best_name = _metrics.get("best_model_name", "")
        rmse_val = _metrics["metrics_summary"].get(best_name, {}).get("RMSE", 5.8)
        
    ci_lower = max(0.0, predicted_cpu - 1.96 * (rmse_val / 2.0))
    ci_upper = min(100.0, predicted_cpu + 1.96 * (rmse_val / 2.0))
    
    return {
        "predicted_cpu": round(predicted_cpu, 2),
        "ci_lower": round(ci_lower, 2),
        "ci_upper": round(ci_upper, 2),
        "current_cpu": round(features_dict.get("cpu_util_percent", 50.0), 2),
        "current_mem": round(features_dict.get("mem_util_percent", 50.0), 2),
        "current_net": round(features_dict.get("network_total_mbps", 100.0), 2),
        "current_disk": round(features_dict.get("disk_io_percent", 20.0), 2),
        "model_name": _metrics.get("best_model_name", "Gradient Boosting") if _metrics else "Ensemble Regressor"
    }

def predict_cpu(time_input):
    """
    Backward-compatible prediction function for legacy app_flask.py calls.
    Maps time_input step to trace data, updates rolling buffer, and returns realistic predicted CPU.
    """
    load_inference_artifacts()
    
    try:
        step = int(time_input)
    except (ValueError, TypeError):
        step = 1
        
    # Get telemetry record for this time step from trace dataset
    if _trace_df is not None and len(_trace_df) > 0:
        row_idx = (step - 1) % len(_trace_df)
        record = _trace_df.iloc[row_idx].to_dict()
    else:
        # Fallback synthetic record
        record = {
            "time_step": step,
            "hour_of_day": (step * 0.083) % 24,
            "cpu_util_percent": 45.0 + 25.0 * np.sin(step / 10.0),
            "mem_util_percent": 50.0,
            "net_in_mbps": 40.0,
            "net_out_mbps": 60.0,
            "disk_io_percent": 25.0
        }
        
    _telemetry_buffer.add_reading(record)
    pred_result = predict_future_cpu()
    return pred_result["predicted_cpu"]

def get_telemetry_record(step_index):
    """Retrieves full telemetry attributes for a given timeline step."""
    load_inference_artifacts()
    if _trace_df is not None and len(_trace_df) > 0:
        row_idx = (int(step_index) - 1) % len(_trace_df)
        return _trace_df.iloc[row_idx].to_dict()
    return None

def get_model_benchmarks():
    """Returns comparative metrics dictionary."""
    load_inference_artifacts()
    if _metrics:
        return _metrics
    return {}

_last_net_bytes = None
_last_net_time = None

def get_live_hardware_reading():
    """Reads live hardware CPU, Memory, Network, and Disk from the current system."""
    global _last_net_bytes, _last_net_time
    
    cpu = float(psutil.cpu_percent(interval=0.15))
    mem = float(psutil.virtual_memory().percent)
    try:
        disk = float(psutil.disk_usage('C:').percent)
    except:
        disk = 40.0
        
    net = psutil.net_io_counters()
    total_bytes = net.bytes_recv + net.bytes_sent
    now = time.time()
    
    if _last_net_bytes is not None and _last_net_time is not None:
        elapsed = max(0.1, now - _last_net_time)
        net_rate_mb = max(1.0, (total_bytes - _last_net_bytes) / (1024 * 1024 * elapsed))
    else:
        net_rate_mb = 12.5
        
    _last_net_bytes = total_bytes
    _last_net_time = now
    
    dt = datetime.now()
    hour = dt.hour + (dt.minute / 60.0)
    
    reading = {
        "time_step": int(time.time()),
        "hour_of_day": round(hour, 2),
        "cpu_util_percent": round(cpu, 2),
        "mem_util_percent": round(mem, 2),
        "net_in_mbps": round(net_rate_mb * 0.45, 2),
        "net_out_mbps": round(net_rate_mb * 0.55, 2),
        "disk_io_percent": round(disk, 2),
        "source": "live_laptop_hardware"
    }
    return reading

if __name__ == "__main__":
    print("Testing ML Model Inference Pipeline...")
    res = predict_future_cpu()
    print("Inference Result:", res)
    print("Predict CPU for time=10:", predict_cpu(10))
    print("Predict CPU for time=50:", predict_cpu(50))
