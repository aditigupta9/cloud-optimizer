"""
Alibaba Cloud Cluster Trace 2018 - Production Telemetry Generator
Generates realistic multi-day cloud resource metrics:
- CPU Utilization (%)
- Memory Utilization (%)
- Network In / Out (MB/s)
- Disk I/O Utilization (%)

Includes diurnal patterns, batch job bursts, and realistic system noise.
"""

import os
import numpy as np
import pandas as pd

def generate_cloud_trace_data(num_samples=4000, seed=42):
    np.random.seed(seed)
    
    timestamps = np.arange(num_samples)
    
    # 1. Diurnal cycle (24-hour periodicity: 288 samples/day at 5-min intervals)
    day_period = 288
    hour_of_day = (timestamps % day_period) / 12.0 # 0 to 24
    
    # Base daily pattern: peak during work hours, dip during night
    diurnal_wave = np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    base_cpu = 45 + 20 * diurnal_wave
    
    # 2. Weekly variation (7 days cycle = 2016 samples)
    day_of_week = (timestamps // day_period) % 7
    weekend_factor = np.where(day_of_week >= 5, 0.75, 1.0)
    base_cpu = base_cpu * weekend_factor
    
    # 3. Periodic scheduled batch jobs / cron tasks (every 4 hours = 48 steps)
    batch_bursts = np.zeros(num_samples)
    for i in range(0, num_samples, 48):
        duration = np.random.randint(2, 6)
        burst_val = np.random.uniform(15, 30)
        end_idx = min(i + duration, num_samples)
        batch_bursts[i:end_idx] += burst_val
        
    # 4. Spontaneous traffic spikes
    num_spikes = int(num_samples * 0.015)
    spike_indices = np.random.choice(num_samples, size=num_spikes, replace=False)
    spikes = np.zeros(num_samples)
    spikes[spike_indices] = np.random.uniform(20, 35, size=num_spikes)
    
    # 5. Gaussian system noise
    noise = np.random.normal(0, 3.5, size=num_samples)
    
    # Composite CPU
    raw_cpu = base_cpu + batch_bursts + spikes + noise
    cpu_util_percent = np.clip(raw_cpu, 5.0, 99.5)
    
    # Correlated Memory (%)
    mem_base = 35 + 0.45 * cpu_util_percent + np.random.normal(0, 2.0, size=num_samples)
    mem_util_percent = np.clip(mem_base, 15.0, 95.0)
    
    # Correlated Network In (MB/s)
    net_in = (cpu_util_percent / 100.0) * 120.0 + spikes * 2.5 + np.maximum(0, np.random.normal(15, 8, size=num_samples))
    net_in_mbps = np.clip(net_in, 2.0, 350.0)
    
    # Correlated Network Out (MB/s)
    net_out = net_in_mbps * 1.8 + np.maximum(0, np.random.normal(10, 5, size=num_samples))
    net_out_mbps = np.clip(net_out, 3.0, 600.0)
    
    # Correlated Disk I/O (%)
    disk_io = (batch_bursts * 1.5) + (cpu_util_percent * 0.25) + np.maximum(0, np.random.normal(10, 4, size=num_samples))
    disk_io_percent = np.clip(disk_io, 1.0, 98.0)
    
    df = pd.DataFrame({
        "time_step": timestamps,
        "hour_of_day": np.round(hour_of_day, 2),
        "day_of_week": day_of_week,
        "cpu_util_percent": np.round(cpu_util_percent, 2),
        "mem_util_percent": np.round(mem_util_percent, 2),
        "net_in_mbps": np.round(net_in_mbps, 2),
        "net_out_mbps": np.round(net_out_mbps, 2),
        "disk_io_percent": np.round(disk_io_percent, 2)
    })
    
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "cloud_workload_trace.csv")
    df = generate_cloud_trace_data(num_samples=4000)
    df.to_csv(out_path, index=False)
    print(f"Generated realistic cloud workload trace dataset with {len(df)} records at {out_path}")
    print(df.head(5))
