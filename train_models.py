"""
Machine Learning Training & Comparative Benchmarking Pipeline
Trains:
1. Baseline Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

Computes:
- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

Generates academic / viva-ready visualization charts and serializes the best model.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from feature_engineering import build_features, prepare_train_test_split

def train_and_benchmark():
    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.join("static", "images"), exist_ok=True)
    
    # 1. Load Data & Build Features
    data_path = os.path.join("data", "cloud_workload_trace.csv")
    df_raw = pd.read_csv(data_path)
    df_clean, feature_cols = build_features(df_raw)
    
    X_train, X_test, y_train, y_test, _, test_df = prepare_train_test_split(df_clean, feature_cols)
    
    # 2. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Define candidate models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=120, max_depth=12, min_samples_split=4, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=5, random_state=42)
    }
    
    results = {}
    predictions = {}
    fitted_models = {}
    
    print("\n=======================================================")
    print("      CLOUD RESOURCE PREDICTION MODEL BENCHMARKING    ")
    print("=======================================================")
    
    for name, model in models.items():
        print(f"\n[Training] {name}...")
        # Train on scaled features for Linear Regression, unscaled works for trees but scaled works universally
        if name == "Linear Regression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mape = np.mean(np.abs((y_test - preds) / np.maximum(y_test, 1.0))) * 100.0
        
        results[name] = {
            "R2_Score": round(float(r2), 4),
            "MAE": round(float(mae), 4),
            "RMSE": round(float(rmse), 4),
            "MAPE_Percent": round(float(mape), 2)
        }
        predictions[name] = preds
        fitted_models[name] = model
        
        print(f" -> R² Score : {r2:.4f}")
        print(f" -> MAE      : {mae:.4f}% CPU")
        print(f" -> RMSE     : {rmse:.4f}% CPU")
        print(f" -> MAPE     : {mape:.2f}%")
        
    # Determine best model based on R2 Score
    best_name = max(results, key=lambda k: results[k]["R2_Score"])
    best_model = fitted_models[best_name]
    print(f"\n[BEST MODEL] Best Model Selected: {best_name} (R^2 = {results[best_name]['R2_Score']})")
    
    # Save best model, scaler, and metadata
    joblib.dump(best_model, os.path.join("models", "best_model.joblib"))
    joblib.dump(scaler, os.path.join("models", "scaler.joblib"))
    
    metadata = {
        "best_model_name": best_name,
        "feature_columns": feature_cols,
        "metrics_summary": results,
        "test_sample_count": len(y_test),
        "train_sample_count": len(y_train)
    }
    
    with open(os.path.join("models", "model_metrics.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    # 4. Generate Visualizations
    generate_evaluation_charts(results, predictions, y_test, fitted_models[best_name], feature_cols, best_name)
    
    print("\nModel training, benchmarking, and chart generation completed successfully!")
    return metadata

def generate_evaluation_charts(results, predictions, y_test, best_model, feature_cols, best_name):
    # Palette
    bg_color = "#0f172a"
    card_color = "#1e293b"
    text_color = "#f8fafc"
    accent_blue = "#38bdf8"
    accent_green = "#4ade80"
    accent_coral = "#f87171"
    
    # --- Plot 1: Model Comparison Bar Chart ---
    plt.figure(figsize=(9, 5), facecolor=bg_color)
    ax = plt.axes()
    ax.set_facecolor(card_color)
    
    model_names = list(results.keys())
    r2_scores = [results[m]["R2_Score"] for m in model_names]
    mae_scores = [results[m]["MAE"] for m in model_names]
    rmse_scores = [results[m]["RMSE"] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.25
    
    plt.bar(x - width, r2_scores, width, label="R² Score (Higher=Better)", color=accent_green)
    plt.bar(x, mae_scores, width, label="MAE (% CPU, Lower=Better)", color=accent_blue)
    plt.bar(x + width, rmse_scores, width, label="RMSE (% CPU, Lower=Better)", color=accent_coral)
    
    plt.xticks(x, model_names, color=text_color, fontsize=11, fontweight="bold")
    plt.yticks(color=text_color)
    plt.title("Cloud Resource Prediction - Model Benchmark Comparison", color=text_color, fontsize=13, fontweight="bold", pad=15)
    plt.legend(facecolor=card_color, edgecolor=text_color, labelcolor=text_color)
    plt.grid(axis='y', linestyle='--', alpha=0.3, color="#64748b")
    
    for spine in ax.spines.values():
        spine.set_color("#334155")
        
    plt.tight_layout()
    chart1_path = os.path.join("static", "images", "evaluation_model_comparison.png")
    plt.savefig(chart1_path, dpi=200, facecolor=bg_color)
    plt.close()
    
    # --- Plot 2: Actual vs Predicted Curve (Test Subset) ---
    plt.figure(figsize=(11, 5), facecolor=bg_color)
    ax = plt.axes()
    ax.set_facecolor(card_color)
    
    sample_window = 120
    y_test_sub = np.array(y_test)[:sample_window]
    y_pred_sub = predictions[best_name][:sample_window]
    time_sub = np.arange(sample_window)
    
    plt.plot(time_sub, y_test_sub, label="Actual Ground-Truth CPU (%)", color="#94a3b8", linewidth=2.0, alpha=0.85)
    plt.plot(time_sub, y_pred_sub, label=f"Predicted Future CPU ({best_name})", color=accent_blue, linewidth=2.5, linestyle="--")
    
    # Highlight scaling thresholds
    plt.axhline(75, color=accent_coral, linestyle=":", alpha=0.7, label="Scale-Up Threshold (75%)")
    plt.axhline(30, color=accent_green, linestyle=":", alpha=0.7, label="Scale-Down Threshold (30%)")
    
    plt.xlabel("Test Step (5-min intervals)", color=text_color, fontsize=11)
    plt.ylabel("CPU Utilization (%)", color=text_color, fontsize=11)
    plt.title(f"Actual vs. Predicted CPU Utilization Profile ({best_name})", color=text_color, fontsize=13, fontweight="bold", pad=15)
    plt.xticks(color=text_color)
    plt.yticks(color=text_color)
    plt.ylim(0, 105)
    plt.legend(facecolor=card_color, edgecolor=text_color, labelcolor=text_color, loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.25, color="#64748b")
    
    for spine in ax.spines.values():
        spine.set_color("#334155")
        
    plt.tight_layout()
    chart2_path = os.path.join("static", "images", "evaluation_actual_vs_predicted.png")
    plt.savefig(chart2_path, dpi=200, facecolor=bg_color)
    plt.close()
    
    # --- Plot 3: Feature Importance ---
    if hasattr(best_model, "feature_importances_"):
        plt.figure(figsize=(9, 5), facecolor=bg_color)
        ax = plt.axes()
        ax.set_facecolor(card_color)
        
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10] # Top 10
        top_features = [feature_cols[i] for i in indices]
        top_importances = importances[indices]
        
        y_pos = np.arange(len(top_features))
        plt.barh(y_pos, top_importances[::-1], color=accent_blue, edgecolor="#0284c7")
        plt.yticks(y_pos, top_features[::-1], color=text_color, fontsize=10)
        plt.xticks(color=text_color)
        plt.xlabel("Relative Importance Score", color=text_color, fontsize=11)
        plt.title(f"Top 10 Feature Importances ({best_name})", color=text_color, fontsize=13, fontweight="bold", pad=15)
        plt.grid(axis='x', linestyle='--', alpha=0.25, color="#64748b")
        
        for spine in ax.spines.values():
            spine.set_color("#334155")
            
        plt.tight_layout()
        chart3_path = os.path.join("static", "images", "evaluation_feature_importance.png")
        plt.savefig(chart3_path, dpi=200, facecolor=bg_color)
        plt.close()
        
    # --- Plot 4: Residual / Error Distribution ---
    residuals = np.array(y_test) - predictions[best_name]
    plt.figure(figsize=(8, 4.5), facecolor=bg_color)
    ax = plt.axes()
    ax.set_facecolor(card_color)
    
    plt.hist(residuals, bins=35, color=accent_green, edgecolor="#166534", alpha=0.85, density=True)
    plt.axvline(0, color=accent_coral, linestyle="--", linewidth=1.8, label="Zero Error Mean")
    
    plt.xlabel("Prediction Residual Error (Actual - Predicted %)", color=text_color, fontsize=11)
    plt.ylabel("Probability Density", color=text_color, fontsize=11)
    plt.title(f"Prediction Error Residual Distribution ({best_name})", color=text_color, fontsize=13, fontweight="bold", pad=15)
    plt.xticks(color=text_color)
    plt.yticks(color=text_color)
    plt.legend(facecolor=card_color, edgecolor=text_color, labelcolor=text_color)
    plt.grid(True, linestyle='--', alpha=0.25, color="#64748b")
    
    for spine in ax.spines.values():
        spine.set_color("#334155")
        
    plt.tight_layout()
    chart4_path = os.path.join("static", "images", "evaluation_residuals.png")
    plt.savefig(chart4_path, dpi=200, facecolor=bg_color)
    plt.close()

if __name__ == "__main__":
    train_and_benchmark()
