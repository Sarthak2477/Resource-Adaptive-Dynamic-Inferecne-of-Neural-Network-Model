import os
import glob
import pickle
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
# Ensure project root and resource_control directories can be imported correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../resource_control")))

from resource_control.controller import FEATURES, extract_config_features

def load_and_enrich_profiles():
    profile_files = glob.glob("profiles/*_profile.csv")
    if not profile_files:
        print("No profile CSV files found in profiles/ directory!")
        return None

    all_dfs = []
    for filepath in profile_files:
        filename = os.path.basename(filepath)
        print(f"Loading and enriching: {filename}")
        df = pd.read_csv(filepath)
        
        # Determine device specs based on filename or contents
        is_gpu = "cuda" in filename or "t4" in filename or "rtx" in filename
        default_speed = 8000.0 if is_gpu else 800.0
        default_cores = 4.0 if is_gpu else 8.0
        default_ram = 16.0 if is_gpu else 16.0
        
        # Populate missing columns with sensible defaults or calculated values
        if "device_speed_score" not in df.columns:
            df["device_speed_score"] = default_speed
        if "cpu_cores" not in df.columns:
            df["cpu_cores"] = default_cores
        if "ram_gb" not in df.columns:
            df["ram_gb"] = default_ram
        if "has_cuda" not in df.columns:
            df["has_cuda"] = 1.0 if is_gpu else 0.0
            
        # Re-calculate config metrics if missing
        if "approx_flops" not in df.columns or "approx_params" not in df.columns:
            approx_flops = []
            approx_params = []
            for _, row in df.iterrows():
                cfg_feat = extract_config_features((row["width_mult"], row["bit_width"]))
                approx_flops.append(cfg_feat["approx_flops"])
                approx_params.append(cfg_feat["approx_params"])
            df["approx_flops"] = approx_flops
            df["approx_params"] = approx_params
            
        if "flops_per_speed" not in df.columns:
            df["flops_per_speed"] = df["approx_flops"] / df["device_speed_score"]
            
        # Set dynamic load telemetry defaults if missing (baseline idle state)
        if "cpu_pct" not in df.columns:
            df["cpu_pct"] = 15.0
        if "mem_available_mb" not in df.columns:
            df["mem_available_mb"] = 1024.0
        if "thermal_c" not in df.columns:
            df["thermal_c"] = 45.0
            
        # Rename target column to latency_ms if it exists as avg_latency_ms
        if "latency_ms" not in df.columns and "avg_latency_ms" in df.columns:
            df["latency_ms"] = df["avg_latency_ms"]

        # Synthesize rows with different telemetry loads to let the model learn 
        # the relationship between contention (CPU, thermals) and latency.
        synthesized_dfs = [df.copy()]
        
        # Scenario 1: High CPU load (cpu_pct=85%, latency * 1.3)
        df_high_cpu = df.copy()
        df_high_cpu["cpu_pct"] = 85.0
        df_high_cpu["latency_ms"] = df_high_cpu["latency_ms"] * 1.3
        synthesized_dfs.append(df_high_cpu)
        
        # Scenario 2: High thermal throttling (thermal_c=80C, latency * 1.25)
        df_high_temp = df.copy()
        df_high_temp["thermal_c"] = 80.0
        df_high_temp["latency_ms"] = df_high_temp["latency_ms"] * 1.25
        synthesized_dfs.append(df_high_temp)
        
        # Scenario 3: Heavy combined stress (cpu_pct=85%, thermal_c=80C, latency * 1.6)
        df_both = df.copy()
        df_both["cpu_pct"] = 85.0
        df_both["thermal_c"] = 80.0
        df_both["latency_ms"] = df_both["latency_ms"] * 1.6
        synthesized_dfs.append(df_both)
        
        df_combined = pd.concat(synthesized_dfs, ignore_index=True)
        all_dfs.append(df_combined)
        
    return pd.concat(all_dfs, ignore_index=True)

def main():
    print("=" * 60)
    print(" SURROGATE MODEL TRAINING PIPELINE ".center(60, "="))
    print("=" * 60)

    df = load_and_enrich_profiles()
    if df is None:
        return

    print(f"\nLoaded dataset with {len(df)} samples for training.")
    print(f"Features: {FEATURES}")

    # Prepare features and targets
    X = df[FEATURES].values
    y = df["latency_ms"].values

    # Train RandomForestRegressor
    print("\nTraining Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Calculate training metrics
    preds = model.predict(X)
    mae = np.mean(np.abs(y - preds))
    mape = np.mean(np.abs((y - preds) / y)) * 100.0
    print(f"Training MAE: {mae:.3f} ms")
    print(f"Training MAPE: {mape:.2f}%")

    # Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\nFeature Importances:")
    for idx in indices:
        print(f"  {FEATURES[idx]:<20} : {importances[idx]:.4f}")

    # Save trained model to surrogate_model.pkl in weights/ folder
    model_dir = "weights"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "surrogate_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved successfully to: {model_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
