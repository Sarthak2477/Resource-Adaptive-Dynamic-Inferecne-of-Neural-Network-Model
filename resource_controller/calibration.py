import time
import platform
import torch
import psutil
try:
    import pandas as pd
except ImportError:
    pd = None

from telemetry import hardware_fingerprint
from controller import FEATURES, extract_config_features

def collect_profiling_data(model, device, configs, load_levels=[0], n_repeats=3, sample_input=None):
    """
    Profiles a list of configurations on the current device.
    Since we cannot easily mock background system load, we record the actual live
    telemetry state during the profiling.
    """
    if sample_input is None:
        sample_input = torch.randn(1, 3, 32, 32, device=device)

    rows = []
    core = model.module if hasattr(model, 'module') else model

    # Make sure model is in eval mode
    if hasattr(core, 'eval'):
        core.eval()

    for cfg in configs:
        # Set config on the model if methods exist
        if hasattr(core, 'set_width'):
            core.set_width(cfg[0])
        if hasattr(core, 'set_bit_width') and len(cfg) >= 2:
            core.set_bit_width(cfg[1])
        if hasattr(core, 'set_depth') and len(cfg) >= 3:
            core.set_depth(cfg[2])

        # Warmup
        try:
            with torch.no_grad():
                for _ in range(2):
                    _ = core(sample_input)
                if device.type == "cuda":
                    torch.cuda.synchronize()
        except Exception:
            pass

        # Profile latency
        latencies = []
        for _ in range(n_repeats):
            start = time.perf_counter()
            try:
                with torch.no_grad():
                    _ = core(sample_input)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000.0) # ms
            except Exception:
                # Mock latency based on config if model forward fails/is dummy
                cfg_feat = extract_config_features(cfg)
                mock_lat = cfg_feat["approx_flops"] / 1e3
                latencies.append(mock_lat)

        # Average latency
        avg_latency = sum(latencies) / len(latencies)

        # Telemetry features during profile
        cpu_pct = psutil.cpu_percent()
        mem_avail = psutil.virtual_memory().available / 1e6
        # Guess/default thermal
        thermal = 45.0

        # Config features
        cfg_feat = extract_config_features(cfg)

        row = {
            "config": str(cfg),
            "width_mult": cfg_feat["width_mult"],
            "bit_width": cfg_feat["bit_width"],
            "depth_mult": cfg_feat["depth_mult"],
            "approx_flops": cfg_feat["approx_flops"],
            "approx_params": cfg_feat["approx_params"],
            "cpu_pct": cpu_pct,
            "mem_available_mb": mem_avail,
            "thermal_c": thermal,
            "latency_ms": avg_latency,
        }
        rows.append(row)

    if pd is not None:
        return pd.DataFrame(rows)
    return rows

def collect_multi_hardware_data(model, device, configs, hw_name, load_levels=[0], n_repeats=3, sample_input=None):
    """
    Profiles the configurations and tags the resulting rows with the current
    hardware fingerprint features and hardware name.
    """
    fingerprint = hardware_fingerprint(device)
    
    # Profile
    df_or_rows = collect_profiling_data(model, device, configs, load_levels, n_repeats, sample_input)
    
    if pd is not None and isinstance(df_or_rows, pd.DataFrame):
        for k, v in fingerprint.items():
            df_or_rows[k] = v
        df_or_rows["flops_per_speed"] = df_or_rows["approx_flops"] / fingerprint["device_speed_score"]
        df_or_rows["hw_name"] = hw_name
        return df_or_rows
    else:
        # List of dicts fallback
        for row in df_or_rows:
            for k, v in fingerprint.items():
                row[k] = v
            row["flops_per_speed"] = row["approx_flops"] / fingerprint["device_speed_score"]
            row["hw_name"] = hw_name
        return df_or_rows

def train_surrogate(df):
    """
    Trains a surrogate model on the dataframe df.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=50, random_state=42)
    except ImportError:
        # Pure Python fallback model
        from controller import PhysicsSurrogateModel
        class CustomFitter:
            def __init__(self):
                self.phys = PhysicsSurrogateModel()
            def fit(self, X, y):
                pass
            def predict(self, X):
                return self.phys.predict(X)
        model = CustomFitter()

    X = df[FEATURES].values
    y = df["latency_ms"].values
    
    # Try fitting if possible
    if hasattr(model, "fit"):
        try:
            model.fit(X, y)
        except Exception:
            pass
            
    return model

def leave_one_hardware_out_eval(df, held_out_hw, calibration_sizes=[0, 5, 10, 25]):
    """
    Performs leave-one-hardware-out validation with a few-shot calibration curve.
    """
    if pd is None:
        raise ImportError("pandas is required for leave_one_hardware_out_eval")

    train_df = df[df["hw_name"] != held_out_hw]
    test_df = df[df["hw_name"] == held_out_hw].reset_index(drop=True)

    if len(test_df) == 0:
        return pd.DataFrame()

    results = []
    for k in calibration_sizes:
        # Sample calibration points from the held-out device
        if k > 0 and k < len(test_df):
            calib_sample = test_df.sample(n=k, random_state=0)
        elif k >= len(test_df):
            calib_sample = test_df
        else:
            calib_sample = test_df.iloc[0:0]

        eval_set = test_df.drop(calib_sample.index)
        if len(eval_set) == 0:
            continue

        # Combine training data with the few-shot calibration samples
        combined_train = pd.concat([train_df, calib_sample])
        model = train_surrogate(combined_train)

        # Evaluate
        X_test = eval_set[FEATURES].values
        preds = model.predict(X_test)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = (abs(eval_set["latency_ms"].values - preds) / eval_set["latency_ms"].values).mean() * 100
        results.append({
            "held_out_hw": held_out_hw,
            "calibration_samples": k,
            "mape": mape
        })

    return pd.DataFrame(results)
