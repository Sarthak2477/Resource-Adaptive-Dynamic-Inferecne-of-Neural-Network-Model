import os
import sys
import time
import csv
import platform
import torch
import numpy as np
import argparse

# Ensure project root and resource_control directories can be imported correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "resource_control"))

from models import Model, set_model_width, set_model_bit_width, FLAGS


def configure_cpu_threads(requested=None):
    if requested is None:
        requested = int(os.environ.get("INFERENCE_NUM_THREADS", "1"))
    requested = max(1, int(requested))
    torch.set_num_threads(requested)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)
    return requested

def get_clean_hardware_name(device):
    if device.type == "cuda":
        raw_name = torch.cuda.get_device_name(0)
    else:
        # Use processor name or fallback to machine architecture
        raw_name = f"cpu_{platform.processor() or platform.machine()}"
    
    # Normalize to lowercase and remove/replace invalid file system chars
    clean_name = raw_name.lower().strip()
    clean_name = clean_name.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
    clean_name = "".join([c for c in clean_name if c.isalnum() or c in ("_", ".")])
    return clean_name if clean_name else "unknown_device"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    thread_count = configure_cpu_threads(args.threads)
    print("=" * 60)
    print(" AUTOMATED HARDWARE PROFILER ".center(60, "="))
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hw_name = get_clean_hardware_name(device)
    print(f"Detected Hardware Device: {device} ({hw_name})")

    # 1. Initialize model
    model = Model(num_classes=10, input_size=32)

    # 2. Check and load checkpoint if available
    checkpoint_paths = [
        os.path.join(project_root, "models", "checkpoint", "us_resnet_epoch100_checkpoint.pt.zip"),
        os.path.join(project_root, "models", "checkpoint", "us_resnet_epoch100_checkpoint.pt"),
        os.path.join(project_root, "models", "checkpoints", "best_model.pt"),
        os.path.join(project_root, "models", "checkpoints", "us_resnet_epoch100_checkpoint.pt"),
        os.path.join(project_root, "us_resnet_epoch100_checkpoint.pt")
    ]
    loaded = False
    for path in checkpoint_paths:
        if os.path.exists(path):
            print(f"Loading weights from checkpoint: {path}")
            checkpoint = torch.load(path, map_location=device)
            state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
            
            # Clean DataParallel 'module.' prefix
            clean_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    clean_state_dict[k[7:]] = v
                else:
                    clean_state_dict[k] = v
            model.load_state_dict(clean_state_dict)
            loaded = True
            break
            
    if not loaded:
        print("Warning: No pre-trained model checkpoint found. Profiling with randomly initialized weights.")

    model = model.to(device)
    model.eval()

    # Generate dummy input matching CIFAR-10 shape
    dummy_input = torch.randn(1, 3, 32, 32, device=device)

    print("\nWarmup phase for GPU/CPU framework overheads...")
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Get configurations to profile
    configs = FLAGS.deploy_configs
    results = []

    print(f"\nCPU threads: {thread_count}")
    print(f"Profiling {len(configs)} sub-network configurations (cold sample + 20 warmup + 200 warm samples each):")
    
    for idx, config in enumerate(configs):
        width_mult, bit_width = config
        print(f"[{idx+1}/{len(configs)}] Profiling config: width={width_mult}, bits={bit_width} ... ", end="", flush=True)

        set_model_width(model, width_mult)
        set_model_bit_width(model, bit_width)

        # Repeatedly measure the first inference after applying a configuration.
        # This captures transition overhead instead of treating one sample as a
        # percentile estimate.
        cold_latencies = []
        for _ in range(10):
            set_model_width(model, width_mult)
            set_model_bit_width(model, bit_width)
            with torch.no_grad():
                t_start = time.perf_counter()
                _ = model(dummy_input)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                cold_latencies.append((time.perf_counter() - t_start) * 1000.0)

        # Warmup passes for this specific config
        with torch.no_grad():
            for _ in range(20):
                _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

        # Measurement passes
        latencies = []
        for _ in range(200):
            t_start = time.perf_counter()
            with torch.no_grad():
                _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000.0) # Convert to ms

        avg_lat = np.mean(latencies)
        std_lat = np.std(latencies)
        max_lat = np.max(latencies)
        min_lat = np.min(latencies)
        p50_lat = np.percentile(latencies, 50)
        p95_lat = np.percentile(latencies, 95)
        p99_lat = np.percentile(latencies, 99)

        print(f"Avg: {avg_lat:.2f}ms | Std: {std_lat:.2f}ms")

        from resource_control.telemetry import hardware_fingerprint
        from resource_control.controller import extract_config_features
        fingerprint = hardware_fingerprint(device)
        cfg_feat = extract_config_features(config)
        approx_flops = cfg_feat["approx_flops"]
        approx_params = cfg_feat["approx_params"]

        results.append({
            "width_mult": width_mult,
            "bit_width": bit_width,
            "approx_flops": approx_flops,
            "approx_params": approx_params,
            "flops_per_speed": approx_flops / fingerprint["device_speed_score"],
            "cpu_pct": 15.0, # baseline
            "mem_available_mb": 1024.0, # baseline
            "thermal_c": 45.0, # baseline
            "device_speed_score": fingerprint["device_speed_score"],
            "cpu_cores": fingerprint["cpu_cores"],
            "ram_gb": fingerprint["ram_gb"],
            "has_cuda": fingerprint["has_cuda"],
            "avg_latency_ms": avg_lat,
            "latency_ms": avg_lat,
            "std_latency_ms": std_lat,
            "latency_p50_ms": p50_lat,
            "latency_p95_ms": p95_lat,
            "latency_p99_ms": p99_lat,
            "cold_latency_p50_ms": np.percentile(cold_latencies, 50),
            "cold_latency_p95_ms": np.percentile(cold_latencies, 95),
            "cold_latency_p99_ms": np.percentile(cold_latencies, 99),
            "thread_count": thread_count,
            "max_latency_ms": max_lat,
            "min_latency_ms": min_lat
        })

    # Save to profiles/ directory
    profiles_dir = os.path.join(project_root, "profiles")
    os.makedirs(profiles_dir, exist_ok=True)
    profile_csv_path = os.path.join(profiles_dir, f"{hw_name}_profile.csv")
    
    fieldnames = [
        "width_mult", "bit_width", "approx_flops", "approx_params",
        "flops_per_speed", "cpu_pct", "mem_available_mb", "thermal_c",
        "device_speed_score", "cpu_cores", "ram_gb", "has_cuda",
        "avg_latency_ms", "latency_ms", "std_latency_ms",
        "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
        "cold_latency_p50_ms", "cold_latency_p95_ms", "cold_latency_p99_ms",
        "max_latency_ms", "min_latency_ms", "thread_count"
    ]
    with open(profile_csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 60)
    print(f"Profiling complete! Results saved to: {profile_csv_path}")
    print("=" * 60)

    # Print a markdown table for direct documentation / console viewing
    print("\n| Width Mult | Bit Width | Avg Latency (ms) | Std Dev (ms) | Max Latency (ms) |")
    print("| :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        print(f"| {r['width_mult']:.2f} | {r['bit_width']} | {r['avg_latency_ms']:.3f} | {r['std_latency_ms']:.3f} | {r['max_latency_ms']:.3f} |")

if __name__ == "__main__":
    main()
