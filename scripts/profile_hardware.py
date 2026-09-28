import os
import sys
import time
import csv
import json
import hashlib
import platform
import random
import subprocess
from datetime import datetime
import torch
import numpy as np
import argparse
import psutil

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

def set_config(model, config):
    width_mult, bit_width = config
    set_model_width(model, width_mult)
    set_model_bit_width(model, bit_width)

def measure_latency(model, dummy_input, device):
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000.0

def summarize(values):
    values = np.asarray(values, dtype=float)
    return {
        "sample_count": int(values.size),
        "mean_ms": float(np.mean(values)),
        "std_ms": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "median_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }

def query_nvidia_smi():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,temperature.gpu,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=5)
        values = [value.strip() for value in result.stdout.splitlines()[0].split(",")]
        return {
            "driver_version": values[0],
            "gpu_temperature_c": values[1],
            "gpu_power_draw_w": values[2],
            "gpu_power_limit_w": values[3],
        }
    except (OSError, subprocess.SubprocessError, IndexError):
        return {}

def main():
    session_started_at = datetime.now().astimezone()
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--power-mode", default="unspecified",
                        help="Record the OS/device power mode used for this session")
    parser.add_argument("--warm-samples", type=int, default=200)
    parser.add_argument("--warmup-samples", type=int, default=20)
    parser.add_argument("--transition-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--allow-random-init", action="store_true",
                        help="Allow profiling without the trained checkpoint (not recommended)")
    args = parser.parse_args()
    if min(args.warm_samples, args.transition_repeats) < 1 or args.warmup_samples < 0:
        parser.error("warm samples and transition repeats must be positive; warmup samples cannot be negative")
    thread_count = configure_cpu_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
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
    loaded_checkpoint = None
    checkpoint_sha256 = None
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
            loaded_checkpoint = os.path.relpath(path, project_root)
            digest = hashlib.sha256()
            with open(path, "rb") as checkpoint_file:
                for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                    digest.update(chunk)
            checkpoint_sha256 = digest.hexdigest()
            break
            
    if not loaded:
        if not args.allow_random_init:
            raise FileNotFoundError(
                "No trained model checkpoint found. Supply the Experiment 0 checkpoint or "
                "use --allow-random-init only for an explicitly non-model profiling run.")
        print("WARNING: profiling randomly initialized weights by explicit request.")

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

    cpu_pct = psutil.cpu_percent(interval=0.2)
    mem_available_mb = psutil.virtual_memory().available / 1e6
    device_metrics_start = query_nvidia_smi()
    try:
        thermal_c = float(device_metrics_start["gpu_temperature_c"])
    except (KeyError, TypeError, ValueError):
        thermal_c = None

    # Get configurations to profile
    configs = FLAGS.deploy_configs
    results = []

    print(f"\nCPU threads: {thread_count}")
    print(f"Profiling {len(configs)} configurations: {args.transition_repeats} repeats per ordered "
          f"distinct transition and {args.warm_samples} warm observations per configuration.")
    
    from resource_control.telemetry import hardware_fingerprint
    from resource_control.controller import extract_config_features
    fingerprint = hardware_fingerprint(device)
    observations = []
    warm_by_config = {}
    transition_first = {}
    transition_warm = {}

    for idx, config in enumerate(configs):
        width_mult, bit_width = config
        print(f"[{idx+1}/{len(configs)}] Profiling warm config: width={width_mult}, bits={bit_width} ... ",
              end="", flush=True)
        set_config(model, config)
        with torch.no_grad():
            for _ in range(args.warmup_samples):
                _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

        latencies = []
        for sample_index in range(args.warm_samples):
            latency = measure_latency(model, dummy_input, device)
            latencies.append(latency)
            observations.append({
                "record_type": "warm", "from_width_mult": "", "from_bit_width": "",
                "width_mult": width_mult, "bit_width": bit_width,
                "sample_index": sample_index, "latency_ms": latency,
            })
        warm_by_config[config] = latencies
        print(f"mean: {np.mean(latencies):.2f} ms | n={len(latencies)}")

    # Randomize pair order to avoid systematically tying a particular pair to run drift.
    transitions = [(source, target) for source in configs for target in configs if source != target]
    random.Random(12345).shuffle(transitions)
    for transition_index, (source, target) in enumerate(transitions):
        first_values = []
        warm_values = []
        for repeat_index in range(args.transition_repeats):
            set_config(model, source)
            _ = measure_latency(model, dummy_input, device)
            set_config(model, target)
            first_latency = measure_latency(model, dummy_input, device)
            warm_latency = measure_latency(model, dummy_input, device)
            first_values.append(first_latency)
            warm_values.append(warm_latency)
            for phase, latency in (("transition_first", first_latency), ("transition_warm", warm_latency)):
                observations.append({
                    "record_type": phase,
                    "from_width_mult": source[0], "from_bit_width": source[1],
                    "width_mult": target[0], "bit_width": target[1],
                    "sample_index": repeat_index, "latency_ms": latency,
                })
        transition_first[(source, target)] = first_values
        transition_warm[(source, target)] = warm_values
        if (transition_index + 1) % max(1, len(transitions) // 8) == 0:
            print(f"Measured {transition_index + 1}/{len(transitions)} ordered transitions.")

    results = []
    for config in configs:
        width_mult, bit_width = config
        latencies = warm_by_config[config]
        warm_stats = summarize(latencies)
        incoming_first = [value for (source, target), values in transition_first.items()
                          if target == config for value in values]
        cold_stats = summarize(incoming_first)
        cfg_feat = extract_config_features(config)
        approx_flops = cfg_feat["approx_flops"]
        approx_params = cfg_feat["approx_params"]

        results.append({
            "width_mult": width_mult,
            "bit_width": bit_width,
            "approx_flops": approx_flops,
            "approx_params": approx_params,
            "flops_per_speed": approx_flops / fingerprint["device_speed_score"],
            "cpu_pct": cpu_pct,
            "mem_available_mb": mem_available_mb,
            "thermal_c": thermal_c,
            "device_speed_score": fingerprint["device_speed_score"],
            "cpu_cores": fingerprint["cpu_cores"],
            "ram_gb": fingerprint["ram_gb"],
            "has_cuda": fingerprint["has_cuda"],
            "avg_latency_ms": warm_stats["mean_ms"],
            "latency_ms": warm_stats["mean_ms"],
            "std_latency_ms": warm_stats["std_ms"],
            "latency_p50_ms": warm_stats["median_ms"],
            "latency_p95_ms": warm_stats["p95_ms"],
            "latency_p99_ms": warm_stats["p99_ms"],
            "warm_sample_count": warm_stats["sample_count"],
            "warm_min_ms": warm_stats["min_ms"],
            "warm_max_ms": warm_stats["max_ms"],
            "cold_latency_p50_ms": cold_stats["median_ms"],
            "cold_latency_p95_ms": cold_stats["p95_ms"],
            "cold_latency_p99_ms": cold_stats["p99_ms"],
            "cold_transition_sample_count": cold_stats["sample_count"],
            "cold_transition_mean_ms": cold_stats["mean_ms"],
            "cold_transition_std_ms": cold_stats["std_ms"],
            "cold_transition_min_ms": cold_stats["min_ms"],
            "cold_transition_max_ms": cold_stats["max_ms"],
            "thread_count": thread_count,
            "max_latency_ms": warm_stats["max_ms"],
            "min_latency_ms": warm_stats["min_ms"]
        })

    profiles_dir = os.path.join(project_root, "profiles")
    raw_dir = os.path.join(profiles_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    timestamp = session_started_at.strftime("%Y%m%d_%H%M%S")
    session_id = args.session_id or timestamp
    safe_session_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in session_id)
    stem = f"{hw_name}_{timestamp}_{safe_session_id}_threads{thread_count}"
    observations_path = os.path.join(raw_dir, f"{stem}_observations.csv")
    summary_path = os.path.join(raw_dir, f"{stem}_config_summary.csv")
    transitions_path = os.path.join(raw_dir, f"{stem}_transition_summary.csv")
    metadata_path = os.path.join(raw_dir, f"{stem}_metadata.json")
    output_paths = [observations_path, summary_path, transitions_path, metadata_path]
    if any(os.path.exists(path) for path in output_paths):
        raise FileExistsError(f"Session output already exists; choose a unique --session-id: {stem}")

    raw_fields = ["record_type", "from_width_mult", "from_bit_width", "width_mult", "bit_width",
                  "sample_index", "latency_ms"]
    with open(observations_path, mode="w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(observations)

    fieldnames = [
        "width_mult", "bit_width", "approx_flops", "approx_params",
        "flops_per_speed", "cpu_pct", "mem_available_mb", "thermal_c",
        "device_speed_score", "cpu_cores", "ram_gb", "has_cuda",
        "avg_latency_ms", "latency_ms", "std_latency_ms",
        "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
        "cold_latency_p50_ms", "cold_latency_p95_ms", "cold_latency_p99_ms",
        "max_latency_ms", "min_latency_ms", "thread_count", "warm_sample_count",
        "warm_min_ms", "warm_max_ms", "cold_transition_sample_count",
        "cold_transition_mean_ms", "cold_transition_std_ms",
        "cold_transition_min_ms", "cold_transition_max_ms"
    ]
    with open(summary_path, mode="w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    transition_rows = []
    for (source, target), first_values in transition_first.items():
        for phase, values in (("first", first_values), ("warm", transition_warm[(source, target)])):
            transition_rows.append({
                "from_width_mult": source[0], "from_bit_width": source[1],
                "to_width_mult": target[0], "to_bit_width": target[1],
                "phase": phase, **summarize(values),
            })
    transition_fields = ["from_width_mult", "from_bit_width", "to_width_mult", "to_bit_width",
                         "phase", "sample_count", "mean_ms", "std_ms", "median_ms", "p95_ms",
                         "p99_ms", "min_ms", "max_ms"]
    with open(transitions_path, mode="w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=transition_fields)
        writer.writeheader()
        writer.writerows(transition_rows)

    metadata = {
        "session_id": session_id,
        "started_at_local": session_started_at.isoformat(),
        "device": str(device),
        "device_name": hw_name,
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "thread_count": thread_count,
        "power_mode": args.power_mode,
        "seed": args.seed,
        "checkpoint": loaded_checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "warm_samples_per_config": args.warm_samples,
        "warmup_samples_per_config": args.warmup_samples,
        "transition_repeats_per_ordered_pair": args.transition_repeats,
        "transition_pair_count": len(transitions),
        "transition_order": "seeded randomized; each pair restores and runs source before target",
        "device_fingerprint": fingerprint,
        "device_metrics_start": device_metrics_start,
        "device_metrics_end": query_nvidia_smi(),
    }
    with open(metadata_path, "w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, indent=2)

    print("\n" + "=" * 60)
    print("Profiling complete. Session artifacts:")
    print(f"  Raw observations: {observations_path}")
    print(f"  Config summaries: {summary_path}")
    print(f"  Transition matrix: {transitions_path}")
    print(f"  Session metadata: {metadata_path}")
    print("=" * 60)

    print("\nPer-config warm and incoming first-transition distributions are in the config summary.")
    print("P99 values based on 200 observations are empirical and noisy, not service guarantees.")

if __name__ == "__main__":
    main()
