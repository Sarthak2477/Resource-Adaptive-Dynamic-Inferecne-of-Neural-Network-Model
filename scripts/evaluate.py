import os
import sys
import time
import json
import argparse
import numpy as np
import torch
from torch.utils.data import Subset

# Ensure project root and resource_control directories can be imported correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "resource_control"))
sys.path.append(os.path.dirname(__file__))  # Add scripts/ dir for profile_hardware import

from resource_control.telemetry import ResourceMonitor, ResourceState
from resource_control.controller import (
    MeasuredFrontierLatencyModel,
    SurrogateBackedController,
    extract_config_features,
)
from models import Model, set_model_width, set_model_bit_width, get_dataloaders

def print_banner(msg):
    print("\n" + "=" * 60)
    print(f" {msg} ".center(60, "="))
    print("=" * 60)


def json_default(value):
    """Convert NumPy scalar values produced during evaluation to JSON types."""
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

def expand_frontier_continuous(discrete_frontier, step=0.05):
    """
    Expands a discrete Pareto frontier with 4 width steps to a continuous/dense frontier
    using linear interpolation for accuracy, baseline latency, and standard deviation.
    """
    # Group the discrete frontier entries by bit_width
    by_bit_width = {}
    for entry in discrete_frontier:
        w, b = entry['config']
        if b not in by_bit_width:
            by_bit_width[b] = []
        by_bit_width[b].append({
            'width': w,
            'latency_ms': entry['latency_ms'],
            'std_ms': entry.get('std_ms', 1.0),
            'latency_p50_ms': entry.get('latency_p50_ms', entry['latency_ms']),
            'latency_p95_ms': entry.get('latency_p95_ms', entry['latency_ms']),
            'latency_p99_ms': entry.get('latency_p99_ms', entry['latency_ms']),
            'cold_latency_p50_ms': entry.get('cold_latency_p50_ms', entry['latency_ms']),
            'cold_latency_p95_ms': entry.get('cold_latency_p95_ms', entry['latency_ms']),
            'cold_latency_p99_ms': entry.get('cold_latency_p99_ms', entry['latency_ms']),
            'acc': entry['acc']
        })
        
    # Sort entries for each bit_width by width
    for b in by_bit_width:
        by_bit_width[b] = sorted(by_bit_width[b], key=lambda x: x['width'])
        
    dense_frontier = []
    bit_widths = sorted(by_bit_width.keys())
    
    for b in bit_widths:
        known_points = by_bit_width[b]
        # Generate widths from min_width (0.25) to max_width (1.0) with given step
        min_w = known_points[0]['width']
        max_w = known_points[-1]['width']
        
        # Create a grid of fine-grained widths
        import numpy as np
        grid_widths = np.arange(min_w, max_w + 1e-5, step)
        
        for w in grid_widths:
            w = round(float(w), 3)
            # Find the interval [pt_lower, pt_upper] containing w
            pt_lower = None
            pt_upper = None
            for i in range(len(known_points) - 1):
                if known_points[i]['width'] <= w <= known_points[i+1]['width']:
                    pt_lower = known_points[i]
                    pt_upper = known_points[i+1]
                    break
            
            if pt_lower is None:
                # Exact boundary fallback
                if w <= min_w:
                    pt_lower = pt_upper = known_points[0]
                else:
                    pt_lower = pt_upper = known_points[-1]
            
            if pt_lower == pt_upper:
                acc = pt_lower['acc']
                lat = pt_lower['latency_ms']
                std = pt_lower['std_ms']
                percentile_values = {
                    key: pt_lower[key]
                    for key in (
                        'latency_p50_ms', 'latency_p95_ms', 'latency_p99_ms',
                        'cold_latency_p50_ms', 'cold_latency_p95_ms', 'cold_latency_p99_ms'
                    )
                }
            else:
                frac = (w - pt_lower['width']) / (pt_upper['width'] - pt_lower['width'])
                acc = pt_lower['acc'] + frac * (pt_upper['acc'] - pt_lower['acc'])
                lat = pt_lower['latency_ms'] + frac * (pt_upper['latency_ms'] - pt_lower['latency_ms'])
                std = pt_lower['std_ms'] + frac * (pt_upper['std_ms'] - pt_lower['std_ms'])
                percentile_values = {
                    key: pt_lower[key] + frac * (pt_upper[key] - pt_lower[key])
                    for key in (
                        'latency_p50_ms', 'latency_p95_ms', 'latency_p99_ms',
                        'cold_latency_p50_ms', 'cold_latency_p95_ms', 'cold_latency_p99_ms'
                    )
                }
                
            dense_frontier.append({
                'config': (w, b),
                'latency_ms': lat,
                'std_ms': std,
                **percentile_values,
                'acc': acc
            })
            
    # Sort dense frontier by latency
    return sorted(dense_frontier, key=lambda x: x['latency_ms'])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("avg", "p50", "p95", "p99"), default="p95")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--trace", choices=("sinusoidal", "step", "bursty", "heldout"), default="sinusoidal")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num-samples", type=int, default=500, help="Balanced CIFAR-10 evaluation subset size; defaults to 500 samples")
    parser.add_argument("--interpolate", action="store_true", help="Include unprofiled interpolated widths as an ablation")
    args = parser.parse_args()
    print_banner("1. Initializing Model and Loading Checkpoint")

    seed = args.seed
    np.random.seed(seed)
    thread_count = args.threads or int(os.environ.get("INFERENCE_NUM_THREADS", "1"))
    torch.set_num_threads(max(1, thread_count))
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Evaluation seed: {seed}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = Model(num_classes=10, input_size=32)
    
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
            print(f"Loading checkpoint from {path}...")
            checkpoint = torch.load(path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            
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
        print("Warning: No pre-trained model checkpoint found. Running with randomly initialized weights.")
        
    model = model.to(device)
    model.eval()

    print_banner("2. Preparing CIFAR-10 Evaluation Subset")
    # Load dataset
    train_loader, test_loader = get_dataloaders()
    test_dataset = test_loader.dataset
    from models import recalibrate_bn
    
    num_samples = args.num_samples
    if num_samples <= 0 or num_samples > 10000:
        raise ValueError(f"--num-samples must be between 1 and 10000, got {num_samples}.")

    # Select a balanced subset across the 10 CIFAR-10 classes.
    # By default this is 50 samples per class (500 total), matching the
    # requested evaluation workload while remaining deterministic.
    class_counts = [num_samples // 10 + (1 if i < num_samples % 10 else 0) for i in range(10)]
    subset_indices = []
    for class_idx, class_count in enumerate(class_counts):
        start_idx = class_idx * 1000
        subset_indices.extend(range(start_idx, start_idx + class_count))

    subset_dataset = Subset(test_dataset, subset_indices)
    eval_loader = torch.utils.data.DataLoader(
        subset_dataset,
        batch_size=1,  # process sample-by-sample for real-time control simulation
        shuffle=False
    )
    print(f"Loaded {len(subset_dataset)} test samples.")

    print_banner("3. Initializing Resource Controller")
    monitor = ResourceMonitor()
    hw_fingerprint = monitor.get_hardware_fingerprint(device)
    
    # Try to load hardware-specific latency profile from CSV
    from scripts.profile_hardware import get_clean_hardware_name
    hw_name = get_clean_hardware_name(device)
    profile_csv_path = os.path.join(project_root, "profiles", f"{hw_name}_profile.csv")
    
    # Subnet accuracies are hardware-independent
    CONFIG_ACCURACIES = {
        (0.25, 4): 89.69, (0.25, 8): 90.46, (0.25, 16): 90.57, (0.25, 32): 90.59,
        (0.5, 4): 90.19,  (0.5, 8): 91.83,  (0.5, 16): 91.79,  (0.5, 32): 91.83,
        (0.75, 4): 90.51, (0.75, 8): 92.12, (0.75, 16): 92.16, (0.75, 32): 92.18,
        (1.0, 4): 90.66,  (1.0, 8): 92.32,  (1.0, 16): 92.37,  (1.0, 32): 92.37,
    }
    
    frontier = []
    profile_has_percentiles = False
    if os.path.exists(profile_csv_path):
        print(f"Loading dynamic latency profile from: {profile_csv_path}")
        import csv
        with open(profile_csv_path, mode="r") as f:
            reader = csv.DictReader(f)
            profile_has_percentiles = {
                "latency_p50_ms", "latency_p95_ms", "latency_p99_ms"
            }.issubset(reader.fieldnames or [])
            for row in reader:
                wm = float(row["width_mult"])
                bw = int(row["bit_width"])
                avg_lat = float(row["avg_latency_ms"])
                std_lat = float(row.get("std_latency_ms", 1.0))
                config_key = (wm, bw)
                acc = CONFIG_ACCURACIES.get(config_key, 90.0)
                frontier.append({
                    'config': config_key, 
                    'latency_ms': avg_lat, 
                    'std_ms': std_lat, 
                    'latency_p50_ms': float(row.get('latency_p50_ms', avg_lat)),
                    'latency_p95_ms': float(row.get('latency_p95_ms', avg_lat)),
                    'latency_p99_ms': float(row.get('latency_p99_ms', avg_lat)),
                    'cold_latency_p50_ms': float(row.get('cold_latency_p50_ms', avg_lat)),
                    'cold_latency_p95_ms': float(row.get('cold_latency_p95_ms', avg_lat)),
                    'cold_latency_p99_ms': float(row.get('cold_latency_p99_ms', avg_lat)),
                    'acc': acc
                })
    else:
        print(f"Warning: Profile {profile_csv_path} not found. Initializing with default reference frontier.")
        frontier = [
            {'config': (0.25, 4), 'latency_ms': 5.0, 'std_ms': 1.0, 'acc': 89.69},
            {'config': (0.25, 8), 'latency_ms': 5.5, 'std_ms': 1.0, 'acc': 90.46},
            {'config': (0.25, 16), 'latency_ms': 5.8, 'std_ms': 1.0, 'acc': 90.57},
            {'config': (0.25, 32), 'latency_ms': 6.0, 'std_ms': 1.0, 'acc': 90.59},
            {'config': (0.5, 4), 'latency_ms': 10.0, 'std_ms': 1.5, 'acc': 90.19},
            {'config': (0.5, 8), 'latency_ms': 11.2, 'std_ms': 1.5, 'acc': 91.83},
            {'config': (0.5, 16), 'latency_ms': 11.5, 'std_ms': 1.5, 'acc': 91.79},
            {'config': (0.5, 32), 'latency_ms': 12.0, 'std_ms': 1.5, 'acc': 91.83},
            {'config': (0.75, 4), 'latency_ms': 16.5, 'std_ms': 2.0, 'acc': 90.51},
            {'config': (0.75, 8), 'latency_ms': 18.0, 'std_ms': 2.0, 'acc': 92.12},
            {'config': (0.75, 16), 'latency_ms': 18.5, 'std_ms': 2.0, 'acc': 92.16},
            {'config': (0.75, 32), 'latency_ms': 19.0, 'std_ms': 2.0, 'acc': 92.18},
            {'config': (1.0, 4), 'latency_ms': 22.0, 'std_ms': 2.5, 'acc': 90.66},
            {'config': (1.0, 8), 'latency_ms': 24.5, 'std_ms': 2.5, 'acc': 92.32},
            {'config': (1.0, 16), 'latency_ms': 25.0, 'std_ms': 2.5, 'acc': 92.37},
            {'config': (1.0, 32), 'latency_ms': 26.0, 'std_ms': 2.5, 'acc': 92.37},
        ]

    if args.policy in ("p50", "p95", "p99") and not profile_has_percentiles:
        raise RuntimeError(
            f"Profile {profile_csv_path} lacks percentile latency columns. "
            "Run scripts/profile_hardware.py before using percentile policies."
        )

    if args.interpolate:
        frontier = expand_frontier_continuous(frontier, step=0.05)
        print(f"Expanded Pareto frontier to {len(frontier)} interpolated candidate configurations.")
    else:
        print(f"Using {len(frontier)} measured deployable configurations.")

    # Use measured hardware latency as the primary prediction source. The
    # learned surrogate is retained for separate experiments because its
    # synthetic contention features are not calibrated to this run.
    surrogate_model = MeasuredFrontierLatencyModel(frontier)
    surrogate_path = os.path.join(project_root, "weights", "surrogate_model.pkl")
    print(f"Using measured frontier latency model for {profile_csv_path}")
    print(f"Learned surrogate retained at: {surrogate_path}")

    # Set min_dwell_s=0.0 to allow immediate adaptation per sample
    controller = SurrogateBackedController(
        frontier=frontier, 
        surrogate_model=surrogate_model, 
        min_dwell_s=0.0,
        k_risk=1.2,
        switching_penalty_ms=1.5,
        latency_policy=args.policy,
    )

    print_banner("4. Starting Resource-Controlled Evaluation")
    
    correct = 0
    total = 0
    inference_times = []
    config_counts = {}
    prediction_errors = []
    k_adapt_history = []
    deadline_misses = 0
    sample_records = []
    
    # We will simulate 100 evaluation steps:
    # - Budget: Fluctuating dynamically using a sinusoidal wave (mean=20ms, amplitude=12ms, period=40 steps)
    # - Contention:
    #     Steps 0-30: Normal baseline load (cpu_pct=15%, thermal=45C)
    #     Steps 30-70: Contention Spike (cpu_pct=85%, thermal=80C)
    #     Steps 70-100: Return to normal load (cpu_pct=15%, thermal=50C)

    for idx, (image, label) in enumerate(eval_loader):
        image, label = image.to(device), label.to(device)
        
        # 1. Generate Sinusoidal Budget with minor random jitter
        phase = (2.0 * np.pi * idx) / 40.0
        if args.trace == "sinusoidal":
            budget = 20.0 + 12.0 * np.sin(phase) + np.random.uniform(-1.0, 1.0)
        elif args.trace == "step":
            budget = (12.0 if idx < 25 else 28.0 if idx < 50 else 8.0 if idx < 75 else 24.0) + np.random.uniform(-0.5, 0.5)
        elif args.trace == "bursty":
            budget = 24.0 + np.random.uniform(-2.0, 2.0)
            if idx % 10 in (0, 1, 2):
                budget -= 14.0
        else:
            budget = 18.0 + 10.0 * np.sin(phase * 1.7 + 0.8) + np.random.uniform(-2.0, 2.0)
        budget = max(4.0, budget)  # Enforce minimum budget of 4.0ms
        
        # 2. Simulate hardware contention state
        if args.trace == "heldout":
            contention = 20 <= idx < 45 or 75 <= idx < 90
        else:
            contention = 30 <= idx < 70
        if contention:
            cpu_pct = 85.0
            thermal_c = 80.0
            scenario = "Contention Spike"
        else:
            cpu_pct = 15.0
            thermal_c = 45.0 if idx < 30 else 50.0
            scenario = "Normal Load"

        state = ResourceState(
            cpu_pct=cpu_pct,
            mem_available_mb=1024.0,
            battery_pct=80,
            thermal_c=thermal_c,
            timestamp=time.time()
        )

        # 3. Select configuration and retrieve predicted latency
        selected_cfg = controller.select(state, latency_budget_ms=budget, hw_fingerprint=hw_fingerprint)
        pred_latency = controller.last_predicted_latency
        config_counts[selected_cfg] = config_counts.get(selected_cfg, 0) + 1

        selected_features = extract_config_features(selected_cfg)
        selected_features.update({
            "flops_per_speed": selected_features["approx_flops"] / hw_fingerprint["device_speed_score"],
            "cpu_pct": float(state.cpu_pct),
            "mem_available_mb": float(state.mem_available_mb),
            "thermal_c": float(state.thermal_c) if state.thermal_c is not None else None,
            "device_speed_score": float(hw_fingerprint["device_speed_score"]),
            "cpu_cores": int(hw_fingerprint["cpu_cores"]),
            "ram_gb": float(hw_fingerprint["ram_gb"]),
            "has_cuda": float(hw_fingerprint["has_cuda"]),
        })

        # 4. Set config on model
        set_model_width(model, selected_cfg[0])
        set_model_bit_width(model, selected_cfg[1])

        # 4b. Lazy BN statistics calibration if needed
        key = (selected_cfg[0], selected_cfg[1])
        from models.ops import ResnetBatchNorm2d
        first_bn = next(m for m in model.modules() if isinstance(m, ResnetBatchNorm2d))
        if key not in first_bn.calibrated_running_mean:
            print(f"\n[BN Calibration] Lazily calibrating BN statistics for {key} on 10 batches...")
            recalibrate_bn(model, train_loader, selected_cfg[0], selected_cfg[1], device, num_batches=10)
            print("[BN Calibration] Done!")

        # 5. Inference & Actual Time measurement
        t0 = time.perf_counter()
        with torch.no_grad():
            output = model(image)
            if device.type == "cuda":
                torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0 # ms
        inference_times.append(dt)

        # 6. CLOSED-LOOP FEEDBACK UPDATE
        if pred_latency is not None:
            # Update prediction feedback loop with actual vs predicted latency
            controller.update_feedback(dt, pred_latency, config=selected_cfg)
            prediction_errors.append(abs(dt - pred_latency))
        k_adapt_history.append(controller.k_adapt)

        # Track deadline compliance
        missed_deadline = dt > budget
        if missed_deadline:
            deadline_misses += 1

        # 7. Accuracy tracking
        pred = output.argmax(dim=1)
        is_correct = (pred == label).item()
        if is_correct:
            correct += 1
        total += 1
        sample_records.append({
            "sample_index": int(idx),
            "scenario": scenario,
            "budget_ms": float(budget),
            "selected_config": [float(selected_cfg[0]), int(selected_cfg[1])],
            "selected_width_mult": float(selected_cfg[0]),
            "selected_bit_width": int(selected_cfg[1]),
            "controller_features": selected_features,
            "resource_state": {
                "cpu_pct": float(state.cpu_pct),
                "mem_available_mb": float(state.mem_available_mb),
                "battery_pct": float(state.battery_pct) if state.battery_pct is not None else None,
                "thermal_c": float(state.thermal_c) if state.thermal_c is not None else None,
            },
            "predicted_latency_ms": float(pred_latency) if pred_latency is not None else None,
            "actual_latency_ms": float(dt),
            "missed_deadline": bool(missed_deadline),
            "correct": bool(is_correct),
            "k_adapt": float(controller.k_adapt),
            "budget_feasible": bool(controller.last_budget_feasible),
            "selection_status": controller.last_selection_status,
            "prediction_is_cold": bool(controller.last_prediction_is_cold),
        })

        # Print progress logs
        if (idx + 1) % 10 == 0:
            miss_marker = "MISS" if missed_deadline else "OK  "
            correct_marker = "YES" if is_correct else "NO "
            W, V = 26, 12
            sep  = "+" + "-" * (W + V + 3) + "+" + "-" * (W + V + 3) + "+"
            hdr  = "+" + "=" * (W + V + 3) + "+" + "=" * (W + V + 3) + "+"
            def row(l1, v1, l2, v2):
                return f"| {l1:<{W}} {v1:>{V}} | {l2:<{W}} {v2:>{V}} |"
            print(f"\n{hdr}")
            title = f"  Sample {idx+1:03d}/100  |  {scenario:<16}  |  Deadline: {miss_marker}  |  Correct: {correct_marker}"
            print(f"| {title:<{W+V+W+V+5}} |")
            print(hdr)
            print(row("Metric", "Value", "Metric", "Value"))
            print(sep)
            print(row("Config (width, bits)",    f"({selected_cfg[0]:.2f},{selected_cfg[1]}b)",  "Budget (ms)",           f"{budget:.1f}"))
            print(row("Predicted Latency (ms)",  f"{pred_latency or 0.0:.1f}",                   "Actual Latency (ms)",   f"{dt:.1f}"))
            print(row("K_adapt",                 f"{controller.k_adapt:.3f}",                    "Pred Error (ms)",       f"{abs(dt-(pred_latency or 0.0)):.1f}"))
            print(sep)
            print(row("CPU Load (%)",            f"{cpu_pct:.1f}",                               "Temperature (C)",       f"{thermal_c:.1f}"))
            print(row("RAM Available (MB)",      f"{selected_features['mem_available_mb']:.1f}", "RAM Total (GB)",        f"{selected_features['ram_gb']:.1f}"))
            print(row("Device Speed Score",      f"{selected_features['device_speed_score']:.1f}","CPU Cores",            f"{selected_features['cpu_cores']}"))
            print(row("CUDA Available",          str(bool(selected_features['has_cuda'])),        "Width Multiplier",      f"{selected_features['width_mult']:.2f}"))
            print(sep)
            print(row("Bit Width",               f"{int(selected_features['bit_width'])}",        "Approx FLOPs",          f"{selected_features['approx_flops']:.0f}"))
            print(row("Approx Params",           f"{selected_features['approx_params']:.0f}",     "FLOPs / Speed",         f"{selected_features['flops_per_speed']:.2f}"))
            print(sep)

    print_banner("5. Evaluation Summary")
    avg_latency = sum(inference_times) / len(inference_times)
    accuracy = (correct / total) * 100.0
    mae_error = np.mean(prediction_errors) if prediction_errors else 0.0
    miss_rate = (deadline_misses / total) * 100.0
    p95_latency = np.percentile(inference_times, 95)
    p99_latency = np.percentile(inference_times, 99)
    infeasible_samples = sum(not record["budget_feasible"] for record in sample_records)
    feasible_misses = sum(record["missed_deadline"] and record["budget_feasible"] for record in sample_records)
    cold_misses = sum(record["missed_deadline"] and record["prediction_is_cold"] for record in sample_records)
    warm_misses = deadline_misses - cold_misses
    
    print(f"Total Samples Evaluated: {total}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Average Inference Latency: {avg_latency:.2f} ms")
    print(f"P95 Inference Latency: {p95_latency:.2f} ms")
    print(f"P99 Inference Latency: {p99_latency:.2f} ms")
    print(f"Mean Absolute Prediction Error (MAE): {mae_error:.2f} ms")
    print(f"Deadline Miss Rate: {miss_rate:.2f}% ({deadline_misses}/{total} missed)")
    print(f"Infeasible-Budget Rate: {(infeasible_samples / total) * 100.0:.2f}% ({infeasible_samples}/{total})")
    print(f"Feasible-Budget Miss Rate: {(feasible_misses / max(1, total - infeasible_samples)) * 100.0:.2f}%")
    print(f"Cold-Transition Misses: {cold_misses} | Warm Misses: {warm_misses}")
    print(f"K_adapt Operating Range: [{min(k_adapt_history):.3f}, {max(k_adapt_history):.3f}]")
    print("\nConfiguration Selection Breakdown:")
    for cfg, count in sorted(config_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  Config {cfg} selected: {count} times")

    results = {
        "seed": seed,
        "trace": args.trace,
        "latency_policy": args.policy,
        "interpolated_frontier": args.interpolate,
        "thread_count": thread_count,
        "device": str(device),
        "hardware_profile": profile_csv_path,
        "hardware_fingerprint": {
            "device_speed_score": float(hw_fingerprint["device_speed_score"]),
            "cpu_cores": int(hw_fingerprint["cpu_cores"]),
            "cpu_cores_logical": int(hw_fingerprint.get("cpu_cores_logical", 0)),
            "ram_gb": float(hw_fingerprint["ram_gb"]),
            "has_cuda": float(hw_fingerprint["has_cuda"]),
            "arch": hw_fingerprint.get("arch"),
        },
        "total_samples": total,
        "accuracy_percent": float(accuracy),
        "average_latency_ms": float(avg_latency),
        "p95_latency_ms": float(p95_latency),
        "p99_latency_ms": float(p99_latency),
        "prediction_mae_ms": float(mae_error),
        "deadline_miss_rate_percent": float(miss_rate),
        "deadline_misses": deadline_misses,
        "infeasible_budget_count": infeasible_samples,
        "infeasible_budget_rate_percent": float((infeasible_samples / total) * 100.0),
        "feasible_budget_miss_count": feasible_misses,
        "feasible_budget_miss_rate_percent": float((feasible_misses / max(1, total - infeasible_samples)) * 100.0),
        "cold_transition_misses": cold_misses,
        "warm_misses": warm_misses,
        "config_counts": {str(config): count for config, count in config_counts.items()},
        "samples": sample_records,
    }
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(
        results_dir,
        f"evaluation_{args.policy}_{args.trace}_seed{seed}_threads{thread_count}.json",
    )
    temp_results_path = results_path + ".tmp"
    with open(temp_results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=json_default)
        f.write("\n")
    os.replace(temp_results_path, results_path)
    print(f"Detailed results written to: {results_path}")

if __name__ == "__main__":
    main()
