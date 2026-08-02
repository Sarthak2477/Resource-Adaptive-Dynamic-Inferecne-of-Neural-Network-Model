import os
import sys
import time
import torch
from torch.utils.data import Subset

# Set up paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'resource_controller')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from telemetry import ResourceMonitor, ResourceState
from controller import SurrogateBackedController
from model import Model, set_model_width, set_model_bit_width, get_dataloaders

def print_banner(msg):
    print("\n" + "=" * 60)
    print(f" {msg} ".center(60, "="))
    print("=" * 60)

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
            else:
                frac = (w - pt_lower['width']) / (pt_upper['width'] - pt_lower['width'])
                acc = pt_lower['acc'] + frac * (pt_upper['acc'] - pt_lower['acc'])
                lat = pt_lower['latency_ms'] + frac * (pt_upper['latency_ms'] - pt_lower['latency_ms'])
                std = pt_lower['std_ms'] + frac * (pt_upper['std_ms'] - pt_lower['std_ms'])
                
            dense_frontier.append({
                'config': (w, b),
                'latency_ms': lat,
                'std_ms': std,
                'acc': acc
            })
            
    # Sort dense frontier by latency
    return sorted(dense_frontier, key=lambda x: x['latency_ms'])

def main():
    print_banner("1. Initializing Model and Loading Checkpoint")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = Model(num_classes=10, input_size=32)
    
    checkpoint_paths = [
        "./model/checkpoint/us_resnet_epoch100_checkpoint.pt",
        "./model/checkpoints/best_model.pt",
        "./model/checkpoints/us_resnet_epoch100_checkpoint.pt",
        "us_resnet_epoch100_checkpoint.pt"
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
    from model import recalibrate_bn
    
    # Select a balanced subset of 100 samples (10 from each of the 10 classes)
    subset_indices = []
    for class_idx in range(10):
        start_idx = class_idx * 1000
        subset_indices.extend(range(start_idx, start_idx + 10))
        
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
    from profile_hardware import get_clean_hardware_name
    hw_name = get_clean_hardware_name(device)
    profile_csv_path = f"profiles/{hw_name}_profile.csv"
    
    # Subnet accuracies are hardware-independent
    CONFIG_ACCURACIES = {
        (0.25, 4): 89.69, (0.25, 8): 90.46, (0.25, 16): 90.57, (0.25, 32): 90.59,
        (0.5, 4): 90.19,  (0.5, 8): 91.83,  (0.5, 16): 91.79,  (0.5, 32): 91.83,
        (0.75, 4): 90.51, (0.75, 8): 92.12, (0.75, 16): 92.16, (0.75, 32): 92.18,
        (1.0, 4): 90.66,  (1.0, 8): 92.32,  (1.0, 16): 92.37,  (1.0, 32): 92.37,
    }
    
    frontier = []
    if os.path.exists(profile_csv_path):
        print(f"Loading dynamic latency profile from: {profile_csv_path}")
        import csv
        with open(profile_csv_path, mode="r") as f:
            reader = csv.DictReader(f)
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

    # Expand discrete frontier to a continuous frontier (0.05 step width multipliers)
    frontier = expand_frontier_continuous(frontier, step=0.05)
    print(f"Expanded Pareto frontier to {len(frontier)} continuous candidate configurations.")

    # Load trained surrogate model if available
    surrogate_model = None
    surrogate_path = "resource_controller/surrogate_model.pkl"
    if os.path.exists(surrogate_path):
        print(f"Loading trained learned surrogate model from: {surrogate_path}")
        import pickle
        with open(surrogate_path, "rb") as f:
            surrogate_model = pickle.load(f)
    else:
        print("Trained surrogate model not found. Falling back to physics-based surrogate model.")

    # Set min_dwell_s=0.0 to allow immediate adaptation per sample
    controller = SurrogateBackedController(
        frontier=frontier, 
        surrogate_model=surrogate_model, 
        min_dwell_s=0.0,
        k_risk=1.2,
        switching_penalty_ms=1.5
    )

    print_banner("4. Starting Resource-Controlled Evaluation")
    
    import numpy as np
    
    correct = 0
    total = 0
    inference_times = []
    config_counts = {}
    prediction_errors = []
    k_adapt_history = []
    deadline_misses = 0
    
    # We will simulate 100 evaluation steps:
    # - Budget: Fluctuating dynamically using a sinusoidal wave (mean=20ms, amplitude=12ms, period=40 steps)
    # - Contention:
    #     Steps 0-30: Normal baseline load (cpu_pct=15%, thermal=45C)
    #     Steps 30-70: Contention Spike (cpu_pct=85%, thermal=80C)
    #     Steps 70-100: Return to normal load (cpu_pct=15%, thermal=50C)

    for idx, (image, label) in enumerate(eval_loader):
        image, label = image.to(device), label.to(device)
        
        # 1. Generate Sinusoidal Budget with minor random jitter
        base_budget = 20.0
        amplitude = 12.0
        period = 40.0
        phase = (2.0 * np.pi * idx) / period
        budget = base_budget + amplitude * np.sin(phase) + np.random.uniform(-1.0, 1.0)
        budget = max(4.0, budget)  # Enforce minimum budget of 4.0ms
        
        # 2. Simulate hardware contention state
        if idx >= 30 and idx < 70:
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

        # 4. Set config on model
        set_model_width(model, selected_cfg[0])
        set_model_bit_width(model, selected_cfg[1])

        # 4b. Lazy BN statistics calibration if needed
        key = (selected_cfg[0], selected_cfg[1])
        from model.ops import ResnetBatchNorm2d
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
            controller.update_feedback(dt, pred_latency)
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

        # Print progress logs
        if (idx + 1) % 10 == 0:
            print(
                f"Sample {idx+1:03d}/100 | {scenario:<16} | "
                f"Budget: {budget:4.1f}ms | Pred: {pred_latency or 0.0:4.1f}ms | "
                f"Actual: {dt:4.1f}ms | K_adapt: {controller.k_adapt:.3f} | "
                f"Miss: {str(missed_deadline):5} | Selected: {str(selected_cfg):8} | Correct: {is_correct}"
            )

    print_banner("5. Evaluation Summary")
    avg_latency = sum(inference_times) / len(inference_times)
    accuracy = (correct / total) * 100.0
    mae_error = np.mean(prediction_errors) if prediction_errors else 0.0
    miss_rate = (deadline_misses / total) * 100.0
    
    print(f"Total Samples Evaluated: {total}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Average Inference Latency: {avg_latency:.2f} ms")
    print(f"Mean Absolute Prediction Error (MAE): {mae_error:.2f} ms")
    print(f"Deadline Miss Rate: {miss_rate:.2f}% ({deadline_misses}/{total} missed)")
    print(f"K_adapt Operating Range: [{min(k_adapt_history):.3f}, {max(k_adapt_history):.3f}]")
    print("\nConfiguration Selection Breakdown:")
    for cfg, count in sorted(config_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  Config {cfg} selected: {count} times")

if __name__ == "__main__":
    main()
