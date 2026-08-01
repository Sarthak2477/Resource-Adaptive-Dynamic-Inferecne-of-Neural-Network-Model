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
    
    frontier = [
        {'config': (0.25, 4), 'latency_ms': 5.0, 'acc': 89.69},
        {'config': (0.25, 8), 'latency_ms': 5.5, 'acc': 90.46},
        {'config': (0.25, 16), 'latency_ms': 5.8, 'acc': 90.57},
        {'config': (0.25, 32), 'latency_ms': 6.0, 'acc': 90.59},
        {'config': (0.5, 4), 'latency_ms': 10.0, 'acc': 90.19},
        {'config': (0.5, 8), 'latency_ms': 11.2, 'acc': 91.83},
        {'config': (0.5, 16), 'latency_ms': 11.5, 'acc': 91.79},
        {'config': (0.5, 32), 'latency_ms': 12.0, 'acc': 91.83},
        {'config': (0.75, 4), 'latency_ms': 16.5, 'acc': 90.51},
        {'config': (0.75, 8), 'latency_ms': 18.0, 'acc': 92.12},
        {'config': (0.75, 16), 'latency_ms': 18.5, 'acc': 92.16},
        {'config': (0.75, 32), 'latency_ms': 19.0, 'acc': 92.18},
        {'config': (1.0, 4), 'latency_ms': 22.0, 'acc': 90.66},
        {'config': (1.0, 8), 'latency_ms': 24.5, 'acc': 92.32},
        {'config': (1.0, 16), 'latency_ms': 25.0, 'acc': 92.37},
        {'config': (1.0, 32), 'latency_ms': 26.0, 'acc': 92.37},
    ]

    # Set min_dwell_s=0.0 to allow immediate adaptation per sample
    controller = SurrogateBackedController(frontier=frontier, min_dwell_s=0.0)

    print_banner("4. Starting Resource-Controlled Evaluation")
    
    correct = 0
    total = 0
    inference_times = []
    config_counts = {}
    
    # Simulate dynamic latency budget and CPU utilization cycles during the 100-sample run
    # Sample index 0-24: 35ms budget, low CPU load (generous budget)
    # Sample index 25-49: 12ms budget, low CPU load (tight budget)
    # Sample index 50-74: 12ms budget, 85% CPU load (severe contention)
    # Sample index 75-99: 4ms budget, low CPU load (extreme budget constraint)

    for idx, (image, label) in enumerate(eval_loader):
        image, label = image.to(device), label.to(device)
        
        # Determine simulated state based on current progress
        if idx < 25:
            budget = 35.0
            cpu_pct = 15.0
            thermal_c = 45.0
            scenario = "Generous"
        elif idx < 50:
            budget = 12.0
            cpu_pct = 15.0
            thermal_c = 45.0
            scenario = "Tight"
        elif idx < 75:
            budget = 12.0
            cpu_pct = 85.0
            thermal_c = 78.0
            scenario = "Stressed"
        else:
            budget = 4.0
            cpu_pct = 15.0
            thermal_c = 45.0
            scenario = "Extreme"

        state = ResourceState(
            cpu_pct=cpu_pct,
            mem_available_mb=1024.0,
            battery_pct=80,
            thermal_c=thermal_c,
            timestamp=time.time()
        )

        # 1. Select configuration
        selected_cfg = controller.select(state, latency_budget_ms=budget, hw_fingerprint=hw_fingerprint)
        config_counts[selected_cfg] = config_counts.get(selected_cfg, 0) + 1

        # 2. Set config on model
        set_model_width(model, selected_cfg[0])
        set_model_bit_width(model, selected_cfg[1])

        # 2b. Lazy BN statistics calibration if needed
        key = (selected_cfg[0], selected_cfg[1])
        from model.ops import ResnetBatchNorm2d
        first_bn = next(m for m in model.modules() if isinstance(m, ResnetBatchNorm2d))
        if key not in first_bn.calibrated_running_mean:
            print(f"\n[BN Calibration] Lazily calibrating BN statistics for {key} on 10 batches...")
            recalibrate_bn(model, train_loader, selected_cfg[0], selected_cfg[1], device, num_batches=10)
            print("[BN Calibration] Done!")

        # 3. Inference & Time measurement
        t0 = time.perf_counter()
        with torch.no_grad():
            output = model(image)
            if device.type == "cuda":
                torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0 # ms
        inference_times.append(dt)

        # 4. Accuracy tracking
        pred = output.argmax(dim=1)
        is_correct = (pred == label).item()
        if is_correct:
            correct += 1
        total += 1

        # Print progress every 10 samples
        if (idx + 1) % 10 == 0:
            print(f"Sample {idx+1}/100 | Scenario: {scenario} | Budget: {budget}ms | Selected: {selected_cfg} | Actual Latency: {dt:.2f}ms | Correct: {is_correct}")

    print_banner("5. Evaluation Summary")
    avg_latency = sum(inference_times) / len(inference_times)
    accuracy = (correct / total) * 100.0
    
    print(f"Total Samples Evaluated: {total}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Average Inference Latency: {avg_latency:.2f} ms")
    print("\nConfiguration Selection Breakdown:")
    for cfg, count in sorted(config_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  Config {cfg} selected: {count} times")

if __name__ == "__main__":
    main()
