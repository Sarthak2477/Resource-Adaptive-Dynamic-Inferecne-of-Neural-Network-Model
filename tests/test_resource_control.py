import os
import sys
import time
import torch

# Ensure resource_controller and model directories can be imported correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'resource_controller')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from telemetry import ResourceMonitor, ResourceState
from controller import SurrogateBackedController
from model import Model, set_model_width, set_model_bit_width

def print_banner(msg):
    print("\n" + "=" * 60)
    print(f" {msg} ".center(60, "="))
    print("=" * 60)

def main():
    print_banner("1. Initializing Model and Resource Monitor")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize the universally slimmable model
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
            
            # Remove 'module.' prefix from keys if model was saved using DataParallel
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
    print("Model initialized successfully.")

    # Initialize the resource monitor
    monitor = ResourceMonitor()
    live_state = monitor.read()
    hw_fingerprint = monitor.get_hardware_fingerprint(device)
    print(f"Live Telemetry State: {live_state}")
    print(f"Hardware Fingerprint: {hw_fingerprint}")

    print_banner("2. Initializing Surrogate-Backed Resource Controller")
    
    # Define a Pareto frontier of (width_mult, bit_width) options and their expected accuracy.
    # The accuracies match the trained results from the CIFAR-10 ResNet-50 notebook.
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

    # Instantiate controller (uses physics-based fallback if no ML model has been trained)
    controller = SurrogateBackedController(frontier=frontier, min_dwell_s=1.0)
    print("Surrogate-Backed Controller initialized.")

    print_banner("3. Running Adaptation Simulation Under Various Conditions")

    # Generate a dummy input matching CIFAR-10 shape
    dummy_input = torch.randn(1, 3, 32, 32, device=device)

    # Helper function to run a mock inference step and print the stats
    def test_step(scenario_name, state, budget_ms):
        print(f"\nScenario: {scenario_name}")
        print(f"Current State -> CPU: {state.cpu_pct}%, Temp: {state.thermal_c}°C, Mem Available: {state.mem_available_mb:.1f} MB")
        print(f"Latency Budget: {budget_ms} ms")

        # 1. Ask the controller for the best configuration
        selected_cfg = controller.select(state, latency_budget_ms=budget_ms, hw_fingerprint=hw_fingerprint)
        print(f"Controller selected configuration: {selected_cfg}")

        # Find the accuracy entry for this config
        entry = next(f for f in frontier if f['config'] == selected_cfg)
        print(f"Expected Acc: {entry['acc']}% (Nominal latency on reference hardware: {entry['latency_ms']} ms)")

        # 2. Apply configuration to the model
        set_model_width(model, selected_cfg[0])
        set_model_bit_width(model, selected_cfg[1])

        # 3. Warm up the configuration
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

        # 4. Measure execution latency
        latencies = []
        for _ in range(5):
            t_start = time.perf_counter()
            with torch.no_grad():
                _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t_start) * 1000.0) # ms

        actual_lat = sum(latencies) / len(latencies)
        print(f"Actual model inference latency measured: {actual_lat:.2f} ms")

    # Scenario A: Normal load, generous budget
    state_normal = ResourceState(
        cpu_pct=15.0,
        mem_available_mb=2048.0,
        battery_pct=90,
        thermal_c=45.0,
        timestamp=time.time()
    )
    test_step("Generous budget, low CPU load", state_normal, budget_ms=30.0)

    # Scenario B: Normal load, tight budget
    # We advance time to bypass the dwell time hysteresis
    time.sleep(1.1)
    test_step("Tight budget, low CPU load", state_normal, budget_ms=10.0)

    # Scenario C: High CPU and thermal stress (throttling simulation)
    # We expect the physics surrogate to scale up latency predictions and select a smaller/faster config
    time.sleep(1.1)
    state_stressed = ResourceState(
        cpu_pct=90.0,
        mem_available_mb=512.0,
        battery_pct=30,
        thermal_c=82.0,
        timestamp=time.time()
    )
    test_step("Tight budget, high CPU + thermal stress", state_stressed, budget_ms=10.0)

    # Scenario D: Extremely tight budget under stress
    time.sleep(1.1)
    test_step("Extreme budget under stress", state_stressed, budget_ms=4.0)

if __name__ == "__main__":
    main()
