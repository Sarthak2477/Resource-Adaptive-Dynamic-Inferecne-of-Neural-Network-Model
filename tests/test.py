import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import pandas as pd

from resource_control.telemetry import ResourceMonitor, ResourceState
from resource_control.controller import SurrogateBackedController
from resource_control.calibration import collect_multi_hardware_data, leave_one_hardware_out_eval


def main():
    print("--- 1. Testing Resource Monitor and Hardware Fingerprint ---")
    monitor = ResourceMonitor()
    print("Live Telemetry:", monitor.read())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fingerprint = monitor.get_hardware_fingerprint(device)
    print("Hardware Fingerprint:", fingerprint)

    print("\n--- 2. Testing Surrogate-Backed Controller ---")
    frontier = [
        {'config': (16, 8),  'latency_ms': 10.0, 'acc': 70.0},
        {'config': (16, 16), 'latency_ms': 12.0, 'acc': 68.0},
        {'config': (32, 8),  'latency_ms': 18.0, 'acc': 78.0},
        {'config': (32, 16), 'latency_ms': 25.0, 'acc': 82.0},
    ]

    controller = SurrogateBackedController(frontier=frontier, min_dwell_s=0.0)

    state_normal = ResourceState(cpu_pct=20, mem_available_mb=2048, battery_pct=90, thermal_c=45, timestamp=time.time())
    cfg_normal = controller.select(state_normal, latency_budget_ms=30.0, hw_fingerprint=fingerprint)
    print("Surrogate select (High budget):", cfg_normal)

    state_stressed = ResourceState(cpu_pct=95, mem_available_mb=2048, battery_pct=90, thermal_c=45, timestamp=time.time())
    cfg_stressed = controller.select(state_stressed, latency_budget_ms=12.0, hw_fingerprint=fingerprint)
    print("Surrogate select (Low budget + high CPU stress):", cfg_stressed)

    print("\n--- 3. Testing Multi-Hardware Calibration Profiling ---")
    class MockSupernet:
        def set_width(self, w): pass
        def set_bit_width(self, bw): pass
        def __call__(self, x): return x

    model = MockSupernet()
    configs = [(16, 8), (16, 16), (32, 8), (32, 16)]

    print("Profiling Environment A...")
    df_a = collect_multi_hardware_data(model, device, configs, "device_A")
    print(df_a[["config", "latency_ms", "flops_per_speed", "hw_name"]])

    print("\nProfiling Environment B...")
    slow_fingerprint = fingerprint.copy()
    slow_fingerprint["device_speed_score"] *= 0.5

    df_b = collect_multi_hardware_data(model, device, configs, "device_B")
    df_b["device_speed_score"] = slow_fingerprint["device_speed_score"]
    df_b["flops_per_speed"] = df_b["approx_flops"] / slow_fingerprint["device_speed_score"]
    df_b["latency_ms"] *= 2.0
    print(df_b[["config", "latency_ms", "flops_per_speed", "hw_name"]])

    print("\n--- 4. Testing Leave-One-Hardware-Out Evaluation Protocol ---")
    all_data = pd.concat([df_a, df_b], ignore_index=True)
    eval_results = leave_one_hardware_out_eval(all_data, held_out_hw="device_B", calibration_sizes=[0, 2])
    print(eval_results)


if __name__ == "__main__":
    main()
