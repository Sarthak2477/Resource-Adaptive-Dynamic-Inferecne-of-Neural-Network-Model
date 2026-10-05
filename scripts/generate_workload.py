import os
import json
import argparse
import numpy as np
import time

def generate_workload(seed, n_samples, trace, filepath):
    np.random.seed(seed)
    requests = []
    for idx in range(n_samples):
        phase = (2.0 * np.pi * idx) / 40.0
        if trace == "sinusoidal":
            budget = 20.0 + 12.0 * np.sin(phase) + np.random.uniform(-1.0, 1.0)
        elif trace == "step":
            budget = (12.0 if idx < 25 else 28.0 if idx < 50 else 8.0 if idx < 75 else 24.0) + np.random.uniform(-0.5, 0.5)
        elif trace == "bursty":
            budget = 24.0 + np.random.uniform(-2.0, 2.0)
            if idx % 10 in (0, 1, 2):
                budget -= 14.0
        elif trace == "heldout":
            budget = 18.0 + 10.0 * np.sin(phase * 1.7 + 0.8) + np.random.uniform(-2.0, 2.0)
        budget = max(4.0, budget)
        arrival_s = idx * 0.01
        scenario_phase = idx % 100
        requests.append({
            "request_id": f"req_{idx+1:04d}",
            "sample_index": int(idx),
            "input_sample_id": f"cifar10_test_order_{idx:05d}",
            "arrival_s": float(arrival_s),
            "interarrival_s": 0.01 if idx else 0.0,
            "batch_size": 1,
            "budget_ms": float(budget),
            "resource_scenario": "contention" if 30 <= scenario_phase < 70 else "baseline",
        })
    data = {
        "meta": {
            "kind": "synthetic_controlled_replay",
            "replay_mode": "deterministic_sequential_replay",
            "seed": seed,
            "n_samples": n_samples,
            "trace": trace,
            "arrival_model": "fixed 10 ms inter-arrival; replay waits for scheduled arrivals and records queue delay",
            "resource_conditions": "baseline/contention phases; replay enables a controlled CPU/GPU stressor during contention",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "requests": requests
    }
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Workload saved to {filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--trace", type=str, default="sinusoidal")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    generate_workload(args.seed, args.n_samples, args.trace, args.output)
