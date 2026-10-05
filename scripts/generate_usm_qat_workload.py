import argparse
import json
from pathlib import Path
import time

import numpy as np

THREE_PHASE_DEADLINES_MS = {
    "phase_1_relaxed": 20.0,
    "phase_2_constrained": 10.0,
    "phase_3_recovery": 20.0,
}


def generate_workload(seed, n_samples, output, interarrival_ms, trace, conditions, three_phase=False):
    if n_samples < 1 or interarrival_ms <= 0:
        raise ValueError("n_samples and interarrival_ms must be positive")
    if not conditions or conditions[0] != "baseline":
        raise ValueError("conditions must start with baseline")
    if len(set(conditions)) != len(conditions):
        raise ValueError("conditions must be unique")
    rng = np.random.default_rng(seed)
    requests = []
    condition_block = max(1, n_samples // max(1, len(conditions) * 2))
    for index in range(n_samples):
        phase = (2.0 * np.pi * index) / 40.0
        if trace == "sinusoidal":
            budget_ms = 20.0 + 12.0 * np.sin(phase) + rng.uniform(-1.0, 1.0)
        elif trace == "step":
            budget_ms = (12.0 if index % 100 < 25 else 28.0 if index % 100 < 50 else 8.0 if index % 100 < 75 else 24.0) + rng.uniform(-0.5, 0.5)
        elif trace == "bursty":
            budget_ms = 24.0 + rng.uniform(-2.0, 2.0)
            if index % 10 in (0, 1, 2):
                budget_ms -= 14.0
        else:
            budget_ms = 18.0 + 10.0 * np.sin(phase * 1.7 + 0.8) + rng.uniform(-2.0, 2.0)
        budget_ms = max(4.0, float(budget_ms))
        if three_phase:
            phase_index = 0 if index < n_samples // 3 else 1 if index < 2 * n_samples // 3 else 2
            phase_name, condition = (
                ("phase_1_relaxed", "baseline"),
                ("phase_2_constrained", "cpu_contention"),
                ("phase_3_recovery", "baseline"),
            )[phase_index]
            deadline_ms = THREE_PHASE_DEADLINES_MS[phase_name]
        else:
            phase_index = None
            phase_name = None
            condition_index = (index // condition_block) % len(conditions)
            condition = conditions[condition_index]
            deadline_ms = budget_ms
        request = {
            "request_id": f"qat_req_{index + 1:06d}",
            "sample_index": index,
            "input_sample_id": f"cifar10_test_holdout_{index:06d}",
            "arrival_s": index * interarrival_ms / 1000.0,
            "interarrival_s": interarrival_ms / 1000.0 if index else 0.0,
            "batch_size": 1,
            "deadline_ms": deadline_ms,
            "resource_condition": condition,
        }
        if three_phase:
            request["phase"] = phase_name
        requests.append(request)
    document = {
        "schema_version": 1,
        "meta": {
            "kind": "controlled_reproducible_laboratory_contention",
            "replay_mode": "paced_single_server_queue",
            "seed": seed,
            "sample_count": n_samples,
            "trace": trace,
            "interarrival_ms": interarrival_ms,
            "resource_conditions": conditions,
            "three_phase_protocol": three_phase,
            "phase_deadlines_ms": THREE_PHASE_DEADLINES_MS if three_phase else None,
            "condition_semantics": "baseline has no stressor; cpu_contention and gpu_contention use isolated in-process matrix-multiply stressors",
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "requests": requests,
    }
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite workload artifact: {output}")
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return document


def main():
    parser = argparse.ArgumentParser(description="Generate one immutable paired QAT workload.")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interarrival-ms", type=float, default=100.0)
    parser.add_argument("--trace", choices=("sinusoidal", "step", "bursty", "heldout"), default="heldout")
    parser.add_argument("--conditions", default="baseline,cpu_contention,gpu_contention")
    parser.add_argument("--three-phase", action="store_true")
    args = parser.parse_args()
    generate_workload(
        args.seed, args.n_samples, args.output, args.interarrival_ms,
        args.trace, [item.strip() for item in args.conditions.split(",") if item.strip()],
        args.three_phase,
    )


if __name__ == "__main__":
    main()