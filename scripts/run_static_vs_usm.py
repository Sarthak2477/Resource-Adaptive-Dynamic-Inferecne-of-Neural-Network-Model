import os
import sys
import subprocess
import argparse
import csv
import json
import re
import platform
import hashlib
from datetime import datetime
from pathlib import Path

import torch
import psutil
import numpy as np
import torchvision

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from resource_control.controller import CANONICAL_EXPERIMENT_PROTOCOL


def read_profile_rows(profile_path):
    with Path(profile_path).open("r", newline="", encoding="utf-8-sig") as profile_file:
        return list(csv.DictReader(profile_file))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_optional_command(command):
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    return result.stdout.strip() or None


def write_run_manifests(run_root, run_id, args):
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None

    selected_device = getattr(args, "device", "auto")
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    if selected_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    environment = {
        "run_id": run_id,
        "repository_commit": commit,
        "repository_worktree_dirty": bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=project_root,
            capture_output=True, text=True, check=False,
        ).stdout.strip()),
        "os": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device": selected_device,
        "gpu_name": torch.cuda.get_device_name(0) if selected_device == "cuda" else None,
        "gpu_driver_and_telemetry_start": capture_optional_command([
            "nvidia-smi", "--query-gpu=driver_version,temperature.gpu,power.draw,power.limit",
            "--format=csv,noheader",
        ]) if selected_device == "cuda" else None,
        "windows_power_scheme": capture_optional_command(
            ["powercfg", "/getactivescheme"]
        ) if os.name == "nt" else None,
        "cpu": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "ram_total_gb": psutil.virtual_memory().total / (1024 ** 3),
        "threads": args.threads,
        "batch_size": 1,
        "bn_calibration_split": "train",
        "bn_calibration_batches_per_width": args.bn_calibration_batches,
        "usm_checkpoint_path": str(Path(args.usm_checkpoint).resolve()),
        "usm_checkpoint_sha256": sha256_file(args.usm_checkpoint),
    }
    with (run_root / "environment.json").open("x", encoding="utf-8") as output:
        json.dump(environment, output, indent=2)
        output.write("\n")

    policy = getattr(args, "policy", "p95")
    warmup_requests = getattr(args, "warmup_requests", 0)
    protocol = {
        "comparison": "USM pinned FP32 versus USM adaptive FP32",
        "precision_bits": CANONICAL_EXPERIMENT_PROTOCOL["precision_bits"],
        "candidate_widths": CANONICAL_EXPERIMENT_PROTOCOL["candidate_widths"],
        "controller_policy": policy,
        "workload_trace": getattr(args, "trace", "sinusoidal"),
        "target_deadline_ms_for_pinned_width_calibration": args.target_deadline_ms,
        "controller_safety_margin": CANONICAL_EXPERIMENT_PROTOCOL["safety_margin"],
        "switching_penalty_ms": CANONICAL_EXPERIMENT_PROTOCOL["switching_penalty_ms"],
        "minimum_dwell_seconds": CANONICAL_EXPERIMENT_PROTOCOL["minimum_dwell_seconds"],
        "fallback": CANONICAL_EXPERIMENT_PROTOCOL["fallback"],
        "pinned_width_calibration": {
            "selection": "widest width whose worst measured condition P95 is at or below target deadline",
            "no_feasible_width_fallback": "lowest worst-condition profiled-P95 width; marked calibration_target_feasible=false in config.json",
        },
        "warmup_policy": {
            "mode": "exclude_from_latency",
            "count": warmup_requests,
            "pinned": "repeat_pinned_width",
            "adaptive": "repeat_controller_initial_width_only",
            "excluded_from_request_latency": True,
        },
        "cold_switch_cost_included": False,
        "width_switch_cost_included": True,
        "resource_state_semantics": {
            "controller_input": "controlled_condition_profile_lookup",
            "condition_source": "workload schedule enables a measured CPU and concurrent GPU stressor during contention intervals",
            "latency_prediction": "measured per-width selected-percentile latency from matching baseline or controlled-contention profile; no shared heuristic multiplier",
            "live_telemetry": "CPU, memory, and available thermal telemetry are recorded and passed in ResourceState but do not alter condition-profile predictions",
            "validation_limit": "controlled synthetic stress is a reproducible lab condition, not a substitute for measurements under deployment workloads",
        },
        "condition_subgroup_analysis": "paired subgroups use each request's frozen scheduled resource scenario; controller prediction uses the measured profile for the condition active at service time",
        "timing": {
            "model_only": "model forward; CUDA synchronized before stop",
            "controller_overhead": "measured separately; included in end_to_end_latency_ms",
            "end_to_end": "scheduled arrival to top-1 response, including queue delay, data retrieval, controller, width assignment, and model forward",
            "deadline_evaluated_on": "end_to_end_latency_ms",
            "online_feedback_update": "measured separately and excluded from response deadline because it runs after prediction is ready",
            "switch_timing": "width assignment and forward after each selection are included; width-specific BN calibration warms candidate paths, so process/model cold startup is excluded",
            "batch_size": 1,
        },
        "workload": "paired controlled replay; the frozen budget/condition/arrival trace is reused across arms; contention stress follows the scheduled condition intervals",
        "replay_mode": "paced_sequential_queue_replay",
        "independent_unit": "complete paired repetition",
        "paired_uncertainty": "95% percentile bootstrap confidence interval over paired repetition differences",
        "online_feedback": "per-width latency correction enabled; clipped error ratio [-0.5, 1.0], learning rate 0.05, correction factor clipped [0.5, 3.0]",
        "repetitions": args.repetitions,
        "seed": args.seed,
        "sample_count": args.n_samples,
        "profile_measurements": {
            "repeats_per_width_condition": getattr(args, "profile_repeats", 100),
            "warmup_samples_per_width_condition": getattr(args, "profile_warmup_samples", 20),
            "conditions": ["baseline", "contention"],
        },
        "accuracy_selection": {
            "dataset_split": "test_selection_calibration",
            "samples_per_class": getattr(args, "accuracy_calibration_per_class", 100),
            "sampling": "seeded random stratified selection; indices excluded from replay",
            "replay_examples_overlap": False,
        },
    }
    with (run_root / "protocol.json").open("x", encoding="utf-8") as output:
        json.dump(protocol, output, indent=2)
        output.write("\n")

def main():
    parser = argparse.ArgumentParser(description="Compare fixed full-capacity USM with resource-aware adaptive width-bit USM.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--usm-checkpoint", default=None)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--bn-calibration-batches", type=int, default=10)
    parser.add_argument("--policy", choices=("avg", "p50", "p95", "p99"), default="p95")
    parser.add_argument("--trace", choices=("sinusoidal", "step", "bursty", "heldout"), default="heldout")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if min(args.repetitions, args.threads, args.n_samples, args.bn_calibration_batches) < 1:
        parser.error("repetitions, threads, samples, and BN calibration batches must be positive")
    if args.policy != "p95":
        parser.error("The final adaptive comparison requires --policy p95")

    run_id = args.run_id or datetime.now().strftime("fp32_%Y%m%d_%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        parser.error("run ID may contain only letters, digits, underscores, and hyphens")

    experiment_root = Path(project_root) / "results" / "static_vs_usm_fp32"
    run_root = experiment_root / "runs" / run_id
    if run_root.exists():
        raise FileExistsError(f"Experiment run already exists: {run_root}")
    for directory in ("accuracy", "profiles", "workload", "raw", "analysis"):
        (run_root / directory).mkdir(parents=True, exist_ok=True)

    from scripts.run_usm_qat_experiment import (
        bind_test_workload, capture_environment, create_partition, run_command,
    )
    from scripts.usm_qat_data import stratified_train_partition
    from models.checkpoint_io import resolve_usm_checkpoint

    checkpoint = resolve_usm_checkpoint(args.usm_checkpoint)
    environment = capture_environment(run_id, checkpoint, args.threads, args.device)
    (run_root / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    protocol = {
        "primary_research_question": "Can resource-aware adaptive USM maintain accuracy while reducing latency/deadline violations by dynamically selecting width and bit-width under changing resource conditions?",
        "comparison": "USM Full (1.0, 32) fixed versus USM Adaptive width-bit switching",
        "pinned_configuration": [1.0, 32],
        "adaptive_candidates": [{"width_mult": width, "bit_width": bits} for width, bits in __import__("resource_control.qat_experiment", fromlist=["QAT_CANDIDATES"]).QAT_CANDIDATES],
        "latency_feasibility": "resource-conditioned measured P95; choose the highest-accuracy feasible FP32 width first; consider lower bit-width only when no FP32 width is feasible; if no candidate is feasible, choose minimum predicted P95 and record deadline fallback",
        "precision_policy": "FP32 width adaptation is preferred; precision reduction is a last-resort fallback only when no FP32 width satisfies the P95 safety-margin deadline",
        "resource_state_limitation": "Replay uses measured baseline and low-duty CPU-contention profile labels; no contention process runs during request replay. This is a controlled profile-conditioned scenario, not a live deployment-resource intervention.",
        "profile_protocol": {"warmup_measurements": 10, "measured_repetitions": 20, "conditions": ["baseline", "cpu_contention"], "pinned_extra_benchmarks": 0},
        "accuracy_calibration": "500 examples from a stratified CIFAR-10 train validation subset; final evaluation uses disjoint CIFAR-10 test examples",
        "warmup_policy": "same seeded randomized one-forward-per-candidate warmup for both arms; excluded from request latency",
        "replay_stress": "disabled; no GPU stress process",
        "repetitions": args.repetitions,
        "sample_count": args.n_samples,
        "threads": args.threads,
        "requested_device": args.device,
        "seed": args.seed,
        "primary_latency_quantile": args.policy,
        "trace": args.trace,
    }
    (run_root / "experiment_metadata.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    partition_path = run_root / "accuracy" / "data_partition.json"
    create_partition(Path(project_root) / "cifar10", partition_path, 50, args.seed)
    accuracy_json = run_root / "accuracy" / "candidate_accuracy.json"
    run_command([
        sys.executable, os.path.join(project_root, "scripts", "evaluate_usm_qat_configs.py"),
        "--checkpoint", str(checkpoint), "--partition", str(partition_path),
        "--output", str(accuracy_json), "--dataset-root", os.path.join(project_root, "cifar10"),
        "--batch-size", "64", "--bn-calibration-batches", str(args.bn_calibration_batches),
        "--seed", str(args.seed), "--device", args.device,
    ])

    profile_dir = run_root / "profiles"
    run_command([
        sys.executable, os.path.join(project_root, "scripts", "profile_usm_qat_configs.py"),
        "--checkpoint", str(checkpoint), "--partition", str(partition_path),
        "--output-dir", str(profile_dir), "--dataset-root", os.path.join(project_root, "cifar10"),
        "--threads", str(args.threads), "--samples-per-repetition", "20",
        "--warmup-samples", "10", "--bn-calibration-batches", str(args.bn_calibration_batches),
        "--seed", str(args.seed), "--lightweight", "--device", args.device,
    ])

    pinned_path = run_root / "config.json"
    pinned_path.write_text(json.dumps({
        "width_mult": 1.0, "bit_width": 32,
        "selection_rule": "fixed full-capacity USM; no calibration or switching",
        "checkpoint_sha256": environment["checkpoint_sha256"],
    }, indent=2) + "\n", encoding="utf-8")
    base_workload = run_root / "workload" / "workload_base.json"
    bound_workload = run_root / "workload" / "workload.json"
    run_command([
        sys.executable, os.path.join(project_root, "scripts", "generate_usm_qat_workload.py"),
        "--seed", str(args.seed), "--n-samples", str(args.n_samples),
        "--output", str(base_workload), "--interarrival-ms", "100",
        "--trace", args.trace, "--conditions", "baseline,cpu_contention", "--three-phase",
    ])
    bind_test_workload(base_workload, Path(project_root) / "cifar10", args.seed, bound_workload)

    run_command([
        sys.executable, os.path.join(project_root, "scripts", "experiment_usm_qat.py"),
        "--run-id", run_id, "--run-root", str(run_root),
        "--checkpoint", str(checkpoint), "--partition", str(partition_path),
        "--accuracy-json", str(accuracy_json),
        "--profile-csv", str(profile_dir / "resource_condition_profiles.csv"),
        "--switch-costs", str(profile_dir / "switch_costs.csv"),
        "--pinned-config", str(pinned_path), "--workload", str(bound_workload),
        "--dataset-root", os.path.join(project_root, "cifar10"),
        "--repetitions", str(args.repetitions), "--threads", str(args.threads),
        "--bn-calibration-batches", str(args.bn_calibration_batches),
        "--seed", str(args.seed), "--device", args.device,
    ])

    if args.repetitions == 1 and args.n_samples <= 50:
        request_keys = [(request["request_id"], request["input_sample_id"], request["deadline_ms"], request["resource_condition"], request["dataset_index"]) for request in json.loads(bound_workload.read_text(encoding="utf-8"))["requests"]]
        adaptive_phase_configs = []
        for arm_name in ("usm_pinned", "usm_adaptive"):
            raw_path = run_root / "raw" / f"{arm_name}_{run_id}_rep001.jsonl"
            rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line]
            observed_keys = [(row["request_id"], row["input_sample_id"], row["deadline_ms"], row["scheduled_resource_condition"], row["dataset_index"]) for row in rows]
            if observed_keys != request_keys:
                raise RuntimeError(f"Smoke check failed: {arm_name} request sequence differs from workload")
            if arm_name == "usm_pinned" and any((row["selected_width"], row["selected_bit_width"]) != (1.0, 32) for row in rows):
                raise RuntimeError("Smoke check failed: pinned arm changed from (1.0, 32)")
            if arm_name == "usm_adaptive":
                adaptive_phase_configs = [
                    {(row["selected_width"], row["selected_bit_width"]) for row in rows if row["phase"] == phase}
                    for phase in ("phase_1_relaxed", "phase_2_constrained", "phase_3_recovery")
                ]
        if any(not selected for selected in adaptive_phase_configs) or len({next(iter(selected)) for selected in adaptive_phase_configs}) < 2:
            raise RuntimeError("Smoke check failed: resource/deadline phases did not influence adaptive configuration selection")
        if environment["device"] == "cuda" and torch.cuda.is_available() is False:
            raise RuntimeError("Smoke check failed: device availability changed during the run")
        (run_root / "smoke_validation.json").write_text(json.dumps({
            "pinned_fixed_full": True,
            "shared_request_identity": True,
            "adaptive_configurations_by_phase": [sorted([list(config) for config in phase]) for phase in adaptive_phase_configs],
            "calibration_split": "CIFAR-10 train",
            "evaluation_split": "CIFAR-10 test",
            "replay_stress_enabled": False,
            "resource_conditioning_note": protocol["resource_state_limitation"],
        }, indent=2) + "\n", encoding="utf-8")

    run_command([
        sys.executable, os.path.join(project_root, "scripts", "generate_usm_qat_report.py"),
        "--run-id", run_id, "--run-root", str(run_root),
    ])
    print(f"Experiment artifacts: {run_root}; pinned=(1.0, 32)")

if __name__ == "__main__":
    main()
