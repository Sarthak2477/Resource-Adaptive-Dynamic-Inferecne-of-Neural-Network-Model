import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import torch
import torchvision

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.checkpoint_io import checkpoint_sha256, resolve_usm_checkpoint
from resource_control.qat_experiment import (
    QAT_CANDIDATES,
    select_pinned_configuration,
    validate_accuracy_table,
    validate_latency_table,
)
from scripts.usm_qat_data import make_test_imagefolder, make_train_imagefolders, stratified_train_partition


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows available for {path}")
    with Path(path).open("x", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def run_command(command):
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def create_partition(dataset_root, path, validation_per_class, seed):
    train_dataset, _ = make_train_imagefolders(dataset_root)
    validation_indices, bn_indices = stratified_train_partition(
        train_dataset.targets, len(train_dataset.classes), validation_per_class, seed,
    )
    document = {
        "source_split": "CIFAR-10 train",
        "seed": seed,
        "validation_per_class": validation_per_class,
        "validation_indices": validation_indices,
        "bn_calibration_indices": bn_indices,
        "validation_indices_sha256": sha256_text(",".join(map(str, validation_indices))),
        "bn_indices_sha256": sha256_text(",".join(map(str, bn_indices))),
        "disjoint": not bool(set(validation_indices) & set(bn_indices)),
    }
    if not document["disjoint"] or len(validation_indices) + len(bn_indices) != len(train_dataset):
        raise RuntimeError("Train validation/BN partition failed coverage or disjointness checks")
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return document


def capture_environment(run_id, checkpoint, threads, device_name="auto"):
    if device_name not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    selected_device = "cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu" if device_name == "auto" else device_name
    gpu_name = torch.cuda.get_device_name(0) if selected_device == "cuda" else None
    return {
        "run_id": run_id,
        "repository_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=False,
        ).stdout.strip() or None,
        "worktree_dirty": bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
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
        "gpu_name": gpu_name,
        "cpu": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "ram_total_gb": psutil.virtual_memory().total / (1024 ** 3),
        "threads": threads,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "inference_backend": "PyTorch QAT fake quantization; not integer INT4/INT8 kernels",
    }


def bind_test_workload(workload_path, dataset_root, seed, output_path):
    document = json.loads(Path(workload_path).read_text(encoding="utf-8"))
    test_dataset = make_test_imagefolder(dataset_root)
    by_class = {index: [] for index in range(len(test_dataset.classes))}
    for dataset_index, target in enumerate(test_dataset.targets):
        by_class[int(target)].append(dataset_index)
    for values in by_class.values():
        values.sort()
    maximum = min(map(len, by_class.values()))
    dataset_indices = [
        by_class[class_index][offset]
        for offset in range(maximum)
        for class_index in range(len(test_dataset.classes))
    ]
    requests = document["requests"]
    if len(requests) > len(dataset_indices):
        raise ValueError("Workload is larger than the held-out CIFAR-10 test split")
    dataset_indices = dataset_indices[:len(requests)]
    for request, dataset_index in zip(requests, dataset_indices):
        request["dataset_index"] = dataset_index
        request["dataset_image_path"] = test_dataset.samples[dataset_index][0]
    document["dataset_indices"] = dataset_indices
    document["meta"].update({
        "dataset_split": "CIFAR-10 test held-out evaluation",
        "accuracy_selection_split": "CIFAR-10 train validation partition",
        "bn_calibration_split": "complementary CIFAR-10 train partition",
        "dataset_order": "class-interleaved test order",
        "dataset_mapping_sha256": sha256_text(",".join(map(str, dataset_indices))),
    })
    output_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run the separate 16-candidate USM QAT resource-adaptive experiment.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "cifar10")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--target-p95-ms", type=float, default=30.0)
    parser.add_argument("--validation-per-class", type=int, default=100)
    parser.add_argument("--bn-calibration-batches", type=int, default=10)
    parser.add_argument("--profile-repetitions", type=int, default=6)
    parser.add_argument("--profile-holdout-repetitions", type=int, default=2)
    parser.add_argument("--profile-samples-per-repetition", type=int, default=20)
    parser.add_argument("--profile-warmup-samples", type=int, default=5)
    parser.add_argument("--interarrival-ms", type=float, default=100.0)
    parser.add_argument("--trace", choices=("sinusoidal", "step", "bursty", "heldout"), default="heldout")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        args.repetitions = 1
        args.n_samples = 30
        args.validation_per_class = min(args.validation_per_class, 5)
        args.bn_calibration_batches = 1
        args.profile_repetitions = 3
        args.profile_holdout_repetitions = 1
        args.profile_samples_per_repetition = 3
        args.profile_warmup_samples = 1
        args.interarrival_ms = 100.0
    if min(args.repetitions, args.n_samples, args.threads, args.validation_per_class,
           args.bn_calibration_batches, args.profile_repetitions,
           args.profile_samples_per_repetition) < 1 or args.interarrival_ms <= 0:
        parser.error("counts must be positive and interarrival-ms must be positive")
    if args.profile_holdout_repetitions < 1 or args.profile_holdout_repetitions >= args.profile_repetitions:
        parser.error("profile holdout repetitions must be positive and less than profile repetitions")

    checkpoint = resolve_usm_checkpoint(args.checkpoint)
    run_id = args.run_id or datetime.now().strftime("usm_qat_resource_adaptive_%Y%m%d_%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        parser.error("run ID may contain only letters, digits, underscores, and hyphens")
    runs_root = PROJECT_ROOT / "results" / "usm_qat_resource_adaptive" / "runs"
    run_root = runs_root / run_id
    if run_root.exists():
        raise FileExistsError(f"Refusing to overwrite QAT run directory: {run_root}")
    for directory in ("accuracy", "profiles", "workload", "raw", "analysis"):
        (run_root / directory).mkdir(parents=True, exist_ok=True)

    env = capture_environment(run_id, checkpoint, args.threads, args.device)
    (run_root / "environment.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    conditions = ["baseline", "cpu_contention"] + (["gpu_contention"] if env["device"] == "cuda" else [])
    protocol = {
        "experiment": "USM pinned width-bit versus adaptive width-bit QAT",
        "preserved_ablation": "scripts/run_static_vs_usm.py remains the FP32 width-only experiment",
        "candidate_space": [{"width_mult": width, "bit_width": bits} for width, bits in QAT_CANDIDATES],
        "candidate_count": 16,
        "primary_latency_quantile": "p95",
        "safety_margin": 0.90,
        "fallback": "minimum measured current-condition P95 plus measured transition setter overhead; fallback deadline violation is logged",
        "conditions": conditions,
        "condition_semantics": "controlled reproducible laboratory contention, not production workload",
        "backend": "PyTorch QAT fake quantization; no real integer-kernel speedup claim",
        "data": {
            "accuracy_selection": "seeded stratified CIFAR-10 train validation subset",
            "bn_calibration": "complementary CIFAR-10 train subset",
            "final_evaluation": "untouched CIFAR-10 test split",
        },
        "warmup": "same seeded randomized warm order over all 16 candidates for each policy; excluded from request metrics",
        "timing": "arrival-to-response includes queue, data retrieval, controller, configuration setters, forward, and top-1; test data loading is included",
        "independent_unit": "complete paired repetition",
        "repetitions": args.repetitions,
        "sample_count": args.n_samples,
        "seed": args.seed,
        "target_p95_ms_for_pinned_calibration": args.target_p95_ms,
        "trace": args.trace,
        "interarrival_ms": args.interarrival_ms,
    }
    (run_root / "experiment_metadata.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    partition_path = run_root / "accuracy" / "data_partition.json"
    partition = create_partition(
        args.dataset_root, partition_path, args.validation_per_class, args.seed
    )

    accuracy_json = run_root / "accuracy" / "candidate_accuracy.json"
    run_command([
        sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate_usm_qat_configs.py"),
        "--checkpoint", str(checkpoint), "--partition", str(partition_path),
        "--output", str(accuracy_json), "--dataset-root", str(args.dataset_root),
        "--batch-size", "64", "--bn-calibration-batches", str(args.bn_calibration_batches),
        "--seed", str(args.seed), "--device", args.device,
    ])

    profile_dir = run_root / "profiles"
    run_command([
        sys.executable, str(PROJECT_ROOT / "scripts" / "profile_usm_qat_configs.py"),
        "--checkpoint", str(checkpoint), "--partition", str(partition_path),
        "--output-dir", str(profile_dir), "--dataset-root", str(args.dataset_root),
        "--threads", str(args.threads), "--repetitions", str(args.profile_repetitions),
        "--holdout-repetitions", str(args.profile_holdout_repetitions),
        "--samples-per-repetition", str(args.profile_samples_per_repetition),
        "--warmup-samples", str(args.profile_warmup_samples),
        "--bn-calibration-batches", str(args.bn_calibration_batches), "--seed", str(args.seed),
        "--device", args.device,
    ])

    _, accuracies = __import__("scripts.experiment_usm_qat", fromlist=["read_accuracy"]).read_accuracy(accuracy_json)
    profile_rows = read_csv(profile_dir / "resource_condition_profiles.csv")
    profiles = validate_latency_table(profile_rows, conditions)
    pinned = select_pinned_configuration(accuracies, profiles, args.target_p95_ms)
    pinned_config_path = run_root / "config.json"
    pinned_config_document = {
        "width_mult": pinned["config"][0],
        "bit_width": pinned["config"][1],
        "accuracy_percent_on_train_validation": pinned["accuracy_percent"],
        "worst_condition_p95_ms": pinned["worst_condition_p95_ms"],
        "target_p95_ms": args.target_p95_ms,
        "target_feasible": pinned["target_feasible"],
        "fallback": pinned["fallback"],
        "selection_data_split": "CIFAR-10 train validation subset only",
        "checkpoint_sha256": env["checkpoint_sha256"],
    }
    pinned_config_path.write_text(json.dumps(pinned_config_document, indent=2) + "\n", encoding="utf-8")

    base_workload = run_root / "workload" / "workload_base.json"
    bound_workload = run_root / "workload" / "workload.json"
    run_command([
        sys.executable, str(PROJECT_ROOT / "scripts" / "generate_usm_qat_workload.py"),
        "--seed", str(args.seed), "--n-samples", str(args.n_samples),
        "--output", str(base_workload), "--interarrival-ms", str(args.interarrival_ms),
        "--trace", args.trace, "--conditions", ",".join(conditions),
    ])
    bind_test_workload(base_workload, args.dataset_root, args.seed, bound_workload)

    run_command([
        sys.executable, str(PROJECT_ROOT / "scripts" / "experiment_usm_qat.py"),
        "--run-id", run_id, "--run-root", str(run_root),
        "--checkpoint", str(checkpoint), "--partition", str(partition_path),
        "--accuracy-json", str(accuracy_json),
        "--profile-csv", str(profile_dir / "resource_condition_profiles.csv"),
        "--switch-costs", str(profile_dir / "switch_costs.csv"),
        "--pinned-config", str(pinned_config_path), "--workload", str(bound_workload),
        "--dataset-root", str(args.dataset_root), "--repetitions", str(args.repetitions),
        "--threads", str(args.threads), "--bn-calibration-batches", str(args.bn_calibration_batches),
        "--seed", str(args.seed),
        "--device", args.device,
    ])

    run_command([
        sys.executable, str(PROJECT_ROOT / "scripts" / "generate_usm_qat_report.py"),
        "--run-id", run_id, "--run-root", str(run_root),
    ])
    print(f"QAT experiment artifacts: {run_root}; pinned={tuple(pinned['config'])}")


if __name__ == "__main__":
    main()