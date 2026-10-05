import argparse
import csv
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import Model, recalibrate_bn, set_model_bit_width, set_model_width
from models.checkpoint_io import load_model_checkpoint, resolve_usm_checkpoint
from resource_control.qat_experiment import QAT_CANDIDATES, candidate_configurations, prediction_validation
from resource_control.stress import ResourceStress
from resource_control.telemetry import hardware_fingerprint
from scripts.usm_qat_data import make_train_imagefolders


def set_config(model, config):
    set_model_width(model, float(config[0]))
    set_model_bit_width(model, int(config[1]))


def query_gpu_metrics():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        values = [value.strip() for value in result.stdout.splitlines()[0].split(",")]
        return {
            "gpu_utilization_percent": float(values[0]),
            "gpu_clock_mhz": float(values[1]),
            "gpu_memory_used_mb": float(values[2]),
            "gpu_memory_total_mb": float(values[3]),
            "gpu_temperature_c": float(values[4]),
            "gpu_power_w": float(values[5]),
        }
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return {
            "gpu_utilization_percent": None,
            "gpu_clock_mhz": None,
            "gpu_memory_used_mb": None,
            "gpu_memory_total_mb": None,
            "gpu_temperature_c": None,
            "gpu_power_w": None,
        }


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite profile artifact: {path}")
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def profile_lightweight_configs(
    checkpoint, partition_path, output_dir, dataset_root, threads=1,
    warmup_samples=10, measured_samples=20, bn_batches=10, seed=12345,
    device_name="auto",
):
    if warmup_samples != 10 or measured_samples != 20:
        raise ValueError("The lightweight protocol requires exactly 10 warmups and 20 measurements")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "profiles": output_dir / "resource_condition_profiles.csv",
        "observations": output_dir / "profile_observations.csv",
        "switch_costs": output_dir / "switch_costs.csv",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Refusing to overwrite lightweight QAT profile artifacts")

    torch.set_num_threads(threads)
    if device_name not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    selected_device = "cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu" if device_name == "auto" else device_name
    device = torch.device(selected_device)
    train_augmented, _ = make_train_imagefolders(dataset_root)
    partition = json.loads(Path(partition_path).read_text(encoding="utf-8"))
    bn_indices = [int(index) for index in partition["bn_calibration_indices"]]
    bn_loader = DataLoader(
        Subset(train_augmented, bn_indices), batch_size=32, shuffle=False,
        num_workers=0,
    )
    model = Model(num_classes=len(train_augmented.classes), input_size=32).to(device)
    provenance = load_model_checkpoint(model, checkpoint, device)
    model.eval()
    fingerprint = hardware_fingerprint(device)
    dummy_input = torch.zeros(1, 3, 32, 32, device=device)
    configs = candidate_configurations()
    for index, config in enumerate(configs):
        torch.manual_seed(seed + index)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + index)
        recalibrate_bn(model, bn_loader, config[0], config[1], device, num_batches=bn_batches)

    observations = []
    profile_rows = []
    conditions = ("baseline", "cpu_contention")
    process = psutil.Process()
    for condition in conditions:
        stress = ResourceStress(
            device, cpu=True, gpu=False, matrix_size=128, idle_s=0.02,
        )
        stress.start()
        stress.set_enabled(condition == "cpu_contention")
        try:
            for config in configs:
                set_config(model, config)
                with torch.inference_mode():
                    for _ in range(warmup_samples):
                        model(dummy_input)
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                    latencies = []
                    for sample_index in range(measured_samples):
                        started = time.perf_counter()
                        model(dummy_input)
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        latency_ms = (time.perf_counter() - started) * 1000.0
                        latencies.append(latency_ms)
                        observations.append({
                            "resource_condition": condition,
                            "width_mult": config[0],
                            "bit_width": config[1],
                            "sample_index": sample_index,
                            "latency_ms": latency_ms,
                            "cpu_utilization_percent": psutil.cpu_percent(interval=None),
                            "available_memory_mb": psutil.virtual_memory().available / (1024.0 ** 2),
                            "process_rss_mb": process.memory_info().rss / (1024.0 ** 2),
                        })
                profile_rows.append({
                    "resource_condition": condition,
                    "width_mult": config[0],
                    "bit_width": config[1],
                    "mean_ms": float(np.mean(latencies)),
                    "p50_ms": float(np.percentile(latencies, 50)),
                    "p95_ms": float(np.percentile(latencies, 95)),
                    "p99_ms": float(np.percentile(latencies, 99)),
                    "std_ms": float(np.std(latencies, ddof=1)),
                    "sample_count": len(latencies),
                    "warmup_measurements": warmup_samples,
                    "checkpoint_sha256": provenance["checkpoint_sha256"],
                    "device": str(device),
                    "device_speed_score": fingerprint["device_speed_score"],
                    "cpu_cores": fingerprint["cpu_cores"],
                    "ram_gb": fingerprint["ram_gb"],
                    "has_cuda": fingerprint["has_cuda"],
                    "thread_count": threads,
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "inference_backend": "PyTorch QAT fake quantization; no integer kernels",
                })
                print(f"{condition}: profiled {config} ({warmup_samples} warmup, {len(latencies)} measured)")
        finally:
            stress.close()

    write_csv(paths["profiles"], profile_rows)
    write_csv(paths["observations"], observations)
    write_csv(paths["switch_costs"], [
        {"switch_type": kind, "sample_count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
        for kind in ("width_only", "bit_only", "width_and_bit")
    ])
    (output_dir / "profile_metadata.json").write_text(json.dumps({
        "candidate_count": len(configs),
        "candidate_space": [{"width_mult": w, "bit_width": b} for w, b in configs],
        "resource_conditions": list(conditions),
        "condition_protocol": "CPU-only bounded matrix workload; 20 ms idle per 128x128 operation; GPU stress disabled",
        "measurements_per_candidate_condition": measured_samples,
        "warmup_measurements_per_candidate_condition": warmup_samples,
        "p95_source": "20 direct measured forwards per candidate and resource condition",
        "prediction_validation": "not estimated; no held-out profile repetitions in the lightweight protocol",
        "inference_backend": "PyTorch QAT fake quantization; not real INT4/INT8 execution",
        "checkpoint": provenance,
    }, indent=2) + "\n", encoding="utf-8")
    return profile_rows, {"status": "not_estimated_for_lightweight_profile"}


def profile_configs(
    checkpoint, partition_path, output_dir, dataset_root, threads=1,
    repetitions=6, holdout_repetitions=2, samples_per_repetition=20,
    warmup_samples=5, bn_batches=10, seed=12345, lightweight=False,
    device_name="auto",
):
    if lightweight:
        return profile_lightweight_configs(
            checkpoint, partition_path, output_dir, dataset_root, threads,
            warmup_samples, samples_per_repetition, bn_batches, seed, device_name,
        )
    if repetitions < 3 or not 1 <= holdout_repetitions < repetitions:
        raise ValueError("Need at least 3 repetitions and at least one held-out repetition")
    if samples_per_repetition < 2 or warmup_samples < 0 or bn_batches < 1:
        raise ValueError("Invalid sample, warmup, or BN calibration count")
    configs = candidate_configurations()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "profiles": output_dir / "resource_condition_profiles.csv",
        "observations": output_dir / "profile_observations.csv",
        "validation": output_dir / "prediction_validation.json",
        "validation_rows": output_dir / "prediction_validation.csv",
        "switch_costs": output_dir / "switch_costs.csv",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Refusing to overwrite one or more QAT profile artifacts")

    torch.set_num_threads(threads)
    if device_name not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    selected_device = "cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu" if device_name == "auto" else device_name
    device = torch.device(selected_device)
    train_augmented, _ = make_train_imagefolders(dataset_root)
    partition = json.loads(Path(partition_path).read_text(encoding="utf-8"))
    bn_indices = [int(index) for index in partition["bn_calibration_indices"]]
    bn_loader = DataLoader(
        Subset(train_augmented, bn_indices), batch_size=32, shuffle=False,
        num_workers=0,
    )
    model = Model(num_classes=len(train_augmented.classes), input_size=32).to(device)
    provenance = load_model_checkpoint(model, checkpoint, device)
    model.eval()
    fingerprint = hardware_fingerprint(device)
    dummy_input = torch.randn(1, 3, 32, 32, device=device)

    for index, config in enumerate(configs):
        torch.manual_seed(seed + index)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + index)
        recalibrate_bn(model, bn_loader, config[0], config[1], device, num_batches=bn_batches)

    conditions = ["baseline", "cpu_contention"]
    if device.type == "cuda":
        conditions.append("gpu_contention")
    observations = []
    rng = random.Random(seed)
    calibration_reps = repetitions - holdout_repetitions

    for condition_index, condition in enumerate(conditions):
        stress_mode = {
            "baseline": (False, False),
            "cpu_contention": (True, False),
            "gpu_contention": (False, True),
        }[condition]
        for repetition in range(repetitions):
            order = list(configs)
            rng.shuffle(order)
            stress = ResourceStress(
                device, cpu=stress_mode[0], gpu=stress_mode[1], matrix_size=512
            )
            with stress:
                if condition != "baseline":
                    stress.set_enabled(True)
                for config in order:
                    set_config(model, config)
                    for _ in range(warmup_samples):
                        with torch.inference_mode():
                            model(dummy_input)
                            if device.type == "cuda":
                                torch.cuda.synchronize(device)
                    gpu_before = query_gpu_metrics()
                    for sample_index in range(samples_per_repetition):
                        cpu_before = psutil.cpu_percent(interval=None)
                        mem_before = psutil.virtual_memory().available / (1024.0 ** 2)
                        started = time.perf_counter()
                        with torch.inference_mode():
                            model(dummy_input)
                            if device.type == "cuda":
                                torch.cuda.synchronize(device)
                        latency_ms = (time.perf_counter() - started) * 1000.0
                        observations.append({
                            "resource_condition": condition,
                            "width_mult": config[0],
                            "bit_width": config[1],
                            "repetition": repetition,
                            "profile_split": "calibration" if repetition < calibration_reps else "heldout",
                            "sample_index": sample_index,
                            "latency_ms": latency_ms,
                            "cpu_utilization_percent": cpu_before,
                            "available_memory_mb": mem_before,
                            **gpu_before,
                        })
                    print(f"{condition} rep {repetition + 1}/{repetitions}: {config}")

    profile_rows = []
    heldout_rows = []
    calibration_p95 = {}
    for condition in conditions:
        for config in configs:
            selected = [row for row in observations if row["resource_condition"] == condition
                        and (float(row["width_mult"]), int(row["bit_width"])) == config]
            calibration = [row["latency_ms"] for row in selected if row["profile_split"] == "calibration"]
            calibration_observations = [row for row in selected if row["profile_split"] == "calibration"]
            heldout_grouped = {}
            for row in selected:
                if row["profile_split"] == "heldout":
                    heldout_grouped.setdefault(row["repetition"], []).append(row["latency_ms"])
            calibration_p95_value = float(np.percentile(calibration, 95))
            calibration_p95[(condition, config)] = calibration_p95_value
            profile_rows.append({
                "resource_condition": condition,
                "width_mult": config[0],
                "bit_width": config[1],
                "mean_ms": float(np.mean(calibration)),
                "p50_ms": float(np.percentile(calibration, 50)),
                "p95_ms": calibration_p95_value,
                "p99_ms": float(np.percentile(calibration, 99)),
                "std_ms": float(np.std(calibration, ddof=1)) if len(calibration) > 1 else 0.0,
                "sample_count": len(calibration),
                "calibration_repetitions": calibration_reps,
                "heldout_repetitions": holdout_repetitions,
                "mean_cpu_utilization_percent": float(np.mean([row["cpu_utilization_percent"] for row in calibration_observations])),
                "minimum_available_memory_mb": float(min(row["available_memory_mb"] for row in calibration_observations)),
                "mean_gpu_utilization_percent": float(np.mean([row["gpu_utilization_percent"] for row in calibration_observations if row["gpu_utilization_percent"] is not None])) if any(row["gpu_utilization_percent"] is not None for row in calibration_observations) else None,
                "mean_gpu_clock_mhz": float(np.mean([row["gpu_clock_mhz"] for row in calibration_observations if row["gpu_clock_mhz"] is not None])) if any(row["gpu_clock_mhz"] is not None for row in calibration_observations) else None,
                "maximum_gpu_memory_used_mb": float(max([row["gpu_memory_used_mb"] for row in calibration_observations if row["gpu_memory_used_mb"] is not None], default=0.0)) or None,
                "maximum_gpu_temperature_c": float(max([row["gpu_temperature_c"] for row in calibration_observations if row["gpu_temperature_c"] is not None], default=0.0)) or None,
                "mean_gpu_power_w": float(np.mean([row["gpu_power_w"] for row in calibration_observations if row["gpu_power_w"] is not None])) if any(row["gpu_power_w"] is not None for row in calibration_observations) else None,
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "device": str(device),
                "device_speed_score": fingerprint["device_speed_score"],
                "cpu_cores": fingerprint["cpu_cores"],
                "ram_gb": fingerprint["ram_gb"],
                "has_cuda": fingerprint["has_cuda"],
                "thread_count": threads,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "inference_backend": "PyTorch QAT fake quantization; no integer kernels",
            })
            for repetition, values in heldout_grouped.items():
                heldout_rows.append({
                    "resource_condition": condition,
                    "width_mult": config[0],
                    "bit_width": config[1],
                    "heldout_repetition": repetition,
                    "p95_ms": float(np.percentile(values, 95)),
                })

    validation = prediction_validation(calibration_p95, heldout_rows)
    write_csv(paths["profiles"], profile_rows)
    write_csv(paths["observations"], observations)
    write_csv(paths["validation_rows"], validation["per_candidate_condition"])
    paths["validation"].write_text(json.dumps({
        **{key: value for key, value in validation.items() if key != "per_candidate_condition"},
        "split": "held-out complete profiling repetitions",
        "condition_count": len(conditions),
        "candidate_count": len(configs),
    }, indent=2) + "\n", encoding="utf-8")

    transitions = {"width_only": [], "bit_only": [], "width_and_bit": []}
    targets_by_kind = {
        "width_only": [(a, b) for a in configs for b in configs if a[0] != b[0] and a[1] == b[1]],
        "bit_only": [(a, b) for a in configs for b in configs if a[0] == b[0] and a[1] != b[1]],
        "width_and_bit": [(a, b) for a in configs for b in configs if a[0] != b[0] and a[1] != b[1]],
    }
    for kind, pairs in targets_by_kind.items():
        sample = pairs[:min(64, len(pairs))]
        for source, target in sample:
            samples = []
            for _ in range(10):
                set_config(model, source)
                start = time.perf_counter()
                set_config(model, target)
                samples.append((time.perf_counter() - start) * 1000.0)
            transitions[kind].extend(samples)
    switch_rows = [{
        "switch_type": kind,
        "sample_count": len(values),
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
    } for kind, values in transitions.items()]
    write_csv(paths["switch_costs"], switch_rows)

    metadata = {
        "candidate_count": len(configs),
        "candidate_space": [{"width_mult": w, "bit_width": b} for w, b in configs],
        "resource_conditions": conditions,
        "memory_pressure": "not injected; no portable bounded memory-pressure harness configured",
        "condition_protocol": "controlled reproducible laboratory contention; CPU and GPU stress are isolated conditions",
        "prediction_validation": str(paths["validation"].name),
        "inference_backend": "PyTorch QAT fake quantization; not real INT4/INT8 execution",
        "checkpoint": provenance,
    }
    (output_dir / "profile_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return profile_rows, validation


def main():
    parser = argparse.ArgumentParser(description="Profile all 16 USM QAT width-bit candidates under controlled conditions.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "cifar10")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=6)
    parser.add_argument("--holdout-repetitions", type=int, default=2)
    parser.add_argument("--samples-per-repetition", type=int, default=20)
    parser.add_argument("--warmup-samples", type=int, default=5)
    parser.add_argument("--bn-calibration-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--lightweight", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    checkpoint = resolve_usm_checkpoint(args.checkpoint)
    profile_configs(
        checkpoint, args.partition, args.output_dir, args.dataset_root,
        args.threads, args.repetitions, args.holdout_repetitions,
        args.samples_per_repetition, args.warmup_samples,
        args.bn_calibration_batches, args.seed, args.lightweight, args.device,
    )


if __name__ == "__main__":
    main()