import argparse
import os
import sys
import csv
import torch
import numpy as np
import time
import platform
import random
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "resource_control"))

from models import Model, set_model_width, set_model_bit_width, get_dataloaders, recalibrate_bn
from models.checkpoint_io import load_model_checkpoint, resolve_usm_checkpoint
from resource_control.telemetry import ResourceMonitor
from resource_control.stress import ResourceStress

def profile_fp32_widths(output_path, checkpoint_path, threads=1, warmup_samples=20, repeats=100, seed=12345):
    output_path = Path(output_path).expanduser().resolve()
    metadata_path = output_path.with_suffix(".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite profile artifacts: {output_path} or {metadata_path}")

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(num_classes=10, input_size=32).to(device)
    checkpoint_info = load_model_checkpoint(model, checkpoint_path, device)
    model.eval()
    
    monitor = ResourceMonitor()
    hw_fingerprint = monitor.get_hardware_fingerprint(device)
    
    train_loader, _ = get_dataloaders()
    widths = [0.25, 0.5, 0.75, 1.0]
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    
    results = []
    
    # Approx flops/params for US-ResNet50
    approx_flops_dict = {0.25: 1e8, 0.5: 4e8, 0.75: 9e8, 1.0: 16e8}
    approx_params_dict = {0.25: 1e6, 0.5: 4e6, 0.75: 9e6, 1.0: 25e6}

    for width_index, width in enumerate(widths):
        torch.manual_seed(seed + width_index)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + width_index)
        recalibrate_bn(model, train_loader, width, 32, device, num_batches=10)
    
    rng = random.Random(seed)
    for condition in ("baseline", "contention"):
        condition_widths = list(widths)
        rng.shuffle(condition_widths)
        with ResourceStress(device, cpu=True, gpu=device.type == "cuda") as stress:
            stress.set_enabled(condition == "contention")
            for w in condition_widths:
                set_model_width(model, w)
                set_model_bit_width(model, 32)
                for _ in range(warmup_samples):
                    with torch.inference_mode():
                        model(dummy_input)
                        if device.type == "cuda":
                            torch.cuda.synchronize()

                times = []
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    with torch.inference_mode():
                        model(dummy_input)
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                    times.append((time.perf_counter() - t0) * 1000.0)

                latency_ms = float(np.mean(times))
                p50_ms = float(np.percentile(times, 50))
                p95_ms = float(np.percentile(times, 95))
                p99_ms = float(np.percentile(times, 99))
                std_ms = float(np.std(times))
                results.append({
                    "width_mult": w,
                    "bit_width": 32,
                    "resource_condition": condition,
                    "latency_ms": latency_ms,
                    "approx_flops": approx_flops_dict.get(w, 0),
                    "approx_params": approx_params_dict.get(w, 0),
                    "device_speed_score": hw_fingerprint["device_speed_score"],
                    "cpu_cores": hw_fingerprint["cpu_cores"],
                    "ram_gb": hw_fingerprint["ram_gb"],
                    "has_cuda": hw_fingerprint["has_cuda"],
                    "p50_ms": p50_ms,
                    "p95_ms": p95_ms,
                    "p99_ms": p99_ms,
                    "std_ms": std_ms,
                    "sample_count": repeats,
                    "checkpoint_path": checkpoint_info["checkpoint_path"],
                    "checkpoint_sha256": checkpoint_info["checkpoint_sha256"],
                    "device": str(device),
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "thread_count": threads,
                    "warmup_samples": warmup_samples,
                    "input_kind": "seeded_synthetic_tensor",
                    "seed": seed,
                    "stress_profile": "torch CPU matrix multiply plus concurrent CUDA matrix multiply when CUDA is available" if condition == "contention" else "none",
                })
                print(f"{condition} width {w}: avg {latency_ms:.2f}ms, p95 {p95_ms:.2f}ms")

    with output_path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"Profile saved to {output_path}")
    metadata = {
        **checkpoint_info,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "thread_count": threads,
        "seed": seed,
        "platform": platform.platform(),
        "warmup_samples_per_width": warmup_samples,
        "measured_samples_per_width": repeats,
        "precision_bits": 32,
        "widths": widths,
        "resource_conditions": ["baseline", "contention"],
        "contention_profile": "in-process CPU and concurrent CUDA matrix-multiply stress; measured, not a deployment workload",
    }
    with metadata_path.open("x", encoding="utf-8") as metadata_file:
        import json
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile trained USM subnets in FP32 mode.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path(project_root) / "results" / "static_vs_usm_fp32" / "profiles" / "fp32_width_profile.csv")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup-samples", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    if min(args.threads, args.repeats) < 1 or args.warmup_samples < 0:
        parser.error("threads and repeats must be positive; warmup samples cannot be negative")
    checkpoint_path = resolve_usm_checkpoint(args.checkpoint)
    profile_fp32_widths(
        args.output, checkpoint_path, args.threads,
        args.warmup_samples, args.repeats, args.seed,
    )
