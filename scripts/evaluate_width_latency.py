import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import torch
import torchvision
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import Model, recalibrate_bn, set_model_bit_width, set_model_width
from models.checkpoint_io import load_model_checkpoint, resolve_usm_checkpoint
from resource_control.qat_experiment import WIDTHS
from scripts.usm_qat_data import (
    make_test_imagefolder,
    make_train_imagefolders,
    stratified_train_partition,
)


def select_stratified_indices(targets, num_classes, sample_count, seed):
    if sample_count < 1 or sample_count % num_classes:
        raise ValueError("sample_count must be a positive multiple of num_classes")
    samples_per_class = sample_count // num_classes
    by_class = {class_index: [] for class_index in range(num_classes)}
    for index, target in enumerate(targets):
        target = int(target)
        if target not in by_class:
            raise ValueError(f"Unexpected class index {target}")
        by_class[target].append(index)
    if any(len(indices) < samples_per_class for indices in by_class.values()):
        raise ValueError("Not enough test samples in one or more classes")

    rng = random.Random(seed)
    selected_by_class = {}
    for class_index, indices in by_class.items():
        selected_by_class[class_index] = sorted(rng.sample(indices, samples_per_class))
    return [
        selected_by_class[class_index][offset]
        for offset in range(samples_per_class)
        for class_index in range(num_classes)
    ]


def summarize_width(records):
    latencies = np.asarray([row["model_latency_ms"] for row in records], dtype=float)
    correct = sum(bool(row["correct"]) for row in records)
    return {
        "width_mult": float(records[0]["width_mult"]),
        "bit_width": int(records[0]["bit_width"]),
        "sample_count": len(records),
        "correct_count": correct,
        "accuracy_percent": 100.0 * correct / len(records),
        "mean_latency_ms": float(np.mean(latencies)),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "std_latency_ms": float(np.std(latencies, ddof=1)) if len(latencies) > 1 else 0.0,
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
    }


def evaluate_widths(
    checkpoint, dataset_root, output_dir, sample_count=100, threads=1,
    warmup_samples=10, bn_calibration_batches=10, bit_width=32, seed=12345,
    device_name="auto",
):
    if sample_count < 1 or threads < 1 or warmup_samples < 0 or bn_calibration_batches < 1:
        raise ValueError("samples, threads, and BN batches must be positive; warmups cannot be negative")
    if bit_width not in (4, 8, 16, 32):
        raise ValueError("bit_width must be one of 4, 8, 16, or 32")
    if device_name not in ("auto", "cpu", "cuda"):
        raise ValueError("device_name must be auto, cpu, or cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    selected_device = (
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_path = output_dir / "request_latencies.csv"
    summary_path = output_dir / "width_summary.csv"
    metadata_path = output_dir / "run_metadata.json"

    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(selected_device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    train_augmented, _ = make_train_imagefolders(dataset_root)
    test_dataset = make_test_imagefolder(dataset_root)
    num_classes = len(test_dataset.classes)
    if sample_count > len(test_dataset):
        raise ValueError("sample_count exceeds the CIFAR-10 test split size")
    test_indices = select_stratified_indices(
        test_dataset.targets, num_classes, sample_count, seed,
    )
    if train_augmented.class_to_idx != test_dataset.class_to_idx:
        raise ValueError("Train and test class mappings differ")

    _, bn_indices = stratified_train_partition(
        train_augmented.targets, len(train_augmented.classes), 10, seed,
    )
    random.Random(seed).shuffle(bn_indices)
    bn_subset = Subset(train_augmented, bn_indices[:bn_calibration_batches * 32])
    bn_loader = DataLoader(bn_subset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(
        Subset(test_dataset, test_indices), batch_size=1, shuffle=False, num_workers=0,
    )

    model = Model(num_classes=num_classes, input_size=32).to(device)
    checkpoint_info = load_model_checkpoint(model, checkpoint, device)
    model.eval()
    width_order = list(WIDTHS)
    random.Random(seed).shuffle(width_order)
    profile_rows = []
    all_records = []
    process = psutil.Process()

    for width_index, width in enumerate(width_order):
        random.seed(seed + width_index)
        np.random.seed(seed + width_index)
        torch.manual_seed(seed + width_index)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + width_index)
        recalibrate_bn(
            model, bn_loader, width, bit_width, device,
            num_batches=min(bn_calibration_batches, len(bn_loader)),
        )
        set_model_width(model, width)
        set_model_bit_width(model, bit_width)
        model.eval()

        first_image, _ = test_dataset[test_indices[0]]
        warm_input = first_image.unsqueeze(0).to(device)
        with torch.inference_mode():
            for _ in range(warmup_samples):
                model(warm_input)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

        records = []
        for sample_position, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            with torch.inference_mode():
                logits = model(images)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            prediction = int(logits.argmax(dim=1).item())
            row = {
                "width_mult": float(width),
                "bit_width": int(bit_width),
                "request_id": f"width_{width:g}_sample_{sample_position:04d}",
                "sample_id": f"cifar10_test_index_{test_indices[sample_position]:05d}",
                "dataset_index": int(test_indices[sample_position]),
                "true_label": int(labels.item()),
                "predicted_label": prediction,
                "correct": prediction == int(labels.item()),
                "model_latency_ms": elapsed_ms,
                "process_rss_mb": process.memory_info().rss / (1024.0 ** 2),
                "gpu_peak_allocated_mb": (
                    torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
                    if device.type == "cuda" else None
                ),
            }
            records.append(row)
            all_records.append(row)

        profile_rows.append(summarize_width(records))
        print(
            f"width={width:.2f} bits={bit_width} accuracy="
            f"{profile_rows[-1]['accuracy_percent']:.2f}% "
            f"mean={profile_rows[-1]['mean_latency_ms']:.3f} ms "
            f"P95={profile_rows[-1]['p95_latency_ms']:.3f} ms"
        )

    with raw_path.open("x", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(all_records[0]))
        writer.writeheader()
        writer.writerows(all_records)
    with summary_path.open("x", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(profile_rows[0]))
        writer.writeheader()
        writer.writerows(sorted(profile_rows, key=lambda row: row["width_mult"]))

    metadata = {
        "experiment": "USM per-width held-out test latency evaluation",
        "dataset_split": "CIFAR-10 test",
        "sample_count": sample_count,
        "samples_per_class": sample_count // num_classes,
        "dataset_indices": test_indices,
        "dataset_indices_sha256": __import__("hashlib").sha256(
            ",".join(map(str, test_indices)).encode("utf-8")
        ).hexdigest(),
        "widths": list(WIDTHS),
        "bit_width": bit_width,
        "warmup_forward_count_per_width": warmup_samples,
        "warmup_sample": "first selected test image; excluded from measured rows",
        "timing_boundary": "single-sample model forward; CUDA synchronized before timer stop when using CUDA; data loading and device transfer excluded",
        "width_order": width_order,
        "threads": threads,
        "batch_size": 1,
        "seed": seed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "os": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "checkpoint": checkpoint_info,
        "bn_calibration": {
            "source_split": "CIFAR-10 train",
            "sample_count": len(bn_subset),
            "batches_per_width": min(bn_calibration_batches, len(bn_loader)),
            "disjoint_from_test": True,
        },
        "inference_backend": "PyTorch fake quantization; sub-32-bit modes are not integer-kernel execution",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return profile_rows


def main():
    parser = argparse.ArgumentParser(
        description="Measure accuracy and per-request latency for each USM width on one shared CIFAR-10 test subset."
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "cifar10")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup-samples", type=int, default=10)
    parser.add_argument("--bn-calibration-batches", type=int, default=10)
    parser.add_argument("--bit-width", type=int, choices=(4, 8, 16, 32), default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    run_id = args.run_id or datetime.now().strftime("width_latency_%Y%m%d_%H%M%S")
    if not all(character.isalnum() or character in "-_" for character in run_id):
        parser.error("run ID may contain only letters, digits, underscores, and hyphens")
    output_dir = args.output_dir or PROJECT_ROOT / "results" / "width_latency" / run_id
    checkpoint = resolve_usm_checkpoint(args.checkpoint)
    evaluate_widths(
        checkpoint, args.dataset_root, output_dir,
        sample_count=args.n_samples,
        threads=args.threads,
        warmup_samples=args.warmup_samples,
        bn_calibration_batches=args.bn_calibration_batches,
        bit_width=args.bit_width,
        seed=args.seed,
        device_name=args.device,
    )
    print(f"Evaluation artifacts: {Path(output_dir).resolve()}")


if __name__ == "__main__":
    main()
