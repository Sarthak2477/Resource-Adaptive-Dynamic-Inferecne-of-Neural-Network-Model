import argparse
import copy
from contextlib import nullcontext
import csv
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.ao.quantization import QConfigMapping, get_default_qconfig
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import Model, recalibrate_bn, set_model_bit_width, set_model_width
from models.checkpoint_io import load_model_checkpoint, resolve_usm_checkpoint
from models.config import FLAGS
from models.ops import ActQuant, ResnetBatchNorm2d, ResnetConv2d, ResnetLinear, make_divisible
from resource_control.qat_experiment import WIDTHS
from scripts.evaluate_width_latency import select_stratified_indices
from scripts.usm_qat_data import make_test_imagefolder, make_train_imagefolders


PRECISIONS = ("fp32", "fp16", "int8", "int4")


class StaticNetwork(nn.Module):
    def __init__(self, features, classifier):
        super().__init__()
        self.features = features
        self.classifier = classifier

    def forward(self, images):
        features = self.features(images)
        return self.classifier(torch.flatten(features, start_dim=1))


def _freeze_module(module, width, bit_width):
    if isinstance(module, ResnetConv2d):
        in_channels = (
            module.max_in_channels if module.is_stem
            else make_divisible(module.max_in_channels * width, FLAGS.width_divisor)
        )
        out_channels = make_divisible(module.max_out_channels * width, FLAGS.width_divisor)
        groups = in_channels if module.groups_ != 1 else 1
        frozen = nn.Conv2d(
            in_channels, out_channels, module.kernel_size, module.stride,
            module.padding, module.dilation, groups, module.bias is not None,
            module.padding_mode,
        )
        with torch.no_grad():
            if module.groups_ == 1:
                frozen.weight.copy_(module.weight[:out_channels, :in_channels])
            else:
                frozen.weight.copy_(module.weight[:out_channels])
            if module.bias is not None:
                frozen.bias.copy_(module.bias[:out_channels])
        return frozen

    if isinstance(module, ResnetLinear):
        in_features = make_divisible(module.max_in_features * width, FLAGS.width_divisor)
        frozen = nn.Linear(in_features, module.max_out_features, module.bias is not None)
        with torch.no_grad():
            frozen.weight.copy_(module.weight[:, :in_features])
            if module.bias is not None:
                frozen.bias.copy_(module.bias)
        return frozen

    if isinstance(module, ResnetBatchNorm2d):
        channels = make_divisible(module.max_features * width, FLAGS.width_divisor)
        key = (width, bit_width)
        if key not in module.calibrated_running_mean:
            raise RuntimeError(f"Missing calibrated BN statistics for width={width}, bits={bit_width}")
        frozen = nn.BatchNorm2d(channels, eps=module.eps, affine=True, track_running_stats=True)
        with torch.no_grad():
            frozen.weight.copy_(module.weight[:channels])
            frozen.bias.copy_(module.bias[:channels])
            frozen.running_mean.copy_(module.calibrated_running_mean[key])
            frozen.running_var.copy_(module.calibrated_running_var[key])
        frozen.eval()
        return frozen

    if isinstance(module, ActQuant):
        return nn.Identity()

    for name, child in list(module.named_children()):
        module._modules[name] = _freeze_module(child, width, bit_width)
    return module


def freeze_usm_width(model, width, bit_width=32):
    frozen = copy.deepcopy(model).cpu().eval()
    features = _freeze_module(frozen.features, width, bit_width)
    classifier = _freeze_module(frozen.classifier, width, bit_width)
    return StaticNetwork(features, classifier).eval()


def available_precisions(device):
    supported = {"fp32": {"device": str(device), "backend": "PyTorch FP32"}}
    if device.type == "cuda":
        supported["fp16"] = {"device": str(device), "backend": "PyTorch CUDA FP16"}
    supported["int8"] = {
        "device": "cpu",
        "backend": f"PyTorch FX static quantization ({_int8_engine()})",
    }
    return supported


def _int8_engine():
    engines = torch.backends.quantized.supported_engines
    for engine in ("x86", "fbgemm", "onednn"):
        if engine in engines:
            return engine
    raise RuntimeError(f"No supported PyTorch CPU INT8 engine; found {engines}")


def quantize_static_int8(model, calibration_loader, calibration_batches):
    engine = _int8_engine()
    torch.backends.quantized.engine = engine
    qconfig = QConfigMapping().set_global(get_default_qconfig(engine))
    example_inputs = (torch.zeros(1, 3, 32, 32),)
    prepared = prepare_fx(model.cpu().eval(), qconfig, example_inputs)
    with torch.inference_mode():
        for batch_index, (images, _) in enumerate(calibration_loader):
            if batch_index >= calibration_batches:
                break
            prepared(images)
    converted = convert_fx(prepared).eval()
    calls = [
        converted.get_submodule(node.target)
        for node in converted.graph.nodes
        if node.op == "call_module"
    ]
    quantized_conv_count = sum(
        isinstance(module, torch.ao.nn.quantized.Conv2d) for module in calls
    )
    quantized_linear_count = sum(
        isinstance(module, torch.ao.nn.quantized.Linear) for module in calls
    )
    float_conv_count = sum(isinstance(module, nn.Conv2d) for module in calls)
    float_linear_count = sum(isinstance(module, nn.Linear) for module in calls)
    if not quantized_conv_count or not quantized_linear_count or float_conv_count or float_linear_count:
        raise RuntimeError(
            "INT8 conversion did not produce a fully quantized convolution/linear graph "
            f"(INT8 conv={quantized_conv_count}, INT8 linear={quantized_linear_count}, "
            f"FP32 conv={float_conv_count}, FP32 linear={float_linear_count})"
        )
    return converted


def _precision_context(device, precision):
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _measure_one(model, image, device, precision, warmup, repeats):
    sample = image.unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        for _ in range(warmup):
            with _precision_context(device, precision):
                model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings = []
        for _ in range(repeats):
            started = time.perf_counter()
            with _precision_context(device, precision):
                model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - started) * 1000.0)
    return timings


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("x", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    checkpoint, dataset_root, output_dir, sample_count=1000, latency_samples=100,
    latency_repeats=5, warmup=5, bn_batches=10, calibration_batches=10,
    batch_size=32, seed=12345, device_name="auto", threads=1,
):
    if min(sample_count, latency_samples, latency_repeats, bn_batches, calibration_batches, batch_size, threads) < 1 or warmup < 0:
        raise ValueError("Sample, repeat, calibration, and batch counts must be positive; warmup cannot be negative")
    if device_name not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    torch.set_num_threads(threads)

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_augmented, train_evaluation = make_train_imagefolders(dataset_root)
    test_dataset = make_test_imagefolder(dataset_root)
    class_count = len(test_dataset.classes)
    if sample_count > len(test_dataset):
        raise ValueError("sample-count exceeds the test split size")
    test_indices = select_stratified_indices(test_dataset.targets, class_count, sample_count, seed)
    latency_count = min(latency_samples, sample_count)
    latency_indices = test_indices[:latency_count]
    bn_indices = select_stratified_indices(train_augmented.targets, class_count, class_count * 20, seed + 2)
    bn_loader = DataLoader(Subset(train_evaluation, bn_indices), batch_size=32, shuffle=False, num_workers=0)
    calibration_loader = DataLoader(
        Subset(train_evaluation, bn_indices), batch_size=batch_size, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(Subset(test_dataset, test_indices), batch_size=1, shuffle=False, num_workers=0)

    source_model = Model(num_classes=class_count, input_size=32)
    checkpoint_info = load_model_checkpoint(source_model, checkpoint, torch.device("cpu"))
    source_model.eval()
    supported = available_precisions(device)
    summary_rows = []
    latency_rows = []
    evaluation_position = {dataset_index: position for position, dataset_index in enumerate(test_indices)}

    for width in WIDTHS:
        recalibrate_bn(source_model, bn_loader, width, 32, torch.device("cpu"), num_batches=bn_batches)
        static_fp32 = freeze_usm_width(source_model, width, 32)

        for precision in PRECISIONS:
            if precision == "int4":
                summary_rows.append({
                    "width_mult": width, "precision": precision, "status": "unsupported",
                    "backend": "none", "device": "none", "accuracy_percent": None,
                    "correct_count": None, "sample_count": sample_count,
                    "latency_samples": 0, "mean_latency_ms": None,
                    "p50_latency_ms": None, "p95_latency_ms": None,
                    "p99_latency_ms": None,
                    "reason": "No INT4 convolution kernel/backend is installed; no fake-quant result substituted.",
                })
                continue
            if precision not in supported:
                summary_rows.append({
                    "width_mult": width, "precision": precision, "status": "unsupported",
                    "backend": "none", "device": "none", "accuracy_percent": None,
                    "correct_count": None, "sample_count": sample_count,
                    "latency_samples": 0, "mean_latency_ms": None,
                    "p50_latency_ms": None, "p95_latency_ms": None,
                    "p99_latency_ms": None,
                    "reason": f"{precision.upper()} requires CUDA; selected device is {device}.",
                })
                continue

            if precision == "int8":
                network = quantize_static_int8(copy.deepcopy(static_fp32), calibration_loader, calibration_batches)
                run_device = torch.device("cpu")
            else:
                network = copy.deepcopy(static_fp32).to(device)
                run_device = device

            correct = 0
            total = 0
            with torch.inference_mode():
                for images, labels in test_loader:
                    with _precision_context(run_device, precision):
                        logits = network(images.to(device=run_device, dtype=torch.float32))
                    correct += int((logits.argmax(dim=1).cpu() == labels).sum().item())
                    total += int(labels.numel())

            measured = []
            if precision == "fp16":
                torch.cuda.synchronize(run_device)
            for dataset_index in latency_indices:
                image, _ = test_dataset[dataset_index]
                sample_position = evaluation_position[dataset_index]
                timings = _measure_one(network, image, run_device, precision, warmup, latency_repeats)
                measured.extend(timings)
                for repeat_index, latency_ms in enumerate(timings):
                    latency_rows.append({
                        "width_mult": width, "precision": precision,
                        "device": str(run_device), "backend": supported[precision]["backend"],
                        "sample_id": f"cifar10_test_index_{dataset_index:05d}",
                        "sample_position": sample_position, "repeat": repeat_index,
                        "latency_ms": latency_ms,
                    })

            summary_rows.append({
                "width_mult": width, "precision": precision, "status": "measured",
                "backend": supported[precision]["backend"], "device": str(run_device),
                "accuracy_percent": 100.0 * correct / total,
                "correct_count": correct, "sample_count": total,
                "latency_samples": latency_count,
                "mean_latency_ms": float(np.mean(measured)),
                "p50_latency_ms": float(np.percentile(measured, 50)),
                "p95_latency_ms": float(np.percentile(measured, 95)),
                "p99_latency_ms": float(np.percentile(measured, 99)),
                "reason": "",
            })
            print(
                f"width={width:.2f} precision={precision} accuracy={100.0 * correct / total:.2f}% "
                f"mean={np.mean(measured):.3f}ms p95={np.percentile(measured, 95):.3f}ms "
                f"device={run_device}"
            )

    summary_path = output_dir / "precision_width_summary.csv"
    latency_path = output_dir / "latency_observations.csv"
    metadata_path = output_dir / "run_metadata.json"
    _write_csv(summary_path, summary_rows)
    _write_csv(latency_path, latency_rows)
    metadata = {
        "experiment": "USM actual-precision width evaluation",
        "precision_semantics": {
            "fp32": "FP32 PyTorch operators",
            "fp16": "CUDA FP16 autocast kernels with FP32 master parameters and normalization",
            "int8": "PyTorch FX post-training static quantization with native CPU quantized operators",
            "int4": "not executed; unsupported backends are recorded as unsupported, never fake quantized",
        },
        "candidate_widths": list(WIDTHS),
        "precisions": list(PRECISIONS),
        "supported_precisions": supported,
        "device_selection": device_name,
        "model_precision_device": str(device),
        "int8_device": "cpu",
        "accuracy_dataset": "CIFAR-10 test split",
        "accuracy_sample_count": sample_count,
        "latency_sample_count": latency_count,
        "latency_repeats_per_sample": latency_repeats,
        "latency_warmups_per_sample": warmup,
        "latency_timing": "single-image model forward only; synchronized CUDA timing; excludes loading and transfers",
        "accuracy_inference_batch_size": 1,
        "bn_calibration": "CIFAR-10 train subset; recalibrated independently per width at FP32",
        "int8_calibration": "CIFAR-10 train evaluation subset, disjoint from test split",
        "seed": seed,
        "threads": threads,
        "os": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "checkpoint": checkpoint_info,
        "outputs": {"summary": summary_path.name, "latency_observations": latency_path.name},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return summary_rows


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark real FP32, CUDA FP16, and CPU INT8 USM inference across widths; report unsupported precisions explicitly."
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "cifar10")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--latency-samples", type=int, default=100)
    parser.add_argument("--latency-repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--bn-batches", type=int, default=10)
    parser.add_argument("--calibration-batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    checkpoint = resolve_usm_checkpoint(args.checkpoint)
    run_experiment(
        checkpoint, args.dataset_root, args.output, args.sample_count,
        args.latency_samples, args.latency_repeats, args.warmup,
        args.bn_batches, args.calibration_batches, args.batch_size,
        args.seed, args.device, args.threads,
    )


if __name__ == "__main__":
    main()