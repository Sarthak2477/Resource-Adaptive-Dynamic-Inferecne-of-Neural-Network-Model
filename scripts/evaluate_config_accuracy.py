"""Measure full-test accuracy for every deployable model configuration."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import Model, get_dataloaders, recalibrate_bn
from models.config import FLAGS
from models.resnet import set_model_bit_width, set_model_width


EXPECTED_CIFAR10_TEST_SIZE = 10_000
WILSON_Z_95 = 1.959963984540054


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wilson_interval(successes, total, z=WILSON_Z_95):
    """Return the two-sided Wilson score interval for a binomial proportion."""
    if total <= 0:
        return None, None

    proportion = successes / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    radius = (
        z
        * ((proportion * (1.0 - proportion) / total)
           + z_squared / (4.0 * total * total)) ** 0.5
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_checkpoint(explicit_path):
    if explicit_path is not None:
        checkpoint = Path(explicit_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    candidates = (
        PROJECT_ROOT / "models" / "checkpoint" / "us_resnet_epoch100_checkpoint.pt.zip",
        PROJECT_ROOT / "models" / "checkpoint" / "us_resnet_epoch100_checkpoint.pt",
        PROJECT_ROOT / "models" / "checkpoints" / "best_model.pt",
        PROJECT_ROOT / "models" / "checkpoints" / "us_resnet_epoch100_checkpoint.pt",
        PROJECT_ROOT / "us_resnet_epoch100_checkpoint.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "No trained checkpoint was found. Pass --checkpoint PATH; refusing to "
        "evaluate randomly initialized weights."
    )


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint does not contain a state dictionary: {checkpoint_path}")

    cleaned_state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(cleaned_state_dict, strict=True)


def configure_determinism(seed, deterministic):
    seed_everything(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True)


def evaluate_configuration(model, test_loader, class_names, device, width_mult, bit_width):
    num_classes = len(class_names)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    total = 0
    correct = 0

    set_model_width(model, width_mult)
    set_model_bit_width(model, bit_width)
    model.eval()

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            predictions = model(images).argmax(dim=1)

            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
            encoded = (labels * num_classes + predictions).cpu()
            batch_confusion = torch.bincount(
                encoded, minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes)
            confusion += batch_confusion

    if total == 0:
        raise RuntimeError("The test split is empty; no accuracy was measured.")

    lower, upper = wilson_interval(correct, total)
    per_class = []
    for class_index, class_name in enumerate(class_names):
        class_total = int(confusion[class_index].sum().item())
        class_correct = int(confusion[class_index, class_index].item())
        class_lower, class_upper = wilson_interval(class_correct, class_total)
        per_class.append({
            "class_name": class_name,
            "total": class_total,
            "correct": class_correct,
            "accuracy_percent": 100.0 * class_correct / class_total if class_total else None,
            "wilson_95_percent": [
                100.0 * class_lower if class_lower is not None else None,
                100.0 * class_upper if class_upper is not None else None,
            ],
        })

    return {
        "width_mult": float(width_mult),
        "bit_width": int(bit_width),
        "total": total,
        "correct": correct,
        "accuracy_percent": 100.0 * correct / total,
        "wilson_95_percent": [100.0 * lower, 100.0 * upper],
        "per_class": per_class,
        "confusion_matrix": {
            "labels": class_names,
            "rows_actual_columns_predicted": confusion.tolist(),
        },
    }


def write_results(output_path, document):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_path.with_suffix(".csv")
    if output_path.exists() or csv_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output_path} or {csv_path}"
        )

    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(document, output_file, indent=2)
        output_file.write("\n")

    with csv_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        fieldnames = [
            "width_mult",
            "bit_width",
            "total",
            "correct",
            "accuracy_percent",
            "wilson_95_lower_percent",
            "wilson_95_upper_percent",
            "per_class_accuracy_percent_json",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in document["results"]:
            interval = result["wilson_95_percent"]
            writer.writerow({
                "width_mult": result["width_mult"],
                "bit_width": result["bit_width"],
                "total": result["total"],
                "correct": result["correct"],
                "accuracy_percent": result["accuracy_percent"],
                "wilson_95_lower_percent": interval[0],
                "wilson_95_upper_percent": interval[1],
                "per_class_accuracy_percent_json": json.dumps({
                    row["class_name"]: row["accuracy_percent"]
                    for row in result["per_class"]
                }, sort_keys=True),
            })
    return csv_path


def resolve_output_path(requested_path):
    if requested_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = PROJECT_ROOT / "results" / f"config_accuracy_{timestamp}.json"
    else:
        output_path = requested_path.expanduser().resolve()

    if output_path.suffix.lower() != ".json":
        raise ValueError("--output must use a .json extension; a sibling .csv will be written")
    csv_path = output_path.with_suffix(".csv")
    if output_path.exists() or csv_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output_path} or {csv_path}"
        )
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every deployable width/bit-width configuration on the "
            "complete CIFAR-10 test split."
        )
    )
    parser.add_argument("--checkpoint", type=Path, help="Trained model checkpoint path")
    parser.add_argument("--output", type=Path, help="Output JSON path; a sibling CSV is also written")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=FLAGS.batch_size)
    parser.add_argument("--bn-calibration-batches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--allow-nonstandard-test-size",
        action="store_true",
        help="Permit a local test split that is not exactly 10,000 images",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Request deterministic PyTorch algorithms; unsupported ops will fail explicitly",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.bn_calibration_batches < 1:
        raise ValueError("--bn-calibration-batches must be at least 1")

    os.chdir(PROJECT_ROOT)
    output_path = resolve_output_path(args.output)
    configure_determinism(args.seed, args.deterministic)
    checkpoint_path = find_checkpoint(args.checkpoint)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch reports CUDA unavailable.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )

    FLAGS.batch_size = args.batch_size
    train_loader, test_loader = get_dataloaders()
    test_dataset = test_loader.dataset
    class_names = list(test_dataset.classes)
    if len(class_names) != FLAGS.num_classes:
        raise ValueError(
            f"Expected {FLAGS.num_classes} classes, found {len(class_names)}: {class_names}"
        )
    class_sample_counts = np.bincount(
        np.asarray(test_dataset.targets), minlength=len(class_names)
    )
    if len(test_dataset) != EXPECTED_CIFAR10_TEST_SIZE and not args.allow_nonstandard_test_size:
        raise ValueError(
            f"Expected the complete CIFAR-10 test split ({EXPECTED_CIFAR10_TEST_SIZE} images), "
            f"found {len(test_dataset)}. Use --allow-nonstandard-test-size only for a deliberate "
            "nonstandard evaluation."
        )
    if (
        not args.allow_nonstandard_test_size
        and not np.all(class_sample_counts == EXPECTED_CIFAR10_TEST_SIZE // FLAGS.num_classes)
    ):
        raise ValueError(
            "The standard CIFAR-10 test split must contain 1,000 images per class; "
            f"found {dict(zip(class_names, class_sample_counts.tolist()))}."
        )

    model = Model(num_classes=FLAGS.num_classes, input_size=FLAGS.image_size).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    calibration_batches = min(args.bn_calibration_batches, len(train_loader))
    results = []
    for config_index, (width_mult, bit_width) in enumerate(FLAGS.deploy_configs):
        config_seed = args.seed + config_index
        configure_determinism(config_seed, args.deterministic)
        if getattr(train_loader, "generator", None) is not None:
            train_loader.generator.manual_seed(config_seed)
        recalibrate_bn(
            model,
            train_loader,
            width_mult,
            bit_width,
            device,
            num_batches=calibration_batches,
        )
        result = evaluate_configuration(
            model,
            test_loader,
            class_names,
            device,
            width_mult,
            bit_width,
        )
        result["bn_calibration_batches"] = calibration_batches
        results.append(result)
        print(
            f"width={width_mult:.2f} bits={bit_width:2d} "
            f"accuracy={result['accuracy_percent']:.3f}% "
            f"({result['correct']}/{result['total']})"
        )

    document = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "per_configuration_full_test_accuracy",
        "dataset": {
            "name": "CIFAR-10",
            "split": "test",
            "root": str((PROJECT_ROOT / "cifar10").resolve()),
            "sample_count": len(test_dataset),
            "expected_full_test_sample_count": EXPECTED_CIFAR10_TEST_SIZE,
            "class_sample_counts": dict(zip(class_names, class_sample_counts.tolist())),
            "class_names": class_names,
            "class_to_idx": test_dataset.class_to_idx,
        },
        "model": {
            "architecture": "models.Model",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "deploy_configs": [
                {"width_mult": float(width), "bit_width": int(bits)}
                for width, bits in FLAGS.deploy_configs
            ],
        },
        "protocol": {
            "seed": args.seed,
            "deterministic_algorithms_requested": args.deterministic,
            "device": str(device),
            "batch_size": args.batch_size,
            "bn_calibration_data_split": "train",
            "bn_calibration_batches_per_configuration": calibration_batches,
            "test_split_used_for_bn_calibration": False,
            "confidence_interval": "95% Wilson score interval",
        },
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cuda_version": torch.version.cuda,
            "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "results": results,
    }
    csv_path = write_results(output_path, document)
    print(f"JSON results: {output_path}")
    print(f"CSV summary:  {csv_path}")


if __name__ == "__main__":
    main()