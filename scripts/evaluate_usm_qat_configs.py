import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from models import Model, recalibrate_bn, set_model_bit_width, set_model_width
from models.checkpoint_io import load_model_checkpoint, resolve_usm_checkpoint
from resource_control.qat_experiment import QAT_CANDIDATES
from scripts.usm_qat_data import make_train_imagefolders


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_all(checkpoint, partition_path, output_json, dataset_root, batch_size, bn_batches, seed, device_name="auto"):
    partition_path = Path(partition_path).resolve()
    output_json = Path(output_json).resolve()
    output_csv = output_json.with_suffix(".csv")
    if output_json.exists() or output_csv.exists():
        raise FileExistsError(f"Refusing to overwrite accuracy outputs: {output_json}")
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    validation_indices = [int(index) for index in partition["validation_indices"]]
    bn_indices = [int(index) for index in partition["bn_calibration_indices"]]
    if set(validation_indices) & set(bn_indices):
        raise ValueError("Validation and BN calibration indices overlap")

    train_augmented, train_evaluation = make_train_imagefolders(dataset_root)
    if train_augmented.class_to_idx != train_evaluation.class_to_idx:
        raise ValueError("Train dataset class mappings differ between transforms")
    if max(validation_indices + bn_indices, default=-1) >= len(train_augmented):
        raise ValueError("Partition index is outside the training dataset")
    validation_loader = DataLoader(
        Subset(train_evaluation, validation_indices), batch_size=batch_size,
        shuffle=False, num_workers=0,
    )
    bn_loader = DataLoader(
        Subset(train_augmented, bn_indices), batch_size=32,
        shuffle=False, num_workers=0,
    )

    if device_name not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    selected_device = "cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu" if device_name == "auto" else device_name
    device = torch.device(selected_device)
    torch.set_num_threads(1)
    model = Model(num_classes=len(train_augmented.classes), input_size=32).to(device)
    provenance = load_model_checkpoint(model, checkpoint, device)
    rows = []
    for candidate_index, (width, bits) in enumerate(QAT_CANDIDATES):
        seed_everything(seed + candidate_index)
        recalibrate_bn(model, bn_loader, width, bits, device, num_batches=bn_batches)
        set_model_width(model, width)
        set_model_bit_width(model, bits)
        correct = 0
        total = 0
        model.eval()
        with torch.inference_mode():
            for images, labels in validation_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                correct += int((model(images).argmax(dim=1) == labels).sum().item())
                total += int(labels.numel())
        if total == 0:
            raise RuntimeError("No validation examples were evaluated")
        rows.append({
            "width_mult": width,
            "bit_width": bits,
            "correct": correct,
            "total": total,
            "accuracy_percent": 100.0 * correct / total,
            "bn_calibration_batches": min(bn_batches, len(bn_loader)),
            "bn_calibration_batch_size": 32,
        })
        print(f"width={width:.2f} bits={bits:2d} accuracy={100.0 * correct / total:.3f}% ({correct}/{total})")

    checkpoint_hash = provenance["checkpoint_sha256"]
    document = {
        "schema_version": 1,
        "experiment": "usm_qat_width_bit_selection_accuracy",
        "checkpoint": provenance,
        "candidate_count": len(QAT_CANDIDATES),
        "candidate_space": [{"width_mult": w, "bit_width": b} for w, b in QAT_CANDIDATES],
        "data_protocol": {
            "source_split": "CIFAR-10 train",
            "accuracy_selection_split": "seeded stratified train validation partition",
            "validation_sample_count": len(validation_indices),
            "validation_indices_sha256": hashlib.sha256(",".join(map(str, validation_indices)).encode()).hexdigest(),
            "bn_calibration_split": "complementary train partition",
            "bn_calibration_sample_count": len(bn_indices),
            "bn_indices_sha256": hashlib.sha256(",".join(map(str, bn_indices)).encode()).hexdigest(),
            "final_evaluation_split": "CIFAR-10 test; not read by this script",
            "seed": seed,
        },
        "inference_backend": "PyTorch QAT fake quantization; not real INT4/INT8 integer kernels",
        "results": rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    with output_csv.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate all 16 QAT USM width-bit candidates on train validation data.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "cifar10")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bn-calibration-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if min(args.batch_size, args.bn_calibration_batches) < 1:
        parser.error("batch size and BN calibration batches must be positive")
    checkpoint = resolve_usm_checkpoint(args.checkpoint)
    evaluate_all(
        checkpoint, args.partition, args.output, args.dataset_root,
        args.batch_size, args.bn_calibration_batches, args.seed, args.device,
    )


if __name__ == "__main__":
    main()