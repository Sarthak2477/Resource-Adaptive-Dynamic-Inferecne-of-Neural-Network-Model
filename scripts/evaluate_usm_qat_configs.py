import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import Model, recalibrate_bn, set_model_bit_width, set_model_width
from models.checkpoint_io import load_model_checkpoint, resolve_usm_checkpoint
from resource_control.qat_experiment import QAT_CANDIDATES, validate_accuracy_table
from scripts.usm_qat_data import (
    make_test_imagefolder,
    make_train_imagefolders,
    stratified_train_partition,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_stratified_indices(targets, num_classes, sample_count, seed):
    """
    EXACT sampling procedure used by evaluate_width_latency.py.

    - Equal number of examples from every class.
    - random.Random(seed)
    - sampled indices sorted within each class.
    - final output interleaves classes by sample position.
    """
    if sample_count < 1 or sample_count % num_classes:
        raise ValueError(
            "sample_count must be a positive multiple of num_classes"
        )

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
        selected_by_class[class_index] = sorted(
            rng.sample(indices, samples_per_class)
        )

    return [
        selected_by_class[class_index][offset]
        for offset in range(samples_per_class)
        for class_index in range(num_classes)
    ]


def evaluate_all(
    checkpoint,
    partition_path,
    output,
    dataset_root,
    batch_size=1,
    bn_calibration_batches=10,
    seed=12345,
    device_name="auto",
    sample_count=500,
):
    """
    Evaluate every USM width/bit configuration using the SAME evaluation
    pipeline as evaluate_width_latency.py.

    Important:
      * Candidate accuracy is evaluated on CIFAR-10 TEST.
      * Test sampling is seeded and stratified exactly like
        evaluate_width_latency.py.
      * Test DataLoader is batch_size=1, shuffle=False, num_workers=0.
      * BN calibration remains on CIFAR-10 TRAIN.
      * BN calibration indices are selected exactly like
        evaluate_width_latency.py.
      * batch_size is retained as a CLI/API compatibility argument, but the
        evaluation batch size is intentionally forced to 1.
    """
    checkpoint = Path(checkpoint).expanduser().resolve()
    partition_path = Path(partition_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve()

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if not partition_path.exists():
        raise FileNotFoundError(f"Partition file not found: {partition_path}")

    if sample_count < 1:
        raise ValueError("sample_count must be positive")

    if bn_calibration_batches < 1:
        raise ValueError("bn_calibration_batches must be positive")

    if device_name not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    selected_device = (
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_name == "auto"
        else device_name
    )

    device = torch.device(selected_device)

    # The old evaluator uses batch_size=1. Keep the argument for compatibility
    # with run_static_vs_usm.py, but do not allow it to change the pipeline.
    evaluation_batch_size = 1

    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # 1. DATASETS
    #
    # EXACTLY like evaluate_width_latency.py:
    #   train_augmented -> BN calibration
    #   test_dataset    -> final accuracy evaluation
    # ------------------------------------------------------------------
    train_augmented, _ = make_train_imagefolders(dataset_root)
    test_dataset = make_test_imagefolder(dataset_root)

    num_classes = len(test_dataset.classes)

    if train_augmented.class_to_idx != test_dataset.class_to_idx:
        raise ValueError("Train and test class mappings differ")

    if sample_count > len(test_dataset):
        raise ValueError(
            f"sample_count={sample_count} exceeds CIFAR-10 test split "
            f"size={len(test_dataset)}"
        )

    # ------------------------------------------------------------------
    # 2. TEST SAMPLING
    #
    # EXACTLY the same sampling function as evaluate_width_latency.py.
    # ------------------------------------------------------------------
    test_indices = select_stratified_indices(
        test_dataset.targets,
        num_classes,
        sample_count,
        seed,
    )

    if len(test_indices) != sample_count:
        raise RuntimeError("Canonical test sampling returned wrong sample count")

    if len(set(test_indices)) != sample_count:
        raise RuntimeError("Canonical test sampling produced duplicate indices")

    # ------------------------------------------------------------------
    # 3. BN CALIBRATION
    #
    # EXACTLY like evaluate_width_latency.py.
    #
    # The partition argument is intentionally NOT used to select BN samples.
    # This is deliberate: the goal here is exact pipeline equivalence with
    # evaluate_width_latency.py.
    # ------------------------------------------------------------------
    _, bn_indices = stratified_train_partition(
        train_augmented.targets,
        len(train_augmented.classes),
        10,
        seed,
    )

    random.Random(seed).shuffle(bn_indices)

    bn_subset = Subset(
        train_augmented,
        bn_indices[:bn_calibration_batches * 32],
    )

    bn_loader = DataLoader(
        bn_subset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    # ------------------------------------------------------------------
    # 4. TEST LOADER
    #
    # EXACTLY like evaluate_width_latency.py.
    # ------------------------------------------------------------------
    test_loader = DataLoader(
        Subset(test_dataset, test_indices),
        batch_size=evaluation_batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ------------------------------------------------------------------
    # 5. MODEL
    # ------------------------------------------------------------------
    model = Model(
        num_classes=num_classes,
        input_size=32,
    ).to(device)

    checkpoint_info = load_model_checkpoint(
        model,
        checkpoint,
        device,
    )

    # Keep the checkpoint identity explicit in the output artifact.
    checkpoint_hash = sha256_file(checkpoint)

    # ------------------------------------------------------------------
    # 6. EVALUATE ALL CONFIGURATIONS
    #
    # Same BN recalibration + width/bit setter order as the old evaluator.
    # ------------------------------------------------------------------
    results = []

    for candidate_index, (width, bits) in enumerate(QAT_CANDIDATES):
        # Same per-candidate deterministic seeding pattern used by the
        # standalone width evaluator.
        candidate_seed = seed + candidate_index

        random.seed(candidate_seed)
        np.random.seed(candidate_seed)
        torch.manual_seed(candidate_seed)

        if device.type == "cuda":
            torch.cuda.manual_seed_all(candidate_seed)

        recalibrate_bn(
            model,
            bn_loader,
            width,
            bits,
            device,
            num_batches=min(
                bn_calibration_batches,
                len(bn_loader),
            ),
        )

        # Same order as evaluate_width_latency.py:
        set_model_width(model, width)
        set_model_bit_width(model, bits)
        model.eval()

        correct = 0
        total = 0

        with torch.inference_mode():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                predictions = logits.argmax(dim=1)

                correct += int(
                    (predictions == labels).sum().item()
                )
                total += int(labels.numel())

        if total != sample_count:
            raise RuntimeError(
                f"Candidate ({width}, {bits}) evaluated {total} samples; "
                f"expected {sample_count}"
            )

        accuracy_percent = 100.0 * correct / total

        row = {
            "width_mult": float(width),
            "bit_width": int(bits),
            "total": int(total),
            "correct": int(correct),
            "sample_count": int(total),
            "correct_count": int(correct),
            "accuracy_percent": float(accuracy_percent),
        }

        results.append(row)

        print(
            f"width={width:.2f} bits={bits} "
            f"accuracy={accuracy_percent:.2f}% "
            f"correct={correct}/{total}"
        )

    # Validate against the same table contract consumed by the controller.
    validate_accuracy_table(results)

    # ------------------------------------------------------------------
    # 7. OUTPUT ARTIFACT
    # ------------------------------------------------------------------
    output.parent.mkdir(parents=True, exist_ok=True)

    document = {
        "experiment": "USM QAT candidate accuracy evaluation",
        "model": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_info": checkpoint_info,
            "num_classes": num_classes,
            "input_size": 32,
        },
        "results": results,
        "data_protocol": {
            "source_split": "CIFAR-10 test",
            "accuracy_selection_split": "seeded stratified CIFAR-10 test subset",
            "final_evaluation_split": "CIFAR-10 test; not read by this script",
            "validation_sample_count": sample_count,
            "samples_per_class": sample_count // num_classes,
            "sampling": (
                "exact evaluate_width_latency.py stratified sampler: "
                "random.Random(seed), equal samples per class, sorted "
                "within class, class-interleaved final order"
            ),
            "seed": seed,
            "dataset_indices": test_indices,
            "evaluation_transform": (
                "scripts.usm_qat_data.make_test_imagefolder"
            ),
            "evaluation_batch_size": 1,
            "evaluation_shuffle": False,
            "evaluation_num_workers": 0,
            "evaluation_inference_mode": True,
            "evaluation_prediction": "argmax(dim=1)",
            "bn_calibration_split": "CIFAR-10 train",
            "bn_calibration_batches": bn_calibration_batches,
            "bn_calibration_batch_size": 32,
            "bn_calibration_sampling": (
                "exact evaluate_width_latency.py train partition "
                "and seeded shuffle"
            ),
            "bn_calibration_disjoint_from_test": True,
        },
        "inference_backend": (
            "PyTorch fake quantization; sub-32-bit modes are not "
            "integer-kernel execution"
        ),
        "pipeline_reference": (
            "evaluate_width_latency.py: test dataset, stratified sampling, "
            "batch_size=1, shuffle=False, inference_mode, argmax"
        ),
    }

    output.write_text(
        json.dumps(document, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nAccuracy artifact: {output}")
    print(
        f"Evaluation pipeline: CIFAR-10 test / "
        f"{sample_count} stratified samples / batch_size=1"
    )

    return document


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all USM width/bit candidates using the same "
            "CIFAR-10 test evaluation pipeline as evaluate_width_latency.py."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--partition",
        type=Path,
        required=True,
        help=(
            "Existing experiment partition file. Retained for CLI "
            "compatibility; BN indices are generated using the exact "
            "evaluate_width_latency.py protocol."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "cifar10",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Compatibility argument. Candidate evaluation is intentionally "
            "forced to batch_size=1 to match evaluate_width_latency.py."
        ),
    )
    parser.add_argument(
        "--bn-calibration-batches",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help=(
            "Number of stratified CIFAR-10 test examples used for candidate "
            "accuracy. Must be divisible by 10. Use 100 for direct "
            "comparison with the original evaluate_width_latency.py run."
        ),
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    if args.n_samples < 1:
        parser.error("--n-samples must be positive")

    if args.n_samples % 10 != 0:
        parser.error("--n-samples must be divisible by 10 for CIFAR-10 stratification")

    evaluate_all(
        checkpoint=resolve_usm_checkpoint(args.checkpoint),
        partition_path=args.partition,
        output=args.output,
        dataset_root=args.dataset_root,
        batch_size=1,
        bn_calibration_batches=args.bn_calibration_batches,
        seed=args.seed,
        device_name=args.device,
        sample_count=args.n_samples,
    )


if __name__ == "__main__":
    main()
