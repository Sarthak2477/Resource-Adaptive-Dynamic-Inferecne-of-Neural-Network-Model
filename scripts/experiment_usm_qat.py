import argparse
import csv
import json
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import Model, recalibrate_bn, set_model_bit_width, set_model_width
from models.checkpoint_io import load_model_checkpoint, resolve_usm_checkpoint
from resource_control.qat_experiment import (
    MeasuredQATController,
    QAT_CANDIDATES,
    candidate_configurations,
    classify_switch,
    validate_accuracy_table,
    validate_latency_table,
)
from resource_control.stress import ResourceStress
from resource_control.telemetry import ResourceMonitor
from scripts.profile_usm_qat_configs import query_gpu_metrics
from scripts.usm_qat_data import make_test_imagefolder, make_train_imagefolders


class ConditionSchedule:
    def __init__(self, requests, origin, cpu_stress, gpu_stress):
        self.requests = requests
        self.origin = origin
        self.cpu_stress = cpu_stress
        self.gpu_stress = gpu_stress
        self.stop_event = threading.Event()
        self.condition_lock = threading.Lock()
        self.condition = "baseline"
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        for index, request in enumerate(self.requests):
            target = self.origin + float(request["arrival_s"])
            while not self.stop_event.is_set():
                remaining = target - time.perf_counter()
                if remaining <= 0:
                    break
                self.stop_event.wait(min(remaining, 0.01))
            if self.stop_event.is_set():
                return
            condition = request["resource_condition"]
            if self.cpu_stress is not None:
                self.cpu_stress.set_enabled(condition == "cpu_contention")
            if self.gpu_stress is not None:
                self.gpu_stress.set_enabled(condition == "gpu_contention")
            with self.condition_lock:
                self.condition = condition

    def current(self):
        with self.condition_lock:
            return self.condition

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=5.0)
        if self.thread.is_alive():
            raise RuntimeError("Resource condition schedule failed to stop")


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def read_accuracy(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = document["results"]
    accuracies = validate_accuracy_table(rows)
    return document, accuracies


def configure_model(model, config):
    set_model_width(model, float(config[0]))
    set_model_bit_width(model, int(config[1]))


def validate_workload(workload, conditions):
    requests = workload["requests"]
    if not requests:
        raise ValueError("QAT replay workload is empty")
    identifiers = set()
    for index, request in enumerate(requests):
        if request["request_id"] in identifiers:
            raise ValueError("Duplicate request_id in QAT workload")
        identifiers.add(request["request_id"])
        if int(request["sample_index"]) != index:
            raise ValueError("QAT workload sample_index must be ordered 0..N-1")
        if request["resource_condition"] not in conditions:
            raise ValueError(f"Workload condition has no measured profile: {request['resource_condition']}")
        if float(request["deadline_ms"]) <= 0:
            raise ValueError("Every QAT workload request needs a positive deadline")
    return requests


def prepare_bn_statistics(model, train_bn_dataset, bn_indices, device, batches, seed):
    loader = DataLoader(
        Subset(train_bn_dataset, bn_indices), batch_size=32, shuffle=False,
        num_workers=0,
    )
    for index, config in enumerate(candidate_configurations()):
        candidate_seed = seed + index
        random.seed(candidate_seed)
        torch.manual_seed(candidate_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(candidate_seed)
        recalibrate_bn(model, loader, config[0], config[1], device, num_batches=batches)


def warm_candidates(model, device, warmup_requests, seed):
    rng = random.Random(seed)
    warm_order = list(QAT_CANDIDATES)
    rng.shuffle(warm_order)
    sample = torch.zeros((1, 3, 32, 32), device=device)
    model.eval()
    with torch.inference_mode():
        for config in warm_order:
            configure_model(model, config)
            for _ in range(warmup_requests):
                model(sample)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
    return warm_order


def run_arm(
    arm, model, requests, test_dataset, test_indices, pinned_config, controller,
    device, run_id, repetition, checkpoint_hash, hw_fingerprint, warmup_rounds=1,
    seed=12345, stress_size=512, enable_stress=False,
):
    if arm not in ("usm_pinned", "usm_adaptive"):
        raise ValueError("Unsupported QAT policy arm")
    if arm == "usm_adaptive" and controller is None:
        raise ValueError("Adaptive arm requires a measured QAT controller")
    if arm == "usm_pinned" and controller is not None:
        raise ValueError("Pinned arm must not receive a controller")

    warm_order = warm_candidates(model, device, warmup_rounds, seed)

    eval_loader = DataLoader(
        Subset(test_dataset, test_indices[:len(requests)]), batch_size=1,
        shuffle=False, num_workers=0,
    )
    cpu_stress = ResourceStress(
        device, cpu=True, gpu=False, matrix_size=128, idle_s=0.02,
    ) if enable_stress else None
    gpu_stress = None
    if cpu_stress is not None:
        cpu_stress.start()
    schedule = None
    records = []
    monitor = ResourceMonitor(live_refresh_s=0.05, smoothing_alpha=0.5)
    process = psutil.Process()
    previous_config = pinned_config
    if controller is not None:
        controller.current_config = pinned_config
    request_iterator = iter(eval_loader)
    replay_origin = time.perf_counter()
    schedule = ConditionSchedule(requests, replay_origin, cpu_stress, gpu_stress)
    schedule.start()
    try:
        for position, request in enumerate(requests):
            arrival = replay_origin + float(request["arrival_s"])
            delay = arrival - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            service_start = time.perf_counter()
            images, labels = next(request_iterator)
            images = images.to(device)
            labels = labels.to(device)
            resource_condition = request["resource_condition"]
            observed = monitor.read()
            controller_overhead_ms = 0.0
            decision = None
            if arm == "usm_pinned":
                selected_config = pinned_config
                predicted_p95_ms = None
                predicted_switch_ms = 0.0
                fallback = False
                feasible = None
                fallback_violates_deadline = False
                feasible_count = None
                fp32_feasible_count = None
                precision_fallback_used = False
                candidate_evaluations = []
            else:
                started = time.perf_counter()
                decision = controller.select(resource_condition, float(request["deadline_ms"]))
                controller_overhead_ms = (time.perf_counter() - started) * 1000.0
                selected_config = decision.config
                predicted_p95_ms = decision.predicted_p95_ms
                predicted_switch_ms = decision.predicted_switch_ms
                fallback = decision.fallback
                feasible = decision.feasible
                fallback_violates_deadline = decision.fallback_violates_deadline
                feasible_count = decision.feasible_candidate_count
                fp32_feasible_count = decision.fp32_feasible_candidate_count
                precision_fallback_used = decision.precision_fallback_used
                candidate_evaluations = list(decision.candidates)

            switch_type = classify_switch(previous_config, selected_config)
            switch_started = time.perf_counter()
            configure_model(model, selected_config)
            switch_overhead_ms = (time.perf_counter() - switch_started) * 1000.0 if switch_type != "none" else 0.0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            forward_started = time.perf_counter()
            with torch.inference_mode():
                logits = model(images)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            model_latency_ms = (time.perf_counter() - forward_started) * 1000.0
            prediction = logits.argmax(dim=1)
            response_time = time.perf_counter()
            end_to_end_ms = (response_time - arrival) * 1000.0
            queue_delay_ms = max(0.0, (service_start - arrival) * 1000.0)
            correct = bool((prediction == labels).item())
            measured_gpu = query_gpu_metrics()
            record = {
                "run_id": run_id,
                "repetition": repetition,
                "arm": arm,
                "request_id": request["request_id"],
                "sample_index": request["sample_index"],
                "sample_id": request["input_sample_id"],
                "input_sample_id": request["input_sample_id"],
                "dataset_index": int(test_indices[position]),
                "arrival_s": float(request["arrival_s"]),
                "deadline_ms": float(request["deadline_ms"]),
                "phase": request.get("phase"),
                "resource_state": resource_condition,
                "scheduled_resource_condition": request["resource_condition"],
                "resource_condition_at_service": resource_condition,
                "observed_cpu_pct": float(observed.cpu_pct),
                "available_memory_mb": float(observed.mem_available_mb),
                "thermal_c": observed.thermal_c,
                **measured_gpu,
                "previous_width": previous_config[0],
                "previous_bit_width": previous_config[1],
                "selected_width": selected_config[0],
                "selected_bit_width": selected_config[1],
                "predicted_p95_ms": predicted_p95_ms,
                "predicted_p95_latency_ms": predicted_p95_ms,
                "predicted_switch_cost_ms": predicted_switch_ms,
                "measured_switch_setting_ms": switch_overhead_ms,
                "switch_type": switch_type,
                "switched": switch_type != "none",
                "switch": switch_type != "none",
                "fallback": fallback,
                "feasible": feasible,
                "fallback_violates_deadline": fallback_violates_deadline,
                "feasible_candidate_count": feasible_count,
                "fp32_feasible_candidate_count": fp32_feasible_count,
                "precision_fallback_used": precision_fallback_used,
                "candidate_evaluations": candidate_evaluations,
                "model_latency_ms": model_latency_ms,
                "measured_model_latency_ms": model_latency_ms,
                "controller_overhead_ms": controller_overhead_ms,
                "queue_delay_ms": queue_delay_ms,
                "end_to_end_latency_ms": end_to_end_ms,
                "missed_deadline": end_to_end_ms > float(request["deadline_ms"]),
                "deadline_met": end_to_end_ms <= float(request["deadline_ms"]),
                "deadline_violation_ms": max(0.0, end_to_end_ms - float(request["deadline_ms"])),
                "correct": correct,
                "process_rss_mb": process.memory_info().rss / (1024.0 ** 2),
                "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024.0 ** 2) if device.type == "cuda" else None,
                "checkpoint_sha256": checkpoint_hash,
                "inference_backend": "PyTorch QAT fake quantization; not real INT4/INT8 kernels",
            }
            records.append(record)
            previous_config = selected_config
    finally:
        schedule.close()
        if cpu_stress is not None:
            cpu_stress.close()
        if gpu_stress is not None:
            gpu_stress.close()
    if len(records) != len(requests):
        raise ValueError("Replay did not emit one result per frozen workload request")
    return records, warm_order


def run_repetitions(args):
    run_root = Path(args.run_root).resolve()
    raw_dir = run_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    accuracy_document, accuracies = read_accuracy(args.accuracy_json)
    if accuracy_document["data_protocol"]["final_evaluation_split"] != "CIFAR-10 test; not read by this script":
        raise ValueError("QAT accuracy artifact does not declare train-only selection data")
    profile_rows = read_csv(args.profile_csv)
    conditions = sorted({row["resource_condition"] for row in profile_rows})
    profiles = validate_latency_table(profile_rows, conditions)
    workload = json.loads(Path(args.workload).read_text(encoding="utf-8"))
    requests = validate_workload(workload, conditions)
    pinned_document = json.loads(Path(args.pinned_config).read_text(encoding="utf-8"))
    pinned_config = (float(pinned_document["width_mult"]), int(pinned_document["bit_width"]))
    if pinned_config != (1.0, 32):
        raise ValueError("The final comparison requires pinned USM configuration (1.0, 32)")
    switch_rows = read_csv(args.switch_costs)
    switch_costs = {row["switch_type"]: float(row["p95_ms"]) for row in switch_rows}
    if set(switch_costs) != {"width_only", "bit_only", "width_and_bit"}:
        raise ValueError("Switch cost table must cover width-only, bit-only, and joint switches")

    checkpoint = resolve_usm_checkpoint(args.checkpoint)
    if args.device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    selected_device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    device = torch.device(selected_device)
    torch.set_num_threads(args.threads)
    train_bn, _ = make_train_imagefolders(args.dataset_root)
    partition = json.loads(Path(args.partition).read_text(encoding="utf-8"))
    model = Model(num_classes=len(train_bn.classes), input_size=32).to(device)
    provenance = load_model_checkpoint(model, checkpoint, device)
    prepare_bn_statistics(
        model, train_bn, partition["bn_calibration_indices"], device,
        args.bn_calibration_batches, args.seed,
    )
    test_dataset = make_test_imagefolder(args.dataset_root)
    test_indices = [int(index) for index in workload["dataset_indices"]]
    if len(test_indices) != len(requests) or len(set(test_indices)) != len(test_indices):
        raise ValueError("Workload test dataset index mapping is invalid")
    if any(index < 0 or index >= len(test_dataset) for index in test_indices):
        raise ValueError("Workload includes a test dataset index outside the test split")

    expected_files = [
        raw_dir / f"{arm}_{args.run_id}_rep{rep + 1:03d}.jsonl"
        for rep in range(args.repetitions)
        for arm in ("usm_pinned", "usm_adaptive")
    ]
    if any(path.exists() for path in expected_files):
        raise FileExistsError("Refusing to overwrite QAT raw replay artifacts")
    monitor = ResourceMonitor()
    fingerprint = monitor.get_hardware_fingerprint(device)
    for repetition in range(args.repetitions):
        arms = ["usm_pinned", "usm_adaptive"] if repetition % 2 == 0 else ["usm_adaptive", "usm_pinned"]
        for arm in arms:
            controller = None
            if arm == "usm_adaptive":
                controller = MeasuredQATController(
                    accuracies, profiles, switch_costs,
                )
            records, warm_order = run_arm(
                arm, model, requests, test_dataset, test_indices, pinned_config,
                controller, device, args.run_id, repetition + 1,
                provenance["checkpoint_sha256"], fingerprint, args.warmup_rounds,
                args.seed, args.stress_matrix_size,
                args.enable_stress,
            )
            output = raw_dir / f"{arm}_{args.run_id}_rep{repetition + 1:03d}.jsonl"
            with output.open("x", encoding="utf-8") as target:
                for record in records:
                    target.write(json.dumps(record) + "\n")
            warmup_path = run_root / "raw" / f"warmup_{arm}_{args.run_id}_rep{repetition + 1:03d}.json"
            warmup_path.write_text(json.dumps([
                {"width_mult": width, "bit_width": bits} for width, bits in warm_order
            ], indent=2) + "\n", encoding="utf-8")
            print(f"Completed repetition {repetition + 1}/{args.repetitions}, arm={arm}, requests={len(records)}")


def main():
    parser = argparse.ArgumentParser(description="Run paired pinned versus adaptive USM QAT width-bit replay.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--accuracy-json", type=Path, required=True)
    parser.add_argument("--profile-csv", type=Path, required=True)
    parser.add_argument("--switch-costs", type=Path, required=True)
    parser.add_argument("--pinned-config", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "cifar10")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--bn-calibration-batches", type=int, default=10)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--stress-matrix-size", type=int, default=512)
    parser.add_argument("--enable-stress", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if min(args.repetitions, args.threads, args.bn_calibration_batches, args.stress_matrix_size) < 1 or args.warmup_rounds < 0:
        parser.error("repetitions, threads, BN batches, and stress matrix size must be positive; warmup rounds cannot be negative")
    run_repetitions(args)


if __name__ == "__main__":
    main()