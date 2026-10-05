import os
import sys
import json
import csv
import time
import argparse
import hashlib
import numpy as np
import torch
import psutil
from datetime import datetime
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "resource_control"))

from resource_control.telemetry import ResourceMonitor, ResourceState
from resource_control.controller import (
    MeasuredFrontierLatencyModel,
    SurrogateBackedController,
)
from models import Model, set_model_width, set_model_bit_width, get_dataloaders, recalibrate_bn
from models.checkpoint_io import (
    checkpoint_sha256,
    load_model_checkpoint,
    resolve_usm_checkpoint,
)
from torch.utils.data import Subset
from resource_control.stress import ResourceStress
from scripts.experiment_data import class_interleaved_indices

def get_fp32_frontier(profile_path, accuracy_path, latency_policy="p95"):
    accuracies = {}
    with open(accuracy_path, mode="r", newline="", encoding="utf-8-sig") as accuracy_file:
        for row in csv.DictReader(accuracy_file):
            if int(row["bit_width"]) == 32:
                accuracies[float(row["width_mult"])] = float(row["accuracy_percent"])

    profiles_by_width = {}
    with open(profile_path, mode="r", newline="", encoding="utf-8-sig") as profile_file:
        for row in csv.DictReader(profile_file):
            if int(row["bit_width"]) != 32:
                continue
            width = float(row["width_mult"])
            if width not in accuracies:
                raise ValueError(f"No measured FP32 accuracy for width {width}")
            condition = row.get("resource_condition", "baseline") or "baseline"
            if condition in profiles_by_width.setdefault(width, {}):
                raise ValueError(f"Duplicate profile row for width {width} under {condition}")
            values = {
                "latency_ms": float(row["latency_ms"]),
                "std_ms": float(row["std_ms"]),
                "latency_p50_ms": float(row["p50_ms"]),
                "latency_p95_ms": float(row["p95_ms"]),
                "latency_p99_ms": float(row["p99_ms"]),
            }
            profiles_by_width[width][condition] = values
    if set(profiles_by_width) != {0.25, 0.5, 0.75, 1.0}:
        raise ValueError("Profile must contain exactly the four supported FP32 widths")
    frontier = []
    for width, condition_profiles in profiles_by_width.items():
        if "baseline" not in condition_profiles:
            raise ValueError(f"Width {width} is missing a baseline resource profile")
        baseline = condition_profiles["baseline"]
        frontier.append({
            "config": (width, 32),
            **baseline,
            "condition_profiles": condition_profiles,
            "acc": accuracies[width],
        })
    latency_key = {
        "avg": "latency_ms",
        "p50": "latency_p50_ms",
        "p95": "latency_p95_ms",
        "p99": "latency_p99_ms",
    }[latency_policy]
    return sorted(frontier, key=lambda row: row[latency_key])

def generate_or_load_workload(workload_file, seed, n_samples, trace):
    if workload_file and os.path.exists(workload_file):
        with open(workload_file, 'r') as f:
            data = json.load(f)
            return data['requests']
    
    import subprocess
    script = os.path.join(project_root, "scripts", "generate_workload.py")
    out_file = workload_file or os.path.join(project_root, "results", "static_vs_usm_fp32", "workload", "default.json")
    subprocess.run([sys.executable, script, "--seed", str(seed), "--n-samples", str(n_samples), "--trace", trace, "--output", out_file], check=True)
    with open(out_file, 'r') as f:
        data = json.load(f)
        return data['requests']


def assert_request_contract(requests):
    if not requests:
        raise ValueError("No replay requests were provided")
    ids = [req.get("request_id") for req in requests]
    if len(ids) != len(set(ids)):
        raise ValueError("Request IDs are not unique across the replay workload")
    for index, req in enumerate(requests):
        if int(req.get("sample_index", index)) != index:
            raise ValueError(f"Sample index mismatch for request {req.get('request_id')}: expected {index}, got {req.get('sample_index')}")
        if int(req.get("batch_size", 1)) != 1:
            raise ValueError(f"Batch size is not 1 for request {req.get('request_id')}")
        if "budget_ms" not in req:
            raise ValueError(f"Request {req.get('request_id')} is missing a deadline budget_ms")
        resource_scenario_for_request(req)
    return ids


def resource_scenario_for_request(request):
    scenario = request.get("resource_scenario", "baseline")
    scenario = {
        "synthetic_baseline": "baseline",
        "synthetic_contention": "contention",
    }.get(scenario, scenario)
    if scenario not in ("baseline", "contention"):
        raise ValueError(f"Unsupported replay resource scenario: {scenario}")
    return scenario


def run_arm(arm_name, model, dataloader, requests, pinned_width, controller, device, run_id, hw_fingerprint, model_provenance=None, warmup_requests=20):
    if arm_name not in ("usm_pinned", "usm_adaptive"):
        raise ValueError("arm_name must be usm_pinned or usm_adaptive")
    if arm_name == "usm_adaptive" and controller is None:
        raise ValueError("USM adaptive arm requires a controller")
    if arm_name == "usm_pinned" and controller is not None:
        raise ValueError("USM pinned arm must not receive a controller")

    assert_request_contract(requests)
    results = []
    model.eval()
    supported_widths = (0.25, 0.5, 0.75, 1.0)

    # Warm-up requests are excluded; adaptive warms only its initial controller width
    # so the first transition and first forward on another width are measured.
    warmup_input = torch.zeros((1, 3, 32, 32), device=device)
    adaptive_initial_width = (
        getattr(controller, "current_config", (pinned_width, 32))[0]
        if controller is not None else pinned_width
    )
    warmup_sequence = (
        [pinned_width] * warmup_requests
        if arm_name == "usm_pinned"
        else [adaptive_initial_width] * warmup_requests
    )
    with torch.inference_mode():
        for width in warmup_sequence:
            set_model_width(model, width)
            set_model_bit_width(model, 32)
            model(warmup_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

    current_width = adaptive_initial_width
    live_monitor = ResourceMonitor(live_refresh_s=0.05, smoothing_alpha=0.5)
    process = psutil.Process()
    stress = ResourceStress(device, cpu=True, gpu=device.type == "cuda")
    stress.start()
    replay_origin = time.perf_counter()
    stress.schedule(requests, replay_origin)
    data_iterator = iter(dataloader)

    for request_position, req in enumerate(requests):
        scenario = resource_scenario_for_request(req)
        scheduled_arrival = replay_origin + float(
            req.get("arrival_s", request_position * 0.01)
        )
        wait_s = scheduled_arrival - time.perf_counter()
        if wait_s > 0:
            time.sleep(wait_s)
        service_start = time.perf_counter()
        batch = next(data_iterator)
        image, label = batch
        image = image.to(device)
        budget = req["budget_ms"]
        idx = req["sample_index"]
        req_id = req["request_id"]

        measured_state = live_monitor.read()
        resource_condition = stress.current_condition()
        controller_state = ResourceState(
            cpu_pct=measured_state.cpu_pct,
            mem_available_mb=measured_state.mem_available_mb,
            battery_pct=measured_state.battery_pct,
            thermal_c=measured_state.thermal_c,
            timestamp=measured_state.timestamp,
            resource_condition=resource_condition,
        )
        request_start = scheduled_arrival
        controller_overhead = 0.0

        if arm_name == "usm_pinned":
            selected_w = pinned_width
            switched = False
            pred_lat = None
        elif arm_name == "usm_adaptive":
            t_c0 = time.perf_counter()
            selected_cfg = controller.select(controller_state, latency_budget_ms=budget, hw_fingerprint=hw_fingerprint)
            t_c1 = time.perf_counter()
            controller_overhead = (t_c1 - t_c0) * 1000.0
            selected_w = selected_cfg[0]
            switched = selected_w != current_width
            pred_lat = controller.last_predicted_latency

        if selected_w not in supported_widths:
            raise ValueError(f"Controller selected unsupported width {selected_w}")
        
        width_change_start = time.perf_counter()
        set_model_width(model, selected_w)
        set_model_bit_width(model, 32)
        width_switch_overhead = (time.perf_counter() - width_change_start) * 1000.0
        if selected_w != current_width:
            switched = True
        
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(image)
            if device.type == "cuda":
                torch.cuda.synchronize()
        t1 = time.perf_counter()
        lat_ms = (t1 - t0) * 1000.0
        pred = out.argmax(dim=1)
        response_ready = time.perf_counter()
        end_to_end_ms = (response_ready - request_start) * 1000.0
        queue_delay_ms = max(0.0, (service_start - scheduled_arrival) * 1000.0)
        current_width = selected_w

        feedback_update_ms = 0.0
        if arm_name == "usm_adaptive" and pred_lat is not None:
            feedback_start = time.perf_counter()
            controller.update_feedback(lat_ms, pred_lat, config=(selected_w, 32))
            feedback_update_ms = (time.perf_counter() - feedback_start) * 1000.0
            
        is_correct = (pred == label.to(device)).item()
        
        res = {
            "run_id": run_id,
            "request_id": req_id,
            "arm": arm_name,
            "timestamp_unix_s": time.time(),
            "sample_index": idx,
            "input_sample_id": req.get("input_sample_id", idx),
            "dataset_index": req.get("dataset_index"),
            "dataset_image_path": req.get("dataset_image_path"),
            "arrival_s": req.get("arrival_s"),
            "interarrival_s": req.get("interarrival_s"),
            "replay_mode": "paced_sequential_queue_replay",
            "batch_size": int(image.shape[0]),
            "budget_ms": budget,
            "selected_bit_width": 32,
            "selected_width": selected_w,
            "predicted_latency_ms": pred_lat,
            "model_latency_ms": lat_ms,
            "end_to_end_latency_ms": end_to_end_ms,
            "queue_delay_ms": queue_delay_ms,
            "controller_overhead_ms": controller_overhead,
            "width_switch_overhead_ms": width_switch_overhead,
            "controller_feedback_update_ms": feedback_update_ms,
            "cold_switch_cost_included": False,
            "width_switch_cost_included": True,
            "is_correct": is_correct,
            "missed_deadline": end_to_end_ms > budget,
            "switched": switched,
            "resource_condition": resource_condition,
            "scheduled_resource_scenario": scenario,
            "resource_trace_kind": "controlled_cpu_gpu_stress" if resource_condition == "contention" else "controlled_baseline",
            "observed_cpu_pct": float(measured_state.cpu_pct),
            "observed_mem_available_mb": float(measured_state.mem_available_mb),
            "observed_thermal_c": measured_state.thermal_c,
            "process_rss_mb": process.memory_info().rss / (1024.0 * 1024.0),
            "gpu_peak_allocated_mb": (
                torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
                if device.type == "cuda" else None
            ),
            "device_fingerprint": hw_fingerprint,
            "model_checkpoint_sha256": (model_provenance or {}).get("checkpoint_sha256"),
        }
        results.append(res)
    stress.wait_for_schedule()
    stress.close()
    if len(results) != len(requests):
        raise ValueError(
            f"Workload/dataset alignment mismatch: {len(requests)} requests, "
            f"{len(results)} evaluated samples"
        )
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("usm_pinned", "usm_adaptive", "all"), default="all")
    parser.add_argument("--pinned-width", type=float)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--trace", choices=("sinusoidal", "step", "bursty", "heldout"), default="sinusoidal")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--workload-file", type=str)
    parser.add_argument("--config", type=Path, default=Path(project_root) / "results" / "static_vs_usm_fp32" / "config.json")
    parser.add_argument("--profile", type=Path, default=Path(project_root) / "results" / "static_vs_usm_fp32" / "profiles" / "fp32_width_profile.csv")
    parser.add_argument("--accuracy-csv", type=Path, default=Path(project_root) / "results" / "config_accuracy_seed12345.csv")
    parser.add_argument("--usm-checkpoint", type=Path)
    parser.add_argument("--raw-dir", type=Path, default=Path(project_root) / "results" / "static_vs_usm_fp32" / "raw")
    parser.add_argument("--run-id", type=str)
    parser.add_argument("--policy", choices=("avg", "p50", "p95", "p99"), default="p95")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--warmup-requests", type=int, default=20)
    parser.add_argument("--bn-calibration-batches", type=int, default=10)
    parser.add_argument("--selection-calibration-per-class", type=int, default=100)
    args = parser.parse_args()
    
    if args.smoke_test:
        args.repetitions = 1
        args.n_samples = 10
        
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with args.config.expanduser().open("r", encoding="utf-8") as config_file:
        cfg = json.load(config_file)
    usm_checkpoint_path = resolve_usm_checkpoint(args.usm_checkpoint)
    usm_checkpoint_hash = checkpoint_sha256(usm_checkpoint_path)
    if cfg.get("usm_checkpoint_sha256") != usm_checkpoint_hash:
        raise ValueError("Frozen width-selection config does not match the supplied USM checkpoint")
    configured_width = float(cfg["pinned_width"])
    if args.pinned_width is not None and args.pinned_width != configured_width:
        raise ValueError("--pinned-width conflicts with the frozen experiment config")
    args.pinned_width = configured_width

    accuracy_json = args.accuracy_csv.with_suffix(".json")
    with accuracy_json.open("r", encoding="utf-8") as accuracy_file:
        accuracy_document = json.load(accuracy_file)
    if accuracy_document["model"]["checkpoint_sha256"] != usm_checkpoint_hash:
        raise ValueError("Accuracy artifact does not match the supplied USM checkpoint")
    if accuracy_document["dataset"].get("selection_calibration_per_class") != args.selection_calibration_per_class:
        raise ValueError("Accuracy artifact calibration partition does not match the replay exclusion")
        
    # Generate/load workload
    workload_file = args.workload_file or os.path.join(project_root, "results", "static_vs_usm_fp32", "workload", f"trace_{args.trace}_seed_{args.seed}.json")
    requests = generate_or_load_workload(workload_file, args.seed, args.n_samples, args.trace)
    assert_request_contract(requests)
    if args.smoke_test:
        requests = requests[:10]
        assert_request_contract(requests)
        
    # Load profile
    profile_path = args.profile.expanduser().resolve()
    frontier = get_fp32_frontier(
        profile_path, args.accuracy_csv.expanduser().resolve(), args.policy
    )
    if any(set(row["condition_profiles"]) != {"baseline", "contention"} for row in frontier):
        raise ValueError("Each FP32 width must have measured baseline and contention profiles")
    with profile_path.open("r", newline="", encoding="utf-8-sig") as profile_file:
        profile_hashes = {row["checkpoint_sha256"] for row in csv.DictReader(profile_file)}
    if profile_hashes != {usm_checkpoint_hash}:
        raise ValueError("FP32 profile does not match the supplied USM checkpoint")
    
    # Dataloader
    train_loader, test_loader = get_dataloaders()
    test_dataset = test_loader.dataset
    calibration_indices = set(
        int(index) for index in accuracy_document["dataset"]["selected_dataset_indices"]
    )
    subset_indices = class_interleaved_indices(
        test_dataset.targets,
        len(test_dataset.classes),
        start_offset=args.selection_calibration_per_class,
        seed=args.seed,
    )
    if calibration_indices.intersection(subset_indices):
        raise ValueError("Width-accuracy calibration samples overlap the replay evaluation samples")
    if len(requests) > len(subset_indices):
        raise ValueError(f"Workload has {len(requests)} requests but test split has only {len(subset_indices)} samples")
    requests = requests[:len(requests)]
    for request_index, request in enumerate(requests):
        if int(request.get("sample_index", -1)) != request_index:
            raise ValueError("Workload sample_index values must be the ordered sequence 0..N-1")
        if int(request.get("batch_size", 1)) != 1:
            raise ValueError("This experiment requires batch_size=1")
        dataset_index = subset_indices[request_index]
        request["dataset_index"] = dataset_index
        request["dataset_image_path"] = os.path.relpath(
            test_dataset.samples[dataset_index][0], project_root
        ).replace(os.sep, "/")
    workload_path = Path(workload_file).expanduser().resolve()
    workload_document = json.loads(workload_path.read_text(encoding="utf-8"))
    workload_document["meta"]["dataset_order"] = "balanced class-interleaved ImageFolder test order"
    workload_document["meta"]["dataset_split"] = "test_replay_holdout"
    workload_document["meta"]["dataset_root"] = "cifar10/test"
    workload_document["meta"]["replay_mode"] = "paced_sequential_queue_replay"
    workload_document["meta"]["width_accuracy_calibration_split"] = "disjoint stratified CIFAR-10 test slice"
    workload_document["meta"]["width_accuracy_calibration_samples"] = len(calibration_indices)
    workload_document["meta"]["dataset_mapping_sha256"] = hashlib.sha256(
        "\n".join(request["dataset_image_path"] for request in requests).encode("utf-8")
    ).hexdigest()
    workload_document["requests"] = requests
    bound_workload_path = workload_path.with_name("workload_bound.json")
    if bound_workload_path.exists():
        raise FileExistsError(f"Refusing to overwrite bound workload: {bound_workload_path}")
    bound_workload_path.write_text(
        json.dumps(workload_document, indent=2) + "\n", encoding="utf-8"
    )

    sample_order_path = workload_path.parent.parent / "sample_order.json"
    if sample_order_path.exists():
        raise FileExistsError(f"Refusing to overwrite sample mapping: {sample_order_path}")
    sample_order_path.write_text(json.dumps({
        "dataset_split": "test_replay_holdout",
        "excluded_selection_calibration_samples": len(calibration_indices),
        "order": [
            {"input_sample_id": request["input_sample_id"], "dataset_index": request["dataset_index"],
             "dataset_image_path": request["dataset_image_path"]}
            for request in requests
        ],
        "sha256": workload_document["meta"]["dataset_mapping_sha256"],
    }, indent=2) + "\n", encoding="utf-8")

    subset_dataset = Subset(test_dataset, subset_indices[:len(requests)])
    eval_loader = torch.utils.data.DataLoader(subset_dataset, batch_size=1, shuffle=False)
    
    usm_model = Model(num_classes=10, input_size=32).to(device)
    usm_provenance = load_model_checkpoint(usm_model, usm_checkpoint_path, device)
    if usm_provenance["checkpoint_sha256"] != usm_checkpoint_hash:
        raise RuntimeError("USM checkpoint changed while preparing the experiment")

    # Pre-calibrate BN
    print("Pre-calibrating BN...")
    for width_index, w in enumerate([0.25, 0.5, 0.75, 1.0]):
        calibration_seed = args.seed + width_index
        torch.manual_seed(calibration_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(calibration_seed)
        sampler_generator = getattr(train_loader.sampler, "generator", None)
        if sampler_generator is None:
            sampler_generator = torch.Generator()
            train_loader.sampler.generator = sampler_generator
        sampler_generator.manual_seed(calibration_seed)
        recalibrate_bn(
            usm_model, train_loader, w, 32, device,
            num_batches=args.bn_calibration_batches,
        )
        
    monitor = ResourceMonitor()
    hw_fingerprint = monitor.get_hardware_fingerprint(device)
    
    arms_to_run = ["usm_pinned", "usm_adaptive"] if args.arm == "all" else [args.arm]
    
    out_dir = str(args.raw_dir.expanduser().resolve())
    os.makedirs(out_dir, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_run_id = args.run_id or f"run_{ts}_{args.seed}"

    expected_outputs = [
        os.path.join(out_dir, f"{arm}_{base_run_id}_rep{rep + 1:03d}.jsonl")
        for rep in range(args.repetitions)
        for arm in arms_to_run
    ]
    existing_outputs = [path for path in expected_outputs if os.path.exists(path)]
    if existing_outputs:
        raise FileExistsError(f"Refusing to overwrite existing raw run: {existing_outputs[0]}")
    
    for rep in range(args.repetitions):
        run_id = f"{base_run_id}_rep{rep+1:03d}"
        ordered_arms = arms_to_run if rep % 2 == 0 else list(reversed(arms_to_run))
        for arm in ordered_arms:
            print(f"Running rep {rep+1}, arm {arm}")
            controller = None
            if arm == "usm_adaptive":
                surrogate_model = MeasuredFrontierLatencyModel(
                    frontier, latency_policy=args.policy
                )
                controller = SurrogateBackedController(
                    frontier=frontier, surrogate_model=surrogate_model, min_dwell_s=0.0,
                    latency_policy=args.policy
                )
            
            results = run_arm(
                arm, usm_model, eval_loader, requests, args.pinned_width,
                controller, device, run_id, hw_fingerprint, usm_provenance,
                warmup_requests=args.warmup_requests,
            )
            
            # Save raw JSONL
            out_file = os.path.join(out_dir, f"{arm}_{run_id}.jsonl")
            with open(out_file, 'x', encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
                    
if __name__ == "__main__":
    main()
