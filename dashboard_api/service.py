from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import platform
import runpy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_CANDIDATES = (
    PROJECT_ROOT / "models" / "checkpoint" / "us_resnet_epoch100_checkpoint.pt.zip",
    PROJECT_ROOT / "models" / "checkpoint" / "us_resnet_epoch100_checkpoint.pt",
    PROJECT_ROOT / "models" / "checkpoints" / "best_model.pt",
    PROJECT_ROOT / "models" / "checkpoints" / "us_resnet_epoch100_checkpoint.pt",
    PROJECT_ROOT / "us_resnet_epoch100_checkpoint.pt",
)


class DashboardData:
    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.project_root = Path(project_root).resolve()
        self.results_root = (self.project_root / "results").resolve()

    def model_configurations(self) -> dict[str, Any]:
        config_path = self.project_root / "models" / "config.py"
        module = runpy.run_path(str(config_path))
        flags = module["FLAGS"]
        configurations = [
            {"width": float(width), "bit_width": int(bits), "id": f"w{width:g}-b{bits}"}
            for width, bits in flags.deploy_configs
        ]
        return {
            "architecture": f"ResNet-{flags.depth}",
            "dataset": flags.dataset,
            "class_count": int(flags.num_classes),
            "image_size": int(flags.image_size),
            "widths": sorted({row["width"] for row in configurations}),
            "bit_widths": sorted({row["bit_width"] for row in configurations}),
            "configurations": configurations,
            "precision_semantics": {
                "32": "Full precision floating-point path",
                "other": "Fake quantization in the dynamic model; not integer-kernel acceleration",
            },
        }

    def system_info(self) -> dict[str, Any]:
        checkpoint = next((path for path in CHECKPOINT_CANDIDATES if path.is_file()), None)
        cifar_test = self.project_root / "cifar10" / "test"
        dataset_ready = cifar_test.is_dir() and any(cifar_test.iterdir())
        profiles = sorted(path.name for path in (self.project_root / "profiles").glob("*_profile.csv"))
        torch_available = importlib.util.find_spec("torch") is not None
        torchvision_available = importlib.util.find_spec("torchvision") is not None
        cuda_available: bool | None = None
        if torch_available:
            try:
                import torch

                cuda_available = bool(torch.cuda.is_available())
            except Exception:
                torch_available = False
        return {
            "status": "connected",
            "architecture": platform.machine(),
            "operating_system": platform.platform(),
            "python_version": platform.python_version(),
            "torch_available": torch_available,
            "torchvision_available": torchvision_available,
            "cuda_available": cuda_available,
            "checkpoint": str(checkpoint.relative_to(self.project_root)) if checkpoint else None,
            "dataset_available": dataset_ready,
            "dataset_path": "cifar10/test",
            "profiles": profiles,
            "real_inference_preflight": {
                "ready": bool(checkpoint and dataset_ready and torch_available and torchvision_available),
                "requirements": {
                    "checkpoint": bool(checkpoint),
                    "cifar10_test_data": dataset_ready,
                    "torch": torch_available,
                    "torchvision": torchvision_available,
                },
                "note": "The current evaluator uses a synthetic resource/deadline trace; model latency is measured on the selected device.",
            },
        }

    @staticmethod
    def system_resources(evaluator_pid: int | None = None) -> dict[str, Any]:
        import psutil

        memory = psutil.virtual_memory()
        process_metrics = None
        if evaluator_pid is not None:
            try:
                process = psutil.Process(evaluator_pid)
                process_metrics = {
                    "pid": evaluator_pid,
                    "cpu_percent": process.cpu_percent(interval=None),
                    "resident_memory_mb": process.memory_info().rss / 1_000_000,
                }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        temperatures = []
        try:
            for sensor_name, readings in psutil.sensors_temperatures().items():
                temperatures.extend(
                    {
                        "sensor": reading.label or sensor_name,
                        "celsius": float(reading.current),
                    }
                    for reading in readings
                    if reading.current is not None
                )
        except (AttributeError, OSError):
            pass

        gpu = None
        try:
            output = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=1.0,
            ).stdout
            row = next(csv.reader(output.splitlines(), skipinitialspace=True))
            gpu = {
                "name": row[0],
                "utilization_percent": float(row[1]),
                "memory_used_mb": float(row[2]),
                "memory_total_mb": float(row[3]),
                "temperature_celsius": float(row[4]),
                "source": "nvidia-smi",
            }
        except (OSError, subprocess.SubprocessError, ValueError, IndexError, StopIteration):
            pass

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "live_host_telemetry",
            "host_cpu_percent": psutil.cpu_percent(interval=None),
            "host_memory_total_mb": memory.total / 1_000_000,
            "host_memory_available_mb": memory.available / 1_000_000,
            "host_memory_used_percent": memory.percent,
            "evaluator_process": process_metrics,
            "temperatures": temperatures,
            "gpu": gpu,
            "notes": {
                "controller_inputs_are_live": False,
                "controller_input_source": "synthetic_trace",
                "temperature_unavailable_reason": "No supported host or NVIDIA temperature sensor was detected."
                if not temperatures and not (gpu and gpu.get("temperature_celsius") is not None)
                else None,
            },
        }

    def list_experiments(self) -> list[dict[str, Any]]:
        if not self.results_root.is_dir():
            return []
        records = []
        for path in self.results_root.rglob("*.json"):
            if path.name.endswith(".tmp") or path.stat().st_size > 100_000_000:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            summary = self._normalize_experiment(path, payload)
            if summary is not None:
                records.append(summary)
        return sorted(records, key=lambda row: row["created_at"], reverse=True)

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        path = self._resolve_result_path(experiment_id)
        if path is None or not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        summary = self._normalize_experiment(path, payload)
        if summary is None:
            return None
        return {**summary, "samples": payload.get("samples", [])}

    def list_profiles(self) -> list[dict[str, Any]]:
        profile_root = self.project_root / "profiles"
        profiles: list[dict[str, Any]] = []
        for path in sorted(profile_root.glob("*_profile.csv")):
            try:
                with path.open(newline="", encoding="utf-8-sig") as handle:
                    rows = list(csv.DictReader(handle))
            except OSError:
                continue
            profiles.append({"id": path.name, "rows": rows})
        return profiles

    def _normalize_experiment(self, path: Path, payload: Any) -> dict[str, Any] | None:
        if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
            return None
        if not any(key in payload for key in (
            "average_latency_ms", "accuracy_percent", "total_samples", "summary"
        )):
            return None
        relative_id = path.resolve().relative_to(self.results_root).as_posix()
        stat = path.stat()
        return {
            "id": relative_id,
            "name": path.stem,
            "created_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "execution_mode": payload.get(
                "execution_mode", "measured_inference_with_simulated_resource_trace"
            ),
            "dataset": "CIFAR-10",
            "architecture": "ResNet-50",
            "device": payload.get("device"),
            "trace": payload.get("trace"),
            "latency_policy": payload.get("latency_policy"),
            "configured_deadline_ms": payload.get("configured_deadline_ms"),
            "deadline_mode": payload.get("deadline_mode"),
            "thread_count": payload.get("thread_count"),
            "total_samples": payload.get("total_samples", len(payload["samples"])),
            "accuracy_percent": payload.get("accuracy_percent"),
            "average_latency_ms": payload.get("average_latency_ms"),
            "p95_latency_ms": payload.get("p95_latency_ms"),
            "p99_latency_ms": payload.get("p99_latency_ms"),
            "deadline_miss_rate_percent": payload.get("deadline_miss_rate_percent"),
            "simulation_summary": payload.get("summary"),
            "scenario": payload.get("scenario"),
            "provenance_note": payload.get("provenance_note"),
            "config_counts": payload.get("config_counts", {}),
            "hardware_fingerprint": payload.get("hardware_fingerprint"),
            "source": "existing_evaluation_artifact",
            "measurement_note": payload.get(
                "provenance_note",
                "Latency is measured inference time. Resource states and deadlines in this evaluator are synthetic trace inputs.",
            ),
        }

    def _resolve_result_path(self, experiment_id: str) -> Path | None:
        candidate = (self.results_root / experiment_id).resolve()
        if not candidate.is_relative_to(self.results_root) or candidate.suffix.lower() != ".json":
            return None
        return candidate