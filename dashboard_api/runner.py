from __future__ import annotations

import asyncio
import json
import re
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dashboard_api.service import DashboardData, PROJECT_ROOT


PROGRESS_PATTERN = re.compile(r"Sample\s+(\d+)/(\d+)")
DASHBOARD_EVENT_PREFIX = "DASHBOARD_EVENT:"


@dataclass
class ExperimentJob:
    run_id: str
    result_path: Path
    status: str = "starting"
    progress: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    return_code: int | None = None
    error: str | None = None
    logs: list[str] = field(default_factory=list)
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    process: asyncio.subprocess.Process | None = None
    phase: str = "starting"
    completed_samples: int = 0
    total_samples: int = 0
    calibration: dict[str, Any] | None = None
    current_decision: dict[str, Any] | None = None
    latest_sample: dict[str, Any] | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at,
            "return_code": self.return_code,
            "error": self.error,
            "result_id": self.result_path.relative_to(PROJECT_ROOT / "results").as_posix(),
            "execution_mode": "measured_inference_with_simulated_resource_trace",
            "phase": self.phase,
            "completed_samples": self.completed_samples,
            "total_samples": self.total_samples,
            "calibration": self.calibration,
            "current_decision": self.current_decision,
            "latest_sample": self.latest_sample,
        }


class ExperimentRunner:
    def __init__(self, data: DashboardData | None = None):
        self.data = data or DashboardData()
        self.jobs: dict[str, ExperimentJob] = {}
        self.active_run_id: str | None = None
        self._lock = asyncio.Lock()

    async def start(self, options: dict[str, Any]) -> ExperimentJob:
        async with self._lock:
            if self.active_run_id is not None:
                active = self.jobs[self.active_run_id]
                if active.status in {"starting", "running", "stopping"}:
                    raise RuntimeError(f"Experiment {active.run_id} is already running")

            readiness = self.data.system_info()["real_inference_preflight"]
            if not readiness["ready"]:
                missing = [key for key, available in readiness["requirements"].items() if not available]
                raise RuntimeError("Real inference prerequisites missing: " + ", ".join(missing))
            if options["device"] == "cuda" and not self.data.system_info()["cuda_available"]:
                raise RuntimeError("CUDA was requested but is unavailable in this Python environment")

            run_id = uuid.uuid4().hex
            result_path = PROJECT_ROOT / "results" / "dashboard_runs" / f"{run_id}.json"
            job = ExperimentJob(run_id=run_id, result_path=result_path)
            self.jobs[run_id] = job
            self.active_run_id = run_id
            command = [
                sys.executable,
                "-u",
                str(PROJECT_ROOT / "scripts" / "evaluate.py"),
                "--policy", options["latency_policy"],
                "--trace", options["trace"],
                "--seed", str(options["seed"]),
                "--num-samples", str(options["num_samples"]),
                "--deadline-ms", str(options["deadline_ms"]),
                "--threads", str(options["threads"]),
                "--device", options["device"],
                "--output", str(result_path),
                "--dashboard-events",
            ]
            job.process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(PROJECT_ROOT),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            job.status = "running"
            asyncio.create_task(self._collect(job))
            await self._publish(job, {"event": "status", **job.snapshot()})
            return job

    async def _collect(self, job: ExperimentJob) -> None:
        assert job.process is not None and job.process.stdout is not None
        try:
            while line := await job.process.stdout.readline():
                message = line.decode("utf-8", errors="replace").rstrip()
                if not message:
                    continue
                if message.startswith(DASHBOARD_EVENT_PREFIX):
                    payload = json.loads(message[len(DASHBOARD_EVENT_PREFIX):])
                    self._apply_runtime_event(job, payload)
                    await self._publish(job, {"event": "runtime", **payload, "progress": job.progress})
                    continue
                job.logs.append(message)
                job.logs = job.logs[-200:]
                match = PROGRESS_PATTERN.search(message)
                if match:
                    current, total = (int(value) for value in match.groups())
                    job.progress = min(1.0, current / max(1, total))
                await self._publish(job, {"event": "log", "message": message, "progress": job.progress})
            job.return_code = await job.process.wait()
            if job.status == "stopping":
                job.status = "stopped"
            elif job.return_code == 0 and job.result_path.is_file():
                job.status = "completed"
                job.progress = 1.0
                job.phase = "completed"
            else:
                job.status = "failed"
                job.phase = "failed"
                job.error = f"Evaluator exited with code {job.return_code}"
            await self._publish(job, {"event": "status", **job.snapshot()})
        except Exception as exc:
            job.status = "failed"
            job.phase = "failed"
            job.error = str(exc)
            await self._publish(job, {"event": "status", **job.snapshot()})
        finally:
            if self.active_run_id == job.run_id:
                self.active_run_id = None

    async def stop(self, run_id: str) -> ExperimentJob | None:
        job = self.jobs.get(run_id)
        if job is None:
            return None
        if job.process is not None and job.process.returncode is None:
            job.status = "stopping"
            job.process.terminate()
            await self._publish(job, {"event": "status", **job.snapshot()})
        return job

    async def _publish(self, job: ExperimentJob, event: dict[str, Any]) -> None:
        for subscriber in list(job.subscribers):
            await subscriber.put(event)

    @staticmethod
    def _apply_runtime_event(job: ExperimentJob, payload: dict[str, Any]) -> None:
        event_type = payload.get("type")
        if payload.get("phase"):
            job.phase = str(payload["phase"])
        if payload.get("total_samples") is not None:
            job.total_samples = int(payload["total_samples"])
        if event_type in {"calibration_started", "calibration_progress", "calibration_completed"}:
            job.phase = "batchnorm_calibration"
            job.calibration = {
                "sample_index": payload.get("sample_index"),
                "selected_config": payload.get("selected_config"),
                "completed_batches": payload.get("completed_batches", 0),
                "total_batches": payload.get("total_batches", 0),
                "status": "completed" if event_type == "calibration_completed" else "running",
            }
        elif event_type == "decision":
            job.phase = "configuration_selection"
            job.current_decision = payload
        elif event_type == "sample_completed":
            job.phase = "inference"
            job.completed_samples = int(payload.get("completed_samples", job.completed_samples))
            job.total_samples = int(payload.get("total_samples", job.total_samples))
            job.progress = min(1.0, job.completed_samples / max(1, job.total_samples))
            job.latest_sample = payload.get("sample")
            job.current_decision = payload


runner = ExperimentRunner()