from __future__ import annotations

import asyncio
import csv
import io
import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dashboard_api.service import DashboardData, PROJECT_ROOT
from dashboard_api.runner import runner
from dashboard_api.simulation import SimulationEngine


app = FastAPI(title="Resource-Adaptive Inference Research API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5174", "http://127.0.0.1:5174", "http://localhost:8080"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
data = DashboardData()
simulation_engine = SimulationEngine()


class ExperimentRequest(BaseModel):
    num_samples: int = Field(default=100, ge=1, le=10000)
    deadline_ms: float = Field(default=30.0, gt=0, le=1000)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    trace: Literal["sinusoidal", "step", "bursty", "heldout"] = "sinusoidal"
    latency_policy: Literal["avg", "p50", "p95", "p99"] = "p95"
    seed: int = Field(default=12345, ge=0)
    threads: int = Field(default=1, ge=1, le=256)


class SimulationRequest(BaseModel):
    scenario: Literal["stable", "workload_spike", "resource_reduction", "repeated_spikes"] = "stable"
    num_samples: int = Field(default=100, ge=1, le=5000)
    deadline_ms: float = Field(default=30.0, gt=0, le=1000)


@app.get("/api/system/info")
def system_info():
    return data.system_info()


@app.get("/api/system/resources")
def system_resources():
    active_job = runner.jobs.get(runner.active_run_id) if runner.active_run_id else None
    process = active_job.process if active_job else None
    evaluator_pid = process.pid if process and process.returncode is None else None
    return data.system_resources(evaluator_pid)


@app.get("/api/model/configurations")
def model_configurations():
    return data.model_configurations()


@app.get("/api/profiles")
def profiles():
    return data.list_profiles()


@app.get("/api/experiments")
def experiments():
    return data.list_experiments()


@app.post("/api/simulations", status_code=201)
def run_simulation(request: SimulationRequest):
    try:
        return simulation_engine.run(request.scenario, request.num_samples, request.deadline_ms)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/experiments", status_code=202)
async def start_experiment(request: ExperimentRequest):
    try:
        job = await runner.start(request.dict())
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return job.snapshot()


@app.get("/api/runs/{run_id}")
def run_status(run_id: str):
    job = runner.jobs.get(run_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return job.snapshot()


@app.post("/api/runs/{run_id}/stop")
async def stop_run(run_id: str):
    job = await runner.stop(run_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return job.snapshot()


@app.websocket("/ws/experiments/{run_id}")
async def experiment_stream(websocket: WebSocket, run_id: str):
    job = runner.jobs.get(run_id)
    if job is None:
        await websocket.close(code=1008, reason="Run not found")
        return
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue()
    job.subscribers.append(queue)
    try:
        await websocket.send_json({"event": "status", **job.snapshot()})
        for message in job.logs[-20:]:
            await websocket.send_json({"event": "log", "message": message, "progress": job.progress})
        while job.status in {"starting", "running", "stopping"}:
            await websocket.send_json(await queue.get())
        await websocket.send_json({"event": "status", **job.snapshot()})
    except WebSocketDisconnect:
        pass
    finally:
        if queue in job.subscribers:
            job.subscribers.remove(queue)


@app.get("/api/experiments/{experiment_id:path}/events")
def experiment_events(experiment_id: str):
    result = data.get_experiment(experiment_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Experiment artifact not found or unsupported")
    return {
        "experiment_id": experiment_id,
        "source": result["source"],
        "events": [
            {"event": "sample_recorded", **sample}
            for sample in result["samples"]
            if isinstance(sample, dict)
        ],
    }


@app.get("/api/experiments/{experiment_id:path}/export")
def export_experiment(experiment_id: str, format: Literal["json", "csv"] = Query(default="json")):
    result = data.get_experiment(experiment_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Experiment artifact not found or unsupported")
    if format == "json":
        body = json.dumps(result, indent=2, allow_nan=False).encode("utf-8")
        return Response(body, media_type="application/json", headers={
            "Content-Disposition": f'attachment; filename="{Path(experiment_id).stem}.json"'
        })
    output = io.StringIO(newline="")
    samples = result["samples"]
    columns = sorted({key for sample in samples if isinstance(sample, dict) for key in sample})
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(sample for sample in samples if isinstance(sample, dict))
    return Response(output.getvalue(), media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="{Path(experiment_id).stem}.csv"'
    })


@app.get("/api/experiments/{experiment_id:path}")
def experiment(experiment_id: str):
    result = data.get_experiment(experiment_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Experiment artifact not found or unsupported")
    return result


app.mount("/", StaticFiles(directory=PROJECT_ROOT / "dashboard", html=True), name="dashboard")