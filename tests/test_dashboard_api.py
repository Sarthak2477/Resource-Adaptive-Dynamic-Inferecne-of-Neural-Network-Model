import asyncio
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from dashboard_api.main import ExperimentRequest, app
from dashboard_api.runner import ExperimentJob, ExperimentRunner, PROJECT_ROOT, runner
from dashboard_api.service import DashboardData
from dashboard_api.simulation import SimulationEngine
from scripts.evaluate import resolve_latency_budget


class DashboardDataTests(unittest.TestCase):
    def test_real_experiment_request_accepts_and_defaults_deadline(self):
        configured = ExperimentRequest(num_samples=2, deadline_ms=12.5)
        defaulted = ExperimentRequest(num_samples=2)

        self.assertEqual(configured.deadline_ms, 12.5)
        self.assertEqual(defaulted.deadline_ms, 30.0)

    def test_configured_deadline_is_the_effective_controller_budget(self):
        self.assertEqual(resolve_latency_budget(42.0, 15.0), 15.0)
        self.assertEqual(resolve_latency_budget(8.0, 15.0), 15.0)
        self.assertEqual(resolve_latency_budget(2.0, None), 4.0)
        self.assertEqual(resolve_latency_budget(42.0, None), 42.0)
        with self.assertRaises(ValueError):
            resolve_latency_budget(42.0, 0.0)

    def test_configuration_discovery_matches_model_config(self):
        service = DashboardData(Path(__file__).resolve().parents[1])

        configurations = service.model_configurations()["configurations"]

        self.assertEqual(len(configurations), 16)
        self.assertIn({"width": 1.0, "bit_width": 32, "id": "w1-b32"}, configurations)
        self.assertIn({"width": 0.25, "bit_width": 4, "id": "w0.25-b4"}, configurations)

    def test_experiment_normalization_preserves_measurement_provenance(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            results = root / "results"
            results.mkdir()
            (results / "evaluation_sample.json").write_text(json.dumps({
                "trace": "step",
                "device": "cpu",
                "configured_deadline_ms": 18.0,
                "deadline_mode": "fixed",
                "total_samples": 1,
                "accuracy_percent": 100.0,
                "average_latency_ms": 4.5,
                "samples": [{
                    "sample_index": 0,
                    "actual_latency_ms": 4.5,
                    "budget_ms": 20.0,
                    "missed_deadline": False,
                }],
            }), encoding="utf-8")

            service = DashboardData(root)
            record = service.list_experiments()[0]
            detail = service.get_experiment(record["id"])

        self.assertEqual(record["execution_mode"], "measured_inference_with_simulated_resource_trace")
        self.assertEqual(record["configured_deadline_ms"], 18.0)
        self.assertEqual(record["deadline_mode"], "fixed")
        self.assertEqual(record["average_latency_ms"], 4.5)
        self.assertEqual(detail["samples"][0]["actual_latency_ms"], 4.5)

    def test_result_path_cannot_escape_results_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            service = DashboardData(Path(temporary_directory))

            self.assertIsNone(service.get_experiment("../outside.json"))

    def test_live_resource_endpoint_marks_controller_input_provenance(self):
        response = TestClient(app).get("/api/system/resources")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["source"], "live_host_telemetry")
        self.assertGreaterEqual(payload["host_memory_total_mb"], payload["host_memory_available_mb"])
        self.assertFalse(payload["notes"]["controller_inputs_are_live"])

    def test_create_experiment_forwards_configured_deadline_to_runner(self):
        response_snapshot = {"run_id": "deadline-test", "status": "running"}
        with patch.object(runner, "start", new_callable=AsyncMock) as start:
            start.return_value = SimpleNamespace(snapshot=lambda: response_snapshot)
            response = TestClient(app).post("/api/experiments", json={"num_samples": 2, "deadline_ms": 12.5})

        self.assertEqual(response.status_code, 202)
        self.assertEqual(start.await_args.args[0]["deadline_ms"], 12.5)

    def test_simulation_uses_saved_measured_profiles_and_marks_estimates(self):
        project_root = Path(__file__).resolve().parents[1]
        output_root = project_root / "results" / "dashboard_simulations_test"
        shutil.rmtree(output_root, ignore_errors=True)
        engine = SimulationEngine(
            project_root / "results" / "usm_qat_resource_adaptive" / "runs",
            output_root,
        )

        try:
            result = engine.run("workload_spike", num_samples=12, deadline_ms=30.0)

            self.assertEqual(result["execution_mode"], "simulation")
            self.assertEqual(len(result["samples"]), 12)
            self.assertTrue(all(row["latency_is_simulated_prediction"] for row in result["samples"]))
            self.assertEqual(result["samples"][0]["resource_condition"], "baseline")
            self.assertEqual(result["samples"][-1]["resource_condition"], "cpu_contention")
        finally:
            shutil.rmtree(output_root, ignore_errors=True)


class ExperimentRunnerTests(unittest.IsolatedAsyncioTestCase):
    def test_batchnorm_recalibration_callback_reports_each_batch(self):
        import torch

        from models.train import recalibrate_bn

        class TinyModel(torch.nn.Module):
            def set_width(self, _width):
                pass

            def set_bit_width(self, _bits):
                pass

            def forward(self, images):
                return images

        updates = []
        loader = [
            (torch.zeros(1, 3, 2, 2), torch.zeros(1, dtype=torch.long))
            for _ in range(3)
        ]
        recalibrate_bn(
            TinyModel(),
            loader,
            0.25,
            32,
            torch.device("cpu"),
            num_batches=2,
            progress_callback=lambda completed, total: updates.append((completed, total)),
        )

        self.assertEqual(updates, [(1, 2), (2, 2)])

    def test_runtime_events_update_calibration_and_sample_progress(self):
        job = ExperimentJob(
            run_id="runtime-events",
            result_path=PROJECT_ROOT / "results" / "dashboard_runs" / "runtime-events.json",
        )

        ExperimentRunner._apply_runtime_event(job, {
            "type": "calibration_progress",
            "sample_index": 0,
            "selected_config": [0.25, 32],
            "completed_batches": 4,
            "total_batches": 10,
        })
        self.assertEqual(job.phase, "batchnorm_calibration")
        self.assertEqual(job.calibration["completed_batches"], 4)

        sample = {"sample_index": 0, "actual_latency_ms": 12.5}
        ExperimentRunner._apply_runtime_event(job, {
            "type": "sample_completed",
            "completed_samples": 1,
            "total_samples": 3,
            "sample": sample,
        })
        self.assertEqual(job.completed_samples, 1)
        self.assertEqual(job.progress, 1 / 3)
        self.assertEqual(job.latest_sample, sample)

    async def test_collect_broadcasts_structured_evaluator_event(self):
        from dashboard_api.runner import DASHBOARD_EVENT_PREFIX

        event = {
            "type": "sample_completed",
            "completed_samples": 1,
            "total_samples": 2,
            "sample": {"sample_index": 0, "actual_latency_ms": 8.0},
        }

        class EventOutput:
            def __init__(self):
                self.lines = [
                    (DASHBOARD_EVENT_PREFIX + json.dumps(event) + "\n").encode(),
                    b"",
                ]

            async def readline(self):
                return self.lines.pop(0)

        class EventProcess:
            def __init__(self):
                self.stdout = EventOutput()
                self.returncode = None

            async def wait(self):
                self.returncode = 0
                return 0

        runner = ExperimentRunner()
        job = ExperimentJob(
            run_id="structured-event",
            result_path=PROJECT_ROOT / "results" / "dashboard_runs" / "structured-event.json",
        )
        job.process = EventProcess()
        subscriber = asyncio.Queue()
        job.subscribers.append(subscriber)

        await runner._collect(job)

        events = []
        while not subscriber.empty():
            events.append(await subscriber.get())
        runtime = next(item for item in events if item["event"] == "runtime")
        self.assertEqual(runtime["type"], "sample_completed")
        self.assertEqual(job.completed_samples, 1)
        self.assertEqual(job.progress, 0.5)

    async def test_failed_evaluator_publishes_failed_status(self):
        class FailedOutput:
            def __init__(self):
                self.lines = [b"RuntimeError: controlled test failure\n", b""]

            async def readline(self):
                return self.lines.pop(0)

        class FailedProcess:
            def __init__(self):
                self.stdout = FailedOutput()
                self.returncode = None

            async def wait(self):
                self.returncode = 17
                return self.returncode

        runner = ExperimentRunner()
        job = ExperimentJob(
            run_id="controlled-failure",
            result_path=PROJECT_ROOT / "results" / "dashboard_runs" / "controlled-failure.json",
        )
        job.process = FailedProcess()
        events = asyncio.Queue()
        job.subscribers.append(events)

        await runner._collect(job)

        published = []
        while not events.empty():
            published.append(await events.get())
        self.assertEqual(job.status, "failed")
        self.assertEqual(job.error, "Evaluator exited with code 17")
        self.assertTrue(any("controlled test failure" in line for line in job.logs))
        self.assertEqual(published[-1]["status"], "failed")


if __name__ == "__main__":
    unittest.main()