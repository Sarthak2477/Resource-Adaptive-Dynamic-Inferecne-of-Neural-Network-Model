from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from resource_control.qat_experiment import (
    MeasuredQATController,
    validate_accuracy_table,
    validate_latency_table,
)


QAT_RUNS_ROOT = Path(__file__).resolve().parents[1] / "results" / "usm_qat_resource_adaptive" / "runs"
SCENARIOS = ("stable", "workload_spike", "resource_reduction", "repeated_spikes")


class SimulationEngine:
    def __init__(self, runs_root: Path = QAT_RUNS_ROOT, results_root: Path | None = None):
        self.runs_root = Path(runs_root)
        self.results_root = Path(results_root or self.runs_root.parents[1] / "dashboard_simulations")

    def run(self, scenario: Literal["stable", "workload_spike", "resource_reduction", "repeated_spikes"],
            num_samples: int, deadline_ms: float) -> dict[str, Any]:
        if scenario not in SCENARIOS:
            raise ValueError(f"Unsupported simulation scenario: {scenario}")
        if not 1 <= num_samples <= 5000 or deadline_ms <= 0:
            raise ValueError("num_samples must be 1..5000 and deadline_ms must be positive")

        source, accuracies, profiles, switch_costs = self._load_calibration()
        controller = MeasuredQATController(accuracies, profiles, switch_costs)
        samples = []
        previous_config = None
        for index in range(num_samples):
            condition = self._condition_for(scenario, index, num_samples)
            decision = controller.select(condition, deadline_ms)
            chosen = decision.config
            sample = {
                "sample_index": index,
                "resource_condition": condition,
                "synthetic_resource_condition": True,
                "deadline_ms": float(deadline_ms),
                "selected_config": [float(chosen[0]), int(chosen[1])],
                "previous_config": list(previous_config) if previous_config else None,
                "configuration_changed": previous_config is not None and previous_config != chosen,
                "selection_reason": self._reason(decision),
                "predicted_p95_ms": float(decision.predicted_p95_ms),
                "predicted_switch_cost_ms": float(decision.predicted_switch_ms),
                "adjusted_predicted_p95_ms": float(decision.predicted_p95_ms + decision.predicted_switch_ms),
                "deadline_met_by_prediction": bool(decision.predicted_p95_ms + decision.predicted_switch_ms <= deadline_ms),
                "candidate_count_feasible": int(decision.feasible_candidate_count),
                "fp32_feasible_candidate_count": int(decision.fp32_feasible_candidate_count),
                "precision_fallback_used": bool(decision.precision_fallback_used),
                "estimated_validation_accuracy_percent": float(accuracies[chosen]["accuracy_percent"]),
                "latency_is_simulated_prediction": True,
            }
            samples.append(sample)
            previous_config = chosen

        changed_count = sum(row["configuration_changed"] for row in samples)
        predicted_p95 = [row["predicted_p95_ms"] for row in samples]
        weighted_accuracy = sum(row["estimated_validation_accuracy_percent"] for row in samples) / num_samples
        violations = sum(not row["deadline_met_by_prediction"] for row in samples)
        run_id = uuid.uuid4().hex
        self.results_root.mkdir(parents=True, exist_ok=True)
        result_path = self.results_root / f"{run_id}.json"
        metadata = self._source_metadata(source)
        result = {
            "schema_version": 1,
            "id": result_path.relative_to(self.results_root.parent).as_posix(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "execution_mode": "simulation",
            "scenario": scenario,
            "deadline_ms": float(deadline_ms),
            "sample_count": num_samples,
            "source_profile_run": str(source.relative_to(self.runs_root)),
            "source_profile_metadata": metadata,
            "summary": {
                "average_predicted_p95_ms": sum(predicted_p95) / num_samples,
                "estimated_selected_accuracy_percent": weighted_accuracy,
                "predicted_deadline_violation_rate_percent": 100.0 * violations / num_samples,
                "configuration_switch_count": changed_count,
                "precision_fallback_count": sum(row["precision_fallback_used"] for row in samples),
            },
            "provenance_note": "Synthetic request/resource sequence; selection uses the existing MeasuredQATController. Latency values are predictions from stored measured P95 profiles, not live inference measurements.",
            "samples": samples,
        }
        result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return result

    def _load_calibration(self):
        candidates = []
        if self.runs_root.is_dir():
            for run in self.runs_root.iterdir():
                accuracy_path = run / "accuracy" / "candidate_accuracy.csv"
                profile_path = run / "profiles" / "resource_condition_profiles.csv"
                switch_path = run / "profiles" / "switch_costs.csv"
                if accuracy_path.is_file() and profile_path.is_file() and switch_path.is_file():
                    candidates.append((run.stat().st_mtime, run, accuracy_path, profile_path, switch_path))
        if not candidates:
            raise RuntimeError("No complete saved QAT calibration run is available for simulation")
        _, source, accuracy_path, profile_path, switch_path = max(candidates)
        with accuracy_path.open(newline="", encoding="utf-8-sig") as handle:
            accuracies = validate_accuracy_table(list(csv.DictReader(handle)))
        with profile_path.open(newline="", encoding="utf-8-sig") as handle:
            profile_rows = list(csv.DictReader(handle))
        conditions = tuple(sorted({row["resource_condition"] for row in profile_rows}))
        profiles = validate_latency_table(profile_rows, conditions)
        with switch_path.open(newline="", encoding="utf-8-sig") as handle:
            switch_rows = list(csv.DictReader(handle))
        switch_costs = {row["switch_type"]: float(row["p95_ms"]) for row in switch_rows}
        return source, accuracies, profiles, switch_costs

    def _source_metadata(self, source: Path) -> dict[str, Any]:
        path = source / "profiles" / "resource_condition_profiles.csv"
        with path.open(newline="", encoding="utf-8-sig") as handle:
            row = next(csv.DictReader(handle), {})
        return {
            "device": row.get("device"),
            "inference_backend": row.get("inference_backend"),
            "checkpoint_sha256": row.get("checkpoint_sha256"),
            "conditions": sorted({item["resource_condition"] for item in self._profile_rows(path)}),
        }

    @staticmethod
    def _profile_rows(path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _condition_for(scenario: str, index: int, total: int) -> str:
        if scenario == "stable":
            return "baseline"
        if scenario == "workload_spike":
            return "baseline" if index < total // 2 else "cpu_contention"
        if scenario == "resource_reduction":
            return "baseline" if index < max(1, total // 5) else "cpu_contention"
        return "cpu_contention" if index % 20 >= 10 else "baseline"

    @staticmethod
    def _reason(decision) -> str:
        if decision.fallback:
            return "No candidate met the safety-adjusted deadline; chose the minimum adjusted P95 fallback."
        if decision.precision_fallback_used:
            return "No FP32 candidate was feasible; selected the best feasible lower-precision candidate."
        return "Selected the highest-accuracy feasible FP32 candidate."