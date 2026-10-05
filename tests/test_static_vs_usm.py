import os
import sys
import unittest
import json
import csv
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from scripts.experiment_static_vs_usm import get_fp32_frontier, run_arm
from scripts.run_static_vs_usm import write_run_manifests
from models import Model, set_model_width, set_model_bit_width
from models.checkpoint_io import load_fixed_width_checkpoint, resolve_usm_checkpoint
from models.checkpoint_io import checkpoint_sha256
from scripts.calibrate_static_width import calibrate_static_width
from scripts.experiment_data import class_interleaved_indices

class DummyController:
    def __init__(self):
        self.last_predicted_latency = 10.0
    def select(self, state, latency_budget_ms, hw_fingerprint):
        return (0.5, 32)
    def update_feedback(self, lat, pred, config):
        pass

class TestStaticVsUSM(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = Model(num_classes=10, input_size=32).to(self.device)
        self.requests = [
            {"request_id": f"req_{i:04d}", "sample_index": i, "budget_ms": 20.0}
            for i in range(10)
        ]
        self.dummy_data = [(torch.randn(1, 3, 32, 32), torch.tensor([0])) for _ in range(10)]
        self.hw_fingerprint = {}
        
    def test_usm_pinned_arm_never_changes_width(self):
        results = run_arm("usm_pinned", self.model, self.dummy_data, self.requests, 0.75, None, self.device, "test_run", self.hw_fingerprint, warmup_requests=1)
        for r in results:
            self.assertEqual(r["selected_width"], 0.75)
            self.assertFalse(r["switched"])
            
    def test_usm_pinned_arm_has_no_controller(self):
        results = run_arm("usm_pinned", self.model, self.dummy_data, self.requests, 0.75, None, self.device, "test_run", self.hw_fingerprint, warmup_requests=1)
        self.assertEqual(len(results), 10)

    def test_usm_pinned_rejects_controller(self):
        with self.assertRaises(ValueError):
            run_arm("usm_pinned", self.model, self.dummy_data, self.requests, 0.75, DummyController(), self.device, "test_run", self.hw_fingerprint, warmup_requests=1)

    def test_frontier_fallback_order_uses_selected_percentile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            profile_path = temp_path / "profile.csv"
            accuracy_path = temp_path / "accuracy.csv"
            profile_rows = [
                {"width_mult": width, "bit_width": 32, "latency_ms": avg,
                 "std_ms": 0.5, "p50_ms": p50, "p95_ms": p95, "p99_ms": p99}
                for width, avg, p50, p95, p99 in (
                    (0.25, 5.0, 8.0, 20.0, 30.0),
                    (0.5, 10.0, 7.0, 12.0, 25.0),
                    (0.75, 15.0, 6.0, 10.0, 20.0),
                    (1.0, 20.0, 5.0, 8.0, 15.0),
                )
            ]
            accuracy_rows = [
                {"width_mult": width, "bit_width": 32, "accuracy_percent": 90.0 + width}
                for width in (0.25, 0.5, 0.75, 1.0)
            ]
            for path, rows in ((profile_path, profile_rows), (accuracy_path, accuracy_rows)):
                with path.open("w", newline="", encoding="utf-8") as output_file:
                    writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)

            frontier = get_fp32_frontier(profile_path, accuracy_path, latency_policy="p95")

            self.assertEqual([row["config"][0] for row in frontier], [1.0, 0.75, 0.5, 0.25])
        
    def test_usm_fp32_only(self):
        controller = DummyController()
        results = run_arm("usm_adaptive", self.model, self.dummy_data, self.requests, 0.75, controller, self.device, "test_run", self.hw_fingerprint, warmup_requests=1)
        self.assertEqual({row["selected_width"] for row in results}, {0.5})
        self.assertEqual({row["selected_bit_width"] for row in results}, {32})
        self.assertTrue(all(row["model_checkpoint_sha256"] is None for row in results))
        
    def test_no_fake_quantization(self):
        set_model_bit_width(self.model, 32)
        with patch("models.ops.fake_quantize_weight") as weight_quant, patch("models.ops.fake_quantize_act") as act_quant:
            out = self.model(torch.randn(1, 3, 32, 32))
        weight_quant.assert_not_called()
        act_quant.assert_not_called()
        self.assertEqual(out.shape, (1, 10))
        
    def test_workload_identity_across_arms(self):
        res_static = run_arm("usm_pinned", self.model, self.dummy_data, self.requests, 0.75, None, self.device, "test_run", self.hw_fingerprint, warmup_requests=1)
        controller = DummyController()
        res_usm = run_arm("usm_adaptive", self.model, self.dummy_data, self.requests, 0.75, controller, self.device, "test_run", self.hw_fingerprint, warmup_requests=1)
        
        req_static = [r["request_id"] for r in res_static]
        req_usm = [r["request_id"] for r in res_usm]
        self.assertEqual(req_static, req_usm)

    def test_accuracy_calibration_and_replay_indices_are_disjoint(self):
        targets = [0, 1, 0, 1, 0, 1]
        calibration = class_interleaved_indices(targets, 2, stop_offset=1)
        replay = class_interleaved_indices(targets, 2, start_offset=1)

        self.assertEqual(len(calibration), 2)
        self.assertEqual(len(replay), 4)
        self.assertEqual(sorted(calibration + replay), list(range(len(targets))))
        self.assertFalse(set(calibration).intersection(replay))
        
    def test_result_joinable(self):
        res_static = run_arm("usm_pinned", self.model, self.dummy_data, self.requests, 0.75, None, self.device, "test_run", self.hw_fingerprint, warmup_requests=1)
        controller = DummyController()
        res_usm = run_arm("usm_adaptive", self.model, self.dummy_data, self.requests, 0.75, controller, self.device, "test_run", self.hw_fingerprint, warmup_requests=1)
        
        df_static = pd.DataFrame(res_static)
        df_usm = pd.DataFrame(res_usm)
        df_joined = pd.merge(df_static, df_usm, on="request_id")
        self.assertEqual(len(df_joined), len(self.requests))

    def test_missing_usm_checkpoint_fails_instead_of_using_random_weights(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing_checkpoint.pt"
            with self.assertRaises(FileNotFoundError):
                resolve_usm_checkpoint(missing_path)

    def test_fixed_checkpoint_requires_separate_fp32_training_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "fixed_width.pt"
            source_model = nn.Linear(3, 2)
            torch.save({
                "model_state_dict": source_model.state_dict(),
                "training_mode": "separately_trained_fixed_width_fp32",
                "width_mult": 0.5,
                "bit_width": 32,
            }, checkpoint_path)

            target_model = nn.Linear(3, 2)
            provenance = load_fixed_width_checkpoint(
                target_model, checkpoint_path, torch.device("cpu"), 0.5
            )

            self.assertEqual(provenance["checkpoint_metadata"]["width_mult"], 0.5)
            self.assertEqual(len(provenance["checkpoint_sha256"]), 64)

    def test_fixed_checkpoint_rejects_usm_or_wrong_width_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "not_fixed.pt"
            source_model = nn.Linear(3, 2)
            torch.save({
                "model_state_dict": source_model.state_dict(),
                "training_mode": "universally_slimmable",
                "width_mult": 0.5,
                "bit_width": 32,
            }, checkpoint_path)

            with self.assertRaises(ValueError):
                load_fixed_width_checkpoint(
                    nn.Linear(3, 2), checkpoint_path, torch.device("cpu"), 0.5
                )

    def test_static_width_calibration_requires_matching_profile_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_path = temp_path / "usm.pt"
            checkpoint_path.write_bytes(b"trained-checkpoint-placeholder")
            profile_path = temp_path / "profile.csv"
            output_path = temp_path / "config.json"
            checkpoint_hash = checkpoint_sha256(checkpoint_path)
            rows = [
                {"width_mult": width, "bit_width": 32, "p95_ms": latency,
                 "checkpoint_sha256": checkpoint_hash}
                for width, latency in ((0.25, 5.0), (0.5, 10.0), (0.75, 15.0), (1.0, 25.0))
            ]
            with profile_path.open("w", newline="", encoding="utf-8") as profile_file:
                writer = csv.DictWriter(profile_file, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            calibrate_static_width(profile_path, checkpoint_path, output_path, 20.0)

            with output_path.open("r", encoding="utf-8") as config_file:
                config = json.load(config_file)
            self.assertEqual(config["pinned_width"], 0.75)
            self.assertEqual(config["usm_checkpoint_sha256"], checkpoint_hash)

    def test_static_width_calibration_falls_back_when_target_is_infeasible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_path = temp_path / "usm.pt"
            checkpoint_path.write_bytes(b"trained-checkpoint-placeholder")
            profile_path = temp_path / "profile.csv"
            output_path = temp_path / "config.json"
            checkpoint_hash = checkpoint_sha256(checkpoint_path)
            rows = [
                {"width_mult": width, "bit_width": 32, "p95_ms": latency,
                 "checkpoint_sha256": checkpoint_hash}
                for width, latency in ((0.25, 12.58), (0.5, 19.98), (0.75, 13.67), (1.0, 10.44))
            ]
            with profile_path.open("w", newline="", encoding="utf-8") as profile_file:
                writer = csv.DictWriter(profile_file, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            calibrate_static_width(profile_path, checkpoint_path, output_path, 10.0)

            with output_path.open("r", encoding="utf-8") as config_file:
                config = json.load(config_file)
            self.assertEqual(config["pinned_width"], 1.0)
            self.assertFalse(config["calibration_target_feasible"])
            self.assertEqual(config["minimum_profile_p95_ms"], 10.44)
            self.assertEqual(
                config["selection_rule"],
                "lowest_profile_p95_width_when_no_width_meets_target",
            )
            self.assertIsNotNone(config["fallback"])

    def test_static_width_calibration_uses_worst_profiled_condition(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_path = temp_path / "usm.pt"
            checkpoint_path.write_bytes(b"trained-checkpoint-placeholder")
            profile_path = temp_path / "profile.csv"
            output_path = temp_path / "config.json"
            checkpoint_hash = checkpoint_sha256(checkpoint_path)
            rows = []
            baseline = {0.25: 4.0, 0.5: 8.0, 0.75: 12.0, 1.0: 15.0}
            contention = {0.25: 8.0, 0.5: 14.0, 0.75: 21.0, 1.0: 26.0}
            for condition, latencies in (("baseline", baseline), ("contention", contention)):
                rows.extend({
                    "width_mult": width,
                    "bit_width": 32,
                    "p95_ms": latency,
                    "resource_condition": condition,
                    "checkpoint_sha256": checkpoint_hash,
                } for width, latency in latencies.items())
            with profile_path.open("w", newline="", encoding="utf-8") as profile_file:
                writer = csv.DictWriter(profile_file, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            calibrate_static_width(profile_path, checkpoint_path, output_path, 20.0)

            config = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(config["pinned_width"], 0.5)
            self.assertTrue(config["calibration_target_feasible"])
            self.assertEqual(config["calibration_p95_by_width"]["0.75"], 21.0)
            self.assertEqual(config["calibration_rule"], "worst measured condition P95 per width")

    def test_static_width_calibration_rejects_mismatched_profile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_path = temp_path / "usm.pt"
            checkpoint_path.write_bytes(b"trained-checkpoint-placeholder")
            profile_path = temp_path / "profile.csv"
            rows = [
                {"width_mult": width, "bit_width": 32, "p95_ms": 10.0,
                 "checkpoint_sha256": "not-the-checkpoint"}
                for width in (0.25, 0.5, 0.75, 1.0)
            ]
            with profile_path.open("w", newline="", encoding="utf-8") as profile_file:
                writer = csv.DictWriter(profile_file, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaises(ValueError):
                calibrate_static_width(
                    profile_path, checkpoint_path, temp_path / "config.json", 20.0
                )

    def test_run_manifest_records_warmup_and_replay_policy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir)
            args = SimpleNamespace(
                threads=1,
                bn_calibration_batches=10,
                usm_checkpoint=str(Path(temp_dir) / "checkpoint.pt"),
                warmup_requests=5,
                repetitions=2,
                n_samples=50,
                seed=123,
                target_deadline_ms=30.0,
            )
            Path(args.usm_checkpoint).write_bytes(b"checkpoint")
            write_run_manifests(run_root, "manifest_run", args)
            protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))
            self.assertEqual(protocol["warmup_policy"]["mode"], "exclude_from_latency")
            self.assertEqual(protocol["warmup_policy"]["pinned"], "repeat_pinned_width")
            self.assertEqual(protocol["warmup_policy"]["adaptive"], "repeat_controller_initial_width_only")
            self.assertEqual(protocol["resource_state_semantics"]["controller_input"], "controlled_condition_profile_lookup")
            self.assertIn("matching baseline or controlled-contention profile", protocol["resource_state_semantics"]["latency_prediction"])
            self.assertIn("lowest worst-condition profiled-P95 width", protocol["pinned_width_calibration"]["no_feasible_width_fallback"])
            self.assertEqual(protocol["replay_mode"], "paced_sequential_queue_replay")
            self.assertFalse(protocol["cold_switch_cost_included"])
            self.assertTrue(protocol["width_switch_cost_included"])
            self.assertFalse(protocol["accuracy_selection"]["replay_examples_overlap"])

if __name__ == '__main__':
    unittest.main()
