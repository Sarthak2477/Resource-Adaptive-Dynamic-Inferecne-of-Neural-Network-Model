import unittest
import tempfile
from pathlib import Path

from resource_control.qat_experiment import (
    BIT_WIDTHS,
    QAT_CANDIDATES,
    WIDTHS,
    MeasuredQATController,
    candidate_configurations,
    classify_switch,
    select_pinned_configuration,
    validate_accuracy_table,
    validate_latency_table,
)
from scripts.generate_usm_qat_workload import THREE_PHASE_DEADLINES_MS, generate_workload
from scripts.evaluate_width_latency import select_stratified_indices


def accuracy_rows():
    return [
        {
            "width_mult": width,
            "bit_width": bits,
            "correct": 100 - index,
            "total": 100,
        }
        for index, (width, bits) in enumerate(QAT_CANDIDATES)
    ]


def profiles(baseline, contention):
    rows = []
    for condition, values in (("baseline", baseline), ("cpu_contention", contention)):
        for config, p95 in zip(QAT_CANDIDATES, values):
            rows.append({
                "resource_condition": condition,
                "width_mult": config[0],
                "bit_width": config[1],
                "p95_ms": p95,
            })
    return rows


class QATExperimentTests(unittest.TestCase):
    def test_width_latency_test_subset_is_deterministic_and_class_balanced(self):
        targets = [0, 1, 2, 3] * 10

        first = select_stratified_indices(targets, 4, 12, seed=23)
        second = select_stratified_indices(targets, 4, 12, seed=23)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertEqual(len(set(first)), 12)
        self.assertEqual([targets[index] for index in first].count(0), 3)
        self.assertEqual([targets[index] for index in first].count(1), 3)
        self.assertEqual([targets[index] for index in first].count(2), 3)
        self.assertEqual([targets[index] for index in first].count(3), 3)
        with self.assertRaises(ValueError):
            select_stratified_indices(targets, 4, 10, seed=23)

    def test_candidate_space_has_all_sixteen_explicit_pairs(self):
        candidates = candidate_configurations()
        self.assertEqual(len(candidates), 16)
        self.assertEqual(set(candidates), {(w, b) for w in WIDTHS for b in BIT_WIDTHS})

    def test_accuracy_mapping_requires_exactly_one_row_per_candidate(self):
        mapping = validate_accuracy_table(accuracy_rows())
        self.assertEqual(set(mapping), set(QAT_CANDIDATES))
        with self.assertRaises(ValueError):
            validate_accuracy_table(accuracy_rows()[:-1])
        with self.assertRaises(ValueError):
            validate_accuracy_table(accuracy_rows() + [accuracy_rows()[0]])

    def test_latency_mapping_requires_all_candidate_condition_pairs(self):
        table = validate_latency_table(
            profiles(range(10, 26), range(20, 36)),
            ("baseline", "cpu_contention"),
        )
        self.assertEqual(len(table), 32)
        with self.assertRaises(ValueError):
            validate_latency_table(
                profiles(range(10, 26), range(20, 36))[:-1],
                ("baseline", "cpu_contention"),
            )

    def test_resource_condition_changes_latency_and_can_change_selected_bits(self):
        accuracy = validate_accuracy_table([
            {
                "width_mult": width,
                "bit_width": bits,
                "correct": int(50 + bits + 5 * width),
                "total": 100,
            }
            for width, bits in QAT_CANDIDATES
        ])
        table = validate_latency_table(
            profiles(
                [6, 8, 10, 13, 7, 9, 11, 14, 8, 10, 12, 15, 9, 11, 13, 16],
                [12, 11, 10, 9, 11, 10, 9, 8, 10, 9, 8, 7, 9, 8, 7, 6],
            ),
            ("baseline", "cpu_contention"),
        )
        controller = MeasuredQATController(accuracy, table, switch_cost_ms=0.0)

        baseline = controller.select("baseline", deadline_ms=13)
        contention = controller.select("cpu_contention", deadline_ms=13)

        self.assertNotEqual(baseline.config, contention.config)
        self.assertEqual(baseline.config[1], 16)
        self.assertEqual(contention.config[1], 32)
        self.assertTrue(baseline.precision_fallback_used)
        self.assertGreater(contention.fp32_feasible_candidate_count, 0)
        self.assertFalse(contention.precision_fallback_used)
        self.assertEqual(len(baseline.candidates), 16)
        self.assertEqual(len(contention.candidates), 16)

    def test_feasible_fp32_width_beats_more_accurate_low_bit_candidate(self):
        accuracies = {
            config: {"accuracy_percent": 99.0 if config == (1.0, 8) else 80.0}
            for config in QAT_CANDIDATES
        }
        table = validate_latency_table(
            profiles([8] * 16, [8] * 16), ("baseline", "cpu_contention")
        )
        controller = MeasuredQATController(accuracies, table, switch_cost_ms=0.0)

        decision = controller.select("baseline", deadline_ms=10.0)

        self.assertEqual(decision.config[1], 32)
        self.assertFalse(decision.precision_fallback_used)
        self.assertEqual(decision.fp32_feasible_candidate_count, len(WIDTHS))

    def test_lower_precision_is_used_when_no_fp32_width_is_feasible(self):
        accuracies = {
            config: {"accuracy_percent": 95.0 if config == (0.5, 8) else 80.0}
            for config in QAT_CANDIDATES
        }
        latency_rows = []
        for condition in ("baseline", "cpu_contention"):
            for config in QAT_CANDIDATES:
                latency_rows.append({
                    "resource_condition": condition,
                    "width_mult": config[0],
                    "bit_width": config[1],
                    "p95_ms": 5.0 if config == (0.5, 8) else 20.0 if config[1] == 32 else 30.0,
                })
        table = validate_latency_table(latency_rows, ("baseline", "cpu_contention"))
        controller = MeasuredQATController(accuracies, table, switch_cost_ms=0.0)

        decision = controller.select("baseline", deadline_ms=10.0)

        self.assertEqual(decision.config, (0.5, 8))
        self.assertEqual(decision.fp32_feasible_candidate_count, 0)
        self.assertTrue(decision.precision_fallback_used)

    def test_infeasible_fallback_uses_minimum_adjusted_latency(self):
        accuracy = validate_accuracy_table(accuracy_rows())
        table = validate_latency_table(
            profiles(range(10, 26), range(26, 42)),
            ("baseline", "cpu_contention"),
        )
        controller = MeasuredQATController(accuracy, table, switch_cost_ms=2.0)
        controller.current_config = (1.0, 32)

        decision = controller.select("cpu_contention", deadline_ms=4.0)

        self.assertTrue(decision.fallback)
        self.assertEqual(decision.config, QAT_CANDIDATES[0])
        self.assertTrue(decision.fallback_violates_deadline)
        self.assertEqual(decision.feasible_candidate_count, 0)

    def test_pinned_configuration_uses_calibration_accuracy_and_worst_condition(self):
        accuracy = validate_accuracy_table(accuracy_rows())
        table = validate_latency_table(
            profiles(range(6, 22), range(10, 26)),
            ("baseline", "cpu_contention"),
        )

        pinned = select_pinned_configuration(accuracy, table, target_p95_ms=12.0)

        expected = max(
            (config for config in QAT_CANDIDATES if max(
                table[(condition, config)]["p95_ms"] for condition in ("baseline", "cpu_contention")
            ) <= 12.0),
            key=lambda config: accuracy[config]["accuracy_percent"],
        )
        self.assertEqual(pinned["config"], expected)

    def test_switch_classification_for_width_and_precision(self):
        current = (1.0, 32)
        self.assertEqual(classify_switch(current, current), "none")
        self.assertEqual(classify_switch(current, (0.5, 32)), "width_only")
        self.assertEqual(classify_switch(current, (1.0, 8)), "bit_only")
        self.assertEqual(classify_switch(current, (0.5, 8)), "width_and_bit")

    def test_workload_is_frozen_and_covers_each_configured_condition(self):
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.json"
            second_path = Path(directory) / "second.json"
            first = generate_workload(
                42, 30, first_path, 20.0, "heldout",
                ["baseline", "cpu_contention", "gpu_contention"],
            )
            second = generate_workload(
                42, 30, second_path, 20.0, "heldout",
                ["baseline", "cpu_contention", "gpu_contention"],
            )
            self.assertEqual(first["requests"], second["requests"])
            self.assertEqual(
                {row["resource_condition"] for row in first["requests"]},
                {"baseline", "cpu_contention", "gpu_contention"},
            )
            self.assertTrue(all(row["deadline_ms"] > 0 for row in first["requests"]))

    def test_three_phase_workload_has_ordered_deadlines_and_resource_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "three_phase.json"
            workload = generate_workload(
                42, 50, path, 100.0, "heldout",
                ["baseline", "cpu_contention"], three_phase=True,
            )
        requests = workload["requests"]
        self.assertEqual(len(requests), 50)
        self.assertEqual(
            [request["phase"] for request in requests],
            ["phase_1_relaxed"] * 16
            + ["phase_2_constrained"] * 17
            + ["phase_3_recovery"] * 17,
        )
        self.assertEqual(
            [request["resource_condition"] for request in requests],
            ["baseline"] * 16 + ["cpu_contention"] * 17 + ["baseline"] * 17,
        )
        self.assertEqual(
            [request["deadline_ms"] for request in requests],
            [THREE_PHASE_DEADLINES_MS["phase_1_relaxed"]] * 16
            + [THREE_PHASE_DEADLINES_MS["phase_2_constrained"]] * 17
            + [THREE_PHASE_DEADLINES_MS["phase_3_recovery"]] * 17,
        )
        self.assertEqual(
            workload["meta"]["phase_deadlines_ms"],
            THREE_PHASE_DEADLINES_MS,
        )

    def test_measured_transition_cost_is_in_controller_feasibility(self):
        accuracies = validate_accuracy_table([
            {"width_mult": width, "bit_width": bits, "correct": bits, "total": 32}
            for width, bits in QAT_CANDIDATES
        ])
        table = validate_latency_table(
            profiles([10] * 16, [10] * 16), ("baseline", "cpu_contention")
        )
        controller = MeasuredQATController(
            accuracies, table,
            {"width_only": 0.5, "bit_only": 20.0, "width_and_bit": 1.0},
        )
        controller.current_config = (0.25, 4)

        decision = controller.select("baseline", deadline_ms=12.0)

        self.assertEqual(decision.config[1], 4)
        self.assertTrue(decision.feasible)
        self.assertEqual(decision.predicted_switch_ms, 0.0)

    def test_measured_transition_cost_can_make_high_accuracy_candidate_infeasible(self):
        accuracies = validate_accuracy_table([
            {"width_mult": width, "bit_width": bits,
             "correct": 99 if (width, bits) == (1.0, 32) else 60, "total": 100}
            for width, bits in QAT_CANDIDATES
        ])
        latency_values = [
            10 if config in ((0.25, 4), (1.0, 32)) else 40
            for config in QAT_CANDIDATES
        ]
        table = validate_latency_table(
            profiles(latency_values, latency_values), ("baseline", "cpu_contention")
        )
        controller = MeasuredQATController(
            accuracies, table,
            {"width_only": 0.0, "bit_only": 0.0, "width_and_bit": 2.0},
        )
        controller.current_config = (0.25, 4)

        decision = controller.select("baseline", deadline_ms=12.0)

        self.assertEqual(decision.config, (0.25, 4))
        joint_candidate = next(row for row in decision.candidates if row["config"] == (1.0, 32))
        self.assertFalse(joint_candidate["feasible"])
        self.assertEqual(joint_candidate["switch_cost_ms"], 2.0)


if __name__ == "__main__":
    unittest.main()