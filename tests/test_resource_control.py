import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from resource_control.controller import (
    FEATURES,
    MeasuredFrontierLatencyModel,
    PhysicsSurrogateModel,
    SurrogateBackedController,
)
from resource_control.telemetry import ResourceState


class ConstantSurrogate:
    def __init__(self, latency_ms):
        self.latency_ms = latency_ms

    def predict(self, rows):
        return [self.latency_ms for _ in rows]


class WidthSurrogate:
    def predict(self, rows):
        return [row[0] * 20.0 for row in rows]


def make_frontier():
    return [
        {"config": (0.25, 32), "latency_ms": 5.0, "std_ms": 0.5, "acc": 89.0},
        {"config": (0.5, 32), "latency_ms": 10.0, "std_ms": 0.5, "acc": 91.0},
        {"config": (1.0, 32), "latency_ms": 20.0, "std_ms": 0.5, "acc": 93.0},
    ]


def make_state():
    return ResourceState(15.0, 2048.0, 90.0, 45.0, 0.0)


def make_fingerprint():
    return {"device_speed_score": 1000.0, "cpu_cores": 4, "ram_gb": 8.0, "has_cuda": 0.0}


class ResourceControlTests(unittest.TestCase):
    def test_physics_surrogate_uses_named_feature_positions(self):
        row = [
            0.5, 32.0, 2_000_000.0, 200_000.0, 2_000.0,
            25.0, 1024.0, 45.0, 1000.0, 8.0, 16.0, 0.0,
        ]

        prediction = PhysicsSurrogateModel().predict([row])[0]

        self.assertAlmostEqual(prediction, 2_500_000.0)
        self.assertEqual(len(FEATURES), len(row))

    def test_controller_selects_highest_accuracy_within_risk_budget(self):
        controller = SurrogateBackedController(
            make_frontier(),
            surrogate_model=WidthSurrogate(),
            min_dwell_s=0.0,
            safety_margin=0.9,
            k_risk=1.0,
        )

        selected = controller.select(make_state(), 12.0, make_fingerprint())

        self.assertEqual(selected, (0.5, 32))
        self.assertAlmostEqual(controller.last_predicted_latency, 10.0)

    def test_measured_frontier_model_returns_profile_latency(self):
        frontier = make_frontier()
        frontier[0]["latency_p95_ms"] = 7.5
        model = MeasuredFrontierLatencyModel(frontier, latency_policy="p95")
        row = [0.5, 32.0, 0.0, 0.0, 0.0, 15.0, 2048.0, 45.0, 1000.0, 4.0, 8.0, 0.0]

        self.assertEqual(model.predict([row]), [10.0])

    def test_p95_policy_uses_percentile_and_classifies_infeasible_budget(self):
        frontier = make_frontier()
        for row in frontier:
            row["latency_p95_ms"] = row["latency_ms"] + 5.0
        controller = SurrogateBackedController(
            frontier,
            surrogate_model=MeasuredFrontierLatencyModel(frontier, latency_policy="p95"),
            min_dwell_s=0.0,
            latency_policy="p95",
        )

        selected = controller.select(make_state(), 9.0, make_fingerprint())

        self.assertEqual(selected, (0.25, 32))
        self.assertFalse(controller.last_budget_feasible)
        self.assertEqual(controller.last_selection_status, "infeasible_budget")
        self.assertEqual(controller.last_min_safe_latency, 10.0)

    def test_feedback_is_scoped_to_selected_configuration(self):
        controller = SurrogateBackedController(make_frontier(), min_dwell_s=0.0)

        controller.update_feedback(20.0, 10.0, config=(0.5, 32))

        self.assertAlmostEqual(controller.k_adapt_by_config[(0.5, 32)], 1.05)
        self.assertAlmostEqual(controller.k_adapt_by_config[(0.25, 32)], 1.0)

    def test_controller_accounts_for_switching_penalty(self):
        controller = SurrogateBackedController(
            make_frontier(),
            surrogate_model=ConstantSurrogate(10.0),
            min_dwell_s=0.0,
            switching_penalty_ms=2.0,
        )

        selected = controller.select(make_state(), 12.0, make_fingerprint())

        self.assertEqual(selected, (0.25, 32))

    def test_controller_falls_back_to_fastest_config_when_none_is_feasible(self):
        controller = SurrogateBackedController(
            make_frontier(),
            surrogate_model=ConstantSurrogate(50.0),
            min_dwell_s=0.0,
        )

        selected = controller.select(make_state(), 1.0, make_fingerprint())

        self.assertEqual(selected, (0.25, 32))
        self.assertAlmostEqual(controller.last_predicted_latency, 50.0)

    def test_feedback_updates_and_clips_adaptation_factor(self):
        controller = SurrogateBackedController(make_frontier(), min_dwell_s=0.0)

        controller.update_feedback(20.0, 10.0)
        self.assertAlmostEqual(controller.k_adapt, 1.05)

        for _ in range(40):
            controller.update_feedback(10_000.0, 1.0)
        self.assertEqual(controller.k_adapt, 3.0)

        for _ in range(100):
            controller.update_feedback(0.0, 1.0)
        self.assertAlmostEqual(controller.k_adapt, 0.5)


if __name__ == "__main__":
    unittest.main()
