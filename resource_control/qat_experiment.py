from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from models.config import FLAGS


WIDTHS = (0.25, 0.50, 0.75, 1.00)
BIT_WIDTHS = (4, 8, 16, 32)
QAT_CANDIDATES = tuple((width, bits) for width in WIDTHS for bits in BIT_WIDTHS)
PRIMARY_LATENCY_QUANTILE = "p95"
SAFETY_MARGIN = 0.90


def candidate_configurations():
    configured = tuple((float(width), int(bits)) for width, bits in FLAGS.deploy_configs)
    if len(configured) != 16 or set(configured) != set(QAT_CANDIDATES):
        raise ValueError("USM QAT experiment requires the canonical 16 width-bit candidates")
    return QAT_CANDIDATES


def validate_accuracy_table(rows):
    result = {}
    for row in rows:
        config = (float(row["width_mult"]), int(row["bit_width"]))
        if config not in QAT_CANDIDATES:
            raise ValueError(f"Unsupported QAT candidate in accuracy table: {config}")
        if config in result:
            raise ValueError(f"Duplicate accuracy row for {config}")
        total = int(row["total"])
        correct = int(row["correct"])
        if total <= 0 or not 0 <= correct <= total:
            raise ValueError(f"Invalid accuracy counts for {config}")
        result[config] = {
            "correct": correct,
            "total": total,
            "accuracy_percent": 100.0 * correct / total,
        }
    if set(result) != set(QAT_CANDIDATES):
        missing = sorted(set(QAT_CANDIDATES) - set(result))
        raise ValueError(f"Accuracy table must cover all 16 candidates; missing {missing}")
    return result


def validate_latency_table(rows, conditions):
    result = {}
    expected = {(condition, config) for condition in conditions for config in QAT_CANDIDATES}
    for row in rows:
        condition = str(row["resource_condition"])
        config = (float(row["width_mult"]), int(row["bit_width"]))
        key = (condition, config)
        if key not in expected:
            raise ValueError(f"Unexpected measured latency entry {key}")
        if key in result:
            raise ValueError(f"Duplicate measured latency entry {key}")
        p95 = float(row.get("p95_ms", row.get("latency_p95_ms")))
        if not math.isfinite(p95) or p95 <= 0:
            raise ValueError(f"Invalid measured P95 for {key}: {p95}")
        result[key] = {**row, "p95_ms": p95}
    if set(result) != expected:
        missing = sorted(expected - set(result))
        raise ValueError(f"Latency table must cover each candidate/condition; missing {missing}")
    return result


def classify_switch(previous_config, selected_config):
    if previous_config is None or previous_config == selected_config:
        return "none"
    width_changed = float(previous_config[0]) != float(selected_config[0])
    bits_changed = int(previous_config[1]) != int(selected_config[1])
    if width_changed and bits_changed:
        return "width_and_bit"
    if width_changed:
        return "width_only"
    return "bit_only"


@dataclass(frozen=True)
class QATDecision:
    config: tuple[float, int]
    predicted_p95_ms: float
    predicted_switch_ms: float
    feasible: bool
    fallback: bool
    fallback_violates_deadline: bool
    feasible_candidate_count: int
    fp32_feasible_candidate_count: int
    precision_fallback_used: bool
    candidates: tuple[dict, ...]


class MeasuredQATController:
    """Prefer width adaptation in FP32; lower precision is a last resort."""

    def __init__(
        self,
        accuracies: Mapping[tuple[float, int], Mapping[str, float]],
        latency_profiles: Mapping[tuple[str, tuple[float, int]], Mapping[str, float]],
        switch_cost_ms: float,
        safety_margin: float = SAFETY_MARGIN,
    ):
        if set(accuracies) != set(QAT_CANDIDATES):
            raise ValueError("Controller accuracy mapping must cover all 16 candidates")
        self.accuracies = accuracies
        self.latency_profiles = latency_profiles
        if isinstance(switch_cost_ms, Mapping):
            self.switch_cost_ms = {
                key: max(0.0, float(value))
                for key, value in switch_cost_ms.items()
            }
        else:
            self.switch_cost_ms = {"width_only": max(0.0, float(switch_cost_ms)),
                                   "bit_only": max(0.0, float(switch_cost_ms)),
                                   "width_and_bit": max(0.0, float(switch_cost_ms))}
        self.safety_margin = float(safety_margin)
        self.current_config = None
        self.last_decision = None

    def predict_p95(self, config, resource_condition):
        key = (resource_condition, (float(config[0]), int(config[1])))
        try:
            return float(self.latency_profiles[key]["p95_ms"])
        except KeyError as exc:
            raise ValueError(f"No measured P95 profile for {key}") from exc

    def select(self, resource_condition, deadline_ms):
        if deadline_ms <= 0:
            raise ValueError("deadline_ms must be positive")
        target_ms = float(deadline_ms) * self.safety_margin
        scored = []
        for config in QAT_CANDIDATES:
            p95_ms = self.predict_p95(config, resource_condition)
            switch_kind = classify_switch(self.current_config, config)
            transition_ms = self.switch_cost_ms.get(switch_kind, 0.0)
            adjusted_ms = p95_ms + transition_ms
            scored.append({
                "config": config,
                "accuracy_percent": float(self.accuracies[config]["accuracy_percent"]),
                "predicted_p95_ms": p95_ms,
                "switch_cost_ms": transition_ms,
                "adjusted_p95_ms": adjusted_ms,
                "feasible": adjusted_ms <= target_ms,
            })

        feasible = [row for row in scored if row["feasible"]]
        fp32_feasible = [row for row in feasible if row["config"][1] == 32]
        precision_fallback_used = not fp32_feasible
        fallback = not feasible
        if fp32_feasible:
            chosen = max(
                fp32_feasible,
                key=lambda row: (row["accuracy_percent"], -row["adjusted_p95_ms"]),
            )
            violates = False
        elif feasible:
            chosen = max(feasible, key=lambda row: (row["accuracy_percent"], -row["adjusted_p95_ms"]))
            violates = False
        else:
            chosen = min(scored, key=lambda row: (row["adjusted_p95_ms"], -row["accuracy_percent"]))
            violates = chosen["adjusted_p95_ms"] > float(deadline_ms)

        self.current_config = chosen["config"]
        self.last_decision = QATDecision(
            config=chosen["config"],
            predicted_p95_ms=chosen["predicted_p95_ms"],
            predicted_switch_ms=chosen["switch_cost_ms"],
            feasible=bool(chosen["feasible"]),
            fallback=fallback,
            fallback_violates_deadline=violates,
            feasible_candidate_count=len(feasible),
            fp32_feasible_candidate_count=len(fp32_feasible),
            precision_fallback_used=precision_fallback_used,
            candidates=tuple(scored),
        )
        return self.last_decision


def select_pinned_configuration(accuracies, latency_profiles, target_p95_ms):
    """Calibration-only pinned choice: highest accuracy meeting worst-condition P95."""
    conditions = sorted({condition for condition, _ in latency_profiles})
    rows = []
    for config in QAT_CANDIDATES:
        worst_p95 = max(float(latency_profiles[(condition, config)]["p95_ms"]) for condition in conditions)
        rows.append((config, float(accuracies[config]["accuracy_percent"]), worst_p95))
    feasible = [row for row in rows if row[2] <= target_p95_ms]
    selected = max(feasible, key=lambda row: (row[1], -row[2])) if feasible else min(
        rows, key=lambda row: (row[2], -row[1])
    )
    return {
        "config": selected[0],
        "accuracy_percent": selected[1],
        "worst_condition_p95_ms": selected[2],
        "target_feasible": bool(feasible),
        "fallback": None if feasible else "no candidate met calibration P95; chose minimum worst-condition P95",
    }


def prediction_validation(calibration_p95, heldout_p95_rows):
    pairs = []
    for row in heldout_p95_rows:
        key = (str(row["resource_condition"]), (float(row["width_mult"]), int(row["bit_width"])))
        if key not in calibration_p95:
            raise ValueError(f"No calibration prediction for held-out profile row {key}")
        observed = float(row["p95_ms"])
        predicted = float(calibration_p95[key])
        pairs.append({**row, "predicted_p95_ms": predicted, "error_ms": predicted - observed})
    if not pairs:
        raise ValueError("At least one held-out profile row is required")
    errors = [item["error_ms"] for item in pairs]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    return {
        "mae_ms": mae,
        "rmse_ms": rmse,
        "p50_absolute_error_ms": sorted(abs(error) for error in errors)[(len(errors) - 1) // 2],
        "p95_absolute_error_ms": sorted(abs(error) for error in errors)[min(len(errors) - 1, math.ceil(0.95 * len(errors)) - 1)],
        "n_validation_pairs": len(pairs),
        "per_candidate_condition": pairs,
    }