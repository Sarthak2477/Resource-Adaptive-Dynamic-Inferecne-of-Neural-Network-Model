import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.checkpoint_io import checkpoint_sha256, resolve_usm_checkpoint

def calibrate_static_width(profile_path, checkpoint_path, output_path, target_deadline_ms=30.0):
    profile_path = Path(profile_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not profile_path.is_file():
        raise FileNotFoundError(f"FP32 profile not found: {profile_path}")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite experiment config: {output_path}")
    expected_hash = checkpoint_sha256(checkpoint_path)

    calibration_p95_by_condition = {}
    observed_hashes = set()
    with profile_path.open("r", newline="", encoding="utf-8-sig") as profile_file:
        for row in csv.DictReader(profile_file):
            if int(row["bit_width"]) != 32:
                continue
            width = float(row["width_mult"])
            if width not in (0.25, 0.5, 0.75, 1.0):
                raise ValueError(f"Unsupported width in profile: {width}")
            if "checkpoint_sha256" not in row:
                raise ValueError("Profile has no checkpoint SHA-256 provenance")
            observed_hashes.add(row["checkpoint_sha256"])
            condition = row.get("resource_condition", "baseline") or "baseline"
            condition_rows = calibration_p95_by_condition.setdefault(condition, {})
            if str(width) in condition_rows:
                raise ValueError(f"Duplicate profile row for width {width} under {condition}")
            condition_rows[str(width)] = float(row["p95_ms"])

    if observed_hashes != {expected_hash}:
        raise ValueError(
            "Profile checkpoint hash does not match the selected USM checkpoint: "
            f"profile={sorted(observed_hashes)}, checkpoint={expected_hash}"
        )
    expected_widths = {"0.25", "0.5", "0.75", "1.0"}
    for condition, condition_rows in calibration_p95_by_condition.items():
        if set(condition_rows) != expected_widths:
            raise ValueError(f"Profile condition {condition} must contain exactly the four FP32 widths")
    calibration_p95 = {
        width: max(rows[width] for rows in calibration_p95_by_condition.values())
        for width in expected_widths
    }

    feasible_widths = [
        float(width) for width, p95 in calibration_p95.items()
        if p95 <= target_deadline_ms
    ]
    calibration_target_feasible = bool(feasible_widths)
    if calibration_target_feasible:
        selected_width = max(feasible_widths)
        selection_rule = f"widest_width_with_profile_p95_le_{target_deadline_ms}ms"
        fallback = None
    else:
        selected_width = float(min(calibration_p95, key=calibration_p95.get))
        selection_rule = "lowest_profile_p95_width_when_no_width_meets_target"
        fallback = "No FP32 width met the calibration P95 target; selected the lowest-P95 profiled width as best effort."
    config_data = {
        "comparison": "USM pinned FP32 versus USM adaptive FP32",
        "pinned_width": selected_width,
        "precision": 32,
        "selection_workload": "trained USM, seeded synthetic tensor input, batch_size=1",
        "selection_rule": selection_rule,
        "target_deadline_ms": float(target_deadline_ms),
        "calibration_target_feasible": calibration_target_feasible,
        "minimum_profile_p95_ms": min(calibration_p95.values()),
        "fallback": fallback,
        "calibration_p95_by_width": calibration_p95,
        "calibration_p95_by_condition": calibration_p95_by_condition,
        "calibration_rule": "worst measured condition P95 per width",
        "profile_path": str(profile_path),
        "usm_checkpoint_path": str(Path(checkpoint_path).resolve()),
        "usm_checkpoint_sha256": expected_hash,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as config_file:
        json.dump(config_data, config_file, indent=2)
        config_file.write("\n")
    print(f"Selected static width: {selected_width}")
    print(f"Config saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select static width from a checkpoint-matched FP32 profile.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--profile", type=Path, default=PROJECT_ROOT / "results" / "static_vs_usm_fp32" / "profiles" / "fp32_width_profile.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "static_vs_usm_fp32" / "config.json")
    parser.add_argument("--target-deadline-ms", type=float, default=30.0)
    args = parser.parse_args()
    if args.target_deadline_ms <= 0:
        parser.error("target deadline must be positive")
    checkpoint_path = resolve_usm_checkpoint(args.checkpoint)
    calibrate_static_width(args.profile, checkpoint_path, args.output, args.target_deadline_ms)
