import os
import sys
import json
import glob
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def main():
    parser = argparse.ArgumentParser(description="Summarize one paired USM pinned/adaptive FP32 experiment.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--accuracy-csv", type=Path, required=True)
    parser.add_argument("--workload-file", type=Path, required=True)
    args = parser.parse_args()

    raw_dir = args.raw_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    expected_outputs = [
        output_dir / "summary.csv",
        output_dir / "resource_condition_summary.csv",
        output_dir / "resource_condition_paired_differences.csv",
        output_dir / "paired_differences.csv",
        output_dir / "accuracy_by_width.csv",
        output_dir / "result_table.md",
        output_dir / "accuracy_vs_p95_latency.png",
        output_dir / "deadline_miss_rate.png",
        output_dir / "width_selection_over_requests.png",
        output_dir / "latency_ecdf.png",
        output_dir / "resource_conditions_width.png",
    ]
    existing = [path for path in expected_outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite prior report output: {existing[0]}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    records = []
    for file_path in glob.glob(str(raw_dir / "*.jsonl")):
        with open(file_path, "r", encoding="utf-8") as fp:
            for line in fp:
                record = json.loads(line)
                if record.get("run_id", "").startswith(f"{args.run_id}_rep"):
                    records.append(record)
                
    if not records:
        print("No raw data found.")
        return
        
    df = pd.DataFrame(records)
    if "synthetic_resource_state" in df.columns and "cpu_pct" not in df.columns:
        for key in ("cpu_pct", "mem_available_mb", "battery_pct", "thermal_c", "timestamp"):
            df[key] = df["synthetic_resource_state"].apply(
                lambda state: state.get(key) if isinstance(state, dict) else None
            )
    if "cpu_pct" not in df.columns and "observed_cpu_pct" in df.columns:
        df["cpu_pct"] = df["observed_cpu_pct"]
    expected_arms = {"usm_pinned", "usm_adaptive"}
    if set(df["arm"].unique()) != expected_arms:
        raise ValueError(f"Expected exactly {sorted(expected_arms)}; got {sorted(df['arm'].unique())}")

    with args.workload_file.expanduser().open("r", encoding="utf-8") as workload_file:
        workload_requests = json.load(workload_file)["requests"]
    expected_workload = [
        (request["request_id"], int(request["sample_index"]), float(request["budget_ms"]),
         request["input_sample_id"], int(request["dataset_index"]), request["dataset_image_path"])
        for request in workload_requests
    ]

    paired_rows = []
    per_run = []
    for run_id, run_df in df.groupby("run_id", sort=True):
        arm_rows = {arm: run_df[run_df["arm"] == arm].sort_values("sample_index") for arm in expected_arms}
        request_keys = {}
        for arm, arm_df in arm_rows.items():
            if arm_df["request_id"].duplicated().any():
                raise ValueError(f"Duplicate request_id in {run_id}/{arm}")
            request_keys[arm] = list(zip(
                arm_df["request_id"], arm_df["sample_index"], arm_df["budget_ms"],
                arm_df["input_sample_id"], arm_df["selected_bit_width"],
                arm_df["dataset_index"], arm_df["dataset_image_path"],
            ))
            if not (arm_df["selected_bit_width"] == 32).all():
                raise ValueError(f"Non-FP32 request found in {run_id}/{arm}")
            if list(zip(
                arm_df["request_id"], arm_df["sample_index"], arm_df["budget_ms"],
                arm_df["input_sample_id"], arm_df["dataset_index"], arm_df["dataset_image_path"]
            )) != expected_workload:
                raise ValueError(f"{run_id}/{arm} does not match the frozen workload artifact")
        if request_keys["usm_pinned"] != request_keys["usm_adaptive"]:
            raise ValueError(f"Request/deadline/input alignment mismatch in {run_id}")

        for arm, arm_df in arm_rows.items():
            model_latency = arm_df["model_latency_ms"].to_numpy(dtype=float)
            end_to_end = arm_df["end_to_end_latency_ms"].to_numpy(dtype=float)
            queue_delay = arm_df["queue_delay_ms"].to_numpy(dtype=float)
            width_switch_overhead = arm_df["width_switch_overhead_ms"].to_numpy(dtype=float)
            budgets = arm_df["budget_ms"].to_numpy(dtype=float)
            missed = end_to_end > budgets
            violations = np.maximum(0.0, end_to_end - budgets)
            widths = arm_df["selected_width"].to_numpy(dtype=float)
            switches = int(arm_df["switched"].sum())
            n_requests = len(arm_df)
            summary_row = {
                "run_id": run_id,
                "repetition": int(run_id.rsplit("rep", 1)[1]),
                "arm": arm,
                "n_requests": n_requests,
                "accuracy_percent": 100.0 * float(arm_df["is_correct"].mean()),
                "model_mean_ms": float(np.mean(model_latency)),
                "model_p50_ms": float(np.percentile(model_latency, 50)),
                "model_p95_ms": float(np.percentile(model_latency, 95)),
                "model_p99_ms": float(np.percentile(model_latency, 99)),
                "e2e_mean_ms": float(np.mean(end_to_end)),
                "e2e_p50_ms": float(np.percentile(end_to_end, 50)),
                "e2e_p95_ms": float(np.percentile(end_to_end, 95)),
                "e2e_p99_ms": float(np.percentile(end_to_end, 99)),
                "mean_queue_delay_ms": float(np.mean(queue_delay)),
                "queue_delay_p95_ms": float(np.percentile(queue_delay, 95)),
                "deadline_misses": int(missed.sum()),
                "deadline_total_requests": n_requests,
                "deadline_miss_rate_percent": 100.0 * float(missed.mean()),
                "mean_violation_ms_all_requests": float(violations.mean()),
                "mean_violation_ms_misses_only": float(violations[missed].mean()) if missed.any() else 0.0,
                "switch_count": switches,
                "switch_rate_percent": 100.0 * switches / max(1, n_requests),
                "controller_overhead_mean_ms": float(arm_df["controller_overhead_ms"].mean()),
                "controller_overhead_p95_ms": float(np.percentile(arm_df["controller_overhead_ms"], 95)),
                "width_switch_overhead_mean_ms": float(np.mean(width_switch_overhead)),
                "controller_feedback_update_mean_ms": float(arm_df["controller_feedback_update_ms"].mean()),
                "controller_feedback_update_p95_ms": float(np.percentile(arm_df["controller_feedback_update_ms"], 95)),
                "mean_selected_width": float(widths.mean()),
                "peak_process_rss_mb": float(arm_df["process_rss_mb"].max()),
                "peak_gpu_allocated_mb": float(arm_df["gpu_peak_allocated_mb"].max()) if arm_df["gpu_peak_allocated_mb"].notna().any() else None,
                "mean_observed_cpu_pct": float(arm_df["observed_cpu_pct"].mean()),
                "minimum_observed_mem_available_mb": float(arm_df["observed_mem_available_mb"].min()),
                "maximum_observed_thermal_c": float(arm_df["observed_thermal_c"].max()) if arm_df["observed_thermal_c"].notna().any() else None,
                "resource_trace_kind": str(arm_df["resource_trace_kind"].iloc[0]),
            }
            per_run.append(summary_row)
        paired_rows.append((run_id, arm_rows))

    summary = pd.DataFrame(per_run).sort_values(["repetition", "arm"])
    summary.to_csv(output_dir / "summary.csv", index=False)

    condition_summary = df.groupby(["run_id", "arm", "scheduled_resource_scenario"], dropna=False).agg(
        n_requests=("request_id", "count"),
        mean_selected_width=("selected_width", "mean"),
        accuracy_percent=("is_correct", lambda values: 100.0 * values.mean()),
        end_to_end_p95_ms=("end_to_end_latency_ms", lambda values: float(np.percentile(values, 95))),
        deadline_miss_rate_percent=("missed_deadline", lambda values: 100.0 * values.mean()),
        mean_queue_delay_ms=("queue_delay_ms", "mean"),
        mean_observed_cpu_pct=("observed_cpu_pct", "mean"),
    ).reset_index()
    condition_summary.to_csv(output_dir / "resource_condition_summary.csv", index=False)

    condition_differences = []
    condition_rng = np.random.default_rng(20261002)
    for condition, condition_rows in df.groupby("scheduled_resource_scenario", sort=True):
        condition_by_run = condition_rows.groupby(["run_id", "arm"]).agg(
            accuracy_percent=("is_correct", lambda values: 100.0 * values.mean()),
            e2e_p95_ms=("end_to_end_latency_ms", lambda values: float(np.percentile(values, 95))),
            deadline_miss_rate_percent=("missed_deadline", lambda values: 100.0 * values.mean()),
            mean_selected_width=("selected_width", "mean"),
            mean_queue_delay_ms=("queue_delay_ms", "mean"),
        )
        for metric in condition_by_run.columns:
            paired_values = condition_by_run[metric].unstack("arm")
            if not {"usm_adaptive", "usm_pinned"}.issubset(paired_values.columns):
                raise ValueError(f"Missing paired arm for resource condition {condition}")
            differences = (
                paired_values["usm_adaptive"] - paired_values["usm_pinned"]
            ).dropna().to_numpy(dtype=float)
            if not len(differences):
                raise ValueError(f"No complete paired repetitions for resource condition {condition}")
            bootstrap_means = np.mean(
                condition_rng.choice(differences, size=(10000, len(differences)), replace=True),
                axis=1,
            )
            condition_differences.append({
                "resource_condition": condition,
                "metric": metric,
                "n_paired_repetitions": len(differences),
                "mean_adaptive_minus_pinned": float(differences.mean()),
                "bootstrap_95_ci_lower": float(np.percentile(bootstrap_means, 2.5)),
                "bootstrap_95_ci_upper": float(np.percentile(bootstrap_means, 97.5)),
                "uncertainty_method": "percentile bootstrap over complete paired repetitions; 10000 resamples; seed=20261002",
            })
    pd.DataFrame(condition_differences).to_csv(
        output_dir / "resource_condition_paired_differences.csv", index=False
    )

    paired_metrics = [
        "accuracy_percent", "model_p50_ms", "model_p95_ms", "model_p99_ms",
        "e2e_p50_ms", "e2e_p95_ms", "e2e_p99_ms", "deadline_miss_rate_percent",
        "mean_violation_ms_all_requests", "peak_process_rss_mb",
        "switch_rate_percent", "controller_overhead_mean_ms",
        "mean_queue_delay_ms", "width_switch_overhead_mean_ms",
        "controller_feedback_update_mean_ms",
    ]
    pair_diffs = []
    by_rep = summary.pivot(index="run_id", columns="arm", values=paired_metrics)
    if by_rep.isna().any().any():
        raise ValueError("At least one repetition is missing a metric for one of the paired arms")
    rng = np.random.default_rng(20260930)
    for metric in paired_metrics:
        differences = (
            by_rep[(metric, "usm_adaptive")] - by_rep[(metric, "usm_pinned")]
        ).to_numpy(dtype=float)
        bootstrap_means = np.mean(
            rng.choice(differences, size=(10000, len(differences)), replace=True), axis=1
        )
        pair_diffs.append({
            "metric": metric,
            "n_paired_repetitions": len(differences),
            "mean_paired_difference_adaptive_minus_pinned": float(differences.mean()),
            "median_paired_difference": float(np.median(differences)),
            "bootstrap_95_ci_lower": float(np.percentile(bootstrap_means, 2.5)),
            "bootstrap_95_ci_upper": float(np.percentile(bootstrap_means, 97.5)),
            "uncertainty_method": "percentile bootstrap over complete paired repetitions; 10000 resamples; seed=20260930",
        })
    paired = pd.DataFrame(pair_diffs)
    paired.to_csv(output_dir / "paired_differences.csv", index=False)

    arm_means = summary.groupby("arm").mean(numeric_only=True)
    with (output_dir / "result_table.md").open("x", encoding="utf-8") as result_file:
        result_file.write("# Paired USM FP32 Results\n\n")
        result_file.write("Values are means over complete evaluation repetitions; requests are not treated as independent replicates.\n\n")
        result_file.write("| Metric | USM pinned FP32 | USM adaptive FP32 | Adaptive - pinned (95% paired bootstrap CI) |\n")
        result_file.write("| --- | ---: | ---: | ---: |\n")
        display_metrics = [
            ("Accuracy (%)", "accuracy_percent"),
            ("Model P50 (ms)", "model_p50_ms"),
            ("Model P95 (ms)", "model_p95_ms"),
            ("Model P99 (ms)", "model_p99_ms"),
            ("End-to-end P50 (ms)", "e2e_p50_ms"),
            ("End-to-end P95 (ms)", "e2e_p95_ms"),
            ("End-to-end P99 (ms)", "e2e_p99_ms"),
            ("Deadline miss rate (%)", "deadline_miss_rate_percent"),
            ("Mean deadline violation (ms/request)", "mean_violation_ms_all_requests"),
            ("Mean queue delay (ms/request)", "mean_queue_delay_ms"),
            ("Peak process RSS (MB)", "peak_process_rss_mb"),
            ("Switch rate (%)", "switch_rate_percent"),
            ("Controller overhead mean (ms)", "controller_overhead_mean_ms"),
            ("Width assignment mean (ms; included in response time)", "width_switch_overhead_mean_ms"),
            ("Controller feedback update mean (ms; excluded from response deadline)", "controller_feedback_update_mean_ms"),
        ]
        for label, metric in display_metrics:
            left = float(arm_means.loc["usm_pinned", metric])
            right = float(arm_means.loc["usm_adaptive", metric])
            diff = paired[paired["metric"] == metric].iloc[0]
            interval = f"{diff['mean_paired_difference_adaptive_minus_pinned']:.4f} [{diff['bootstrap_95_ci_lower']:.4f}, {diff['bootstrap_95_ci_upper']:.4f}]"
            result_file.write(f"| {label} | {left:.4f} | {right:.4f} | {interval} |\n")
        mean_misses = summary.groupby("arm")["deadline_misses"].mean()
        total_requests = int(summary["deadline_total_requests"].mean())
        result_file.write(
            f"| Deadline misses (mean count / requests per run) | "
            f"{mean_misses['usm_pinned']:.2f} / {total_requests} | "
            f"{mean_misses['usm_adaptive']:.2f} / {total_requests} | n/a |\n"
        )
        result_file.write(f"\nComplete paired repetitions: {len(paired_rows)}. Workload identity: {args.workload_file.name}.\n")

    # Width accuracy estimates use a disjoint selection-calibration subset.
    accuracy_df = pd.read_csv(args.accuracy_csv)
    profile_df = pd.read_csv(args.profile)
    baseline_profile = profile_df[
        (profile_df["bit_width"] == 32) & (profile_df["resource_condition"] == "baseline")
    ]
    widths_accuracy = accuracy_df[accuracy_df["bit_width"] == 32].merge(
        baseline_profile, on=["width_mult", "bit_width"], validate="one_to_one"
    )
    accuracy_by_width = widths_accuracy[[
        "width_mult", "bit_width", "accuracy_percent", "total", "correct",
        "wilson_95_lower_percent", "wilson_95_upper_percent", "latency_ms",
        "p50_ms", "p95_ms", "p99_ms", "approx_flops", "approx_params",
    ]].sort_values("width_mult")
    accuracy_by_width.to_csv(output_dir / "accuracy_by_width.csv", index=False)
    pinned_widths = set(df.loc[df["arm"] == "usm_pinned", "selected_width"].astype(float))
    if len(pinned_widths) != 1:
        raise ValueError(f"Pinned arm changed width during the experiment: {sorted(pinned_widths)}")
    pinned_width = pinned_widths.pop()
    pinned_accuracy_row = accuracy_by_width[accuracy_by_width["width_mult"] == pinned_width]
    if len(pinned_accuracy_row) != 1:
        raise ValueError(f"No selection-calibration accuracy result for pinned width {pinned_width}")
    with (output_dir / "result_table.md").open("a", encoding="utf-8") as result_file:
        result_file.write("\n## Width Selection Calibration and Profile\n\n")
        result_file.write(f"Pinned width selected using worst-condition profile P95: **{pinned_width:.2f}**; disjoint selection-calibration top-1 accuracy estimate: **{pinned_accuracy_row.iloc[0]['accuracy_percent']:.3f}%**. These examples are excluded from replay accuracy.\n\n")
        result_file.write("| Width | Precision | Selection-calibration accuracy (%) | Calibration N | Baseline mean (ms) | Baseline P50 (ms) | Baseline P95 (ms) | Baseline P99 (ms) | Approx FLOPs | Approx parameters |\n")
        result_file.write("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for _, row in accuracy_by_width.iterrows():
            result_file.write(
                f"| {row['width_mult']:.2f} | FP32 | {row['accuracy_percent']:.3f} | {int(row['total'])} | "
                f"{row['latency_ms']:.4f} | {row['p50_ms']:.4f} | {row['p95_ms']:.4f} | {row['p99_ms']:.4f} | "
                f"{row['approx_flops']:.0f} | {row['approx_params']:.0f} |\n"
            )
    plt.figure()
    for condition, marker in (("baseline", "o"), ("contention", "x")):
        condition_profile = profile_df[
            (profile_df["bit_width"] == 32)
            & (profile_df["resource_condition"] == condition)
        ]
        condition_accuracy = accuracy_df[accuracy_df["bit_width"] == 32].merge(
            condition_profile, on=["width_mult", "bit_width"], validate="one_to_one"
        )
        plt.scatter(
            condition_accuracy["p95_ms"], condition_accuracy["accuracy_percent"],
            marker=marker, label=f"USM profiles and selection-calibration accuracy: {condition}",
        )
        for _, point in condition_accuracy.iterrows():
            plt.annotate(f"w={point['width_mult']:.2f}", (point["p95_ms"], point["accuracy_percent"]))
    for arm, label, marker in (("usm_pinned", "USM pinned replay", "s"), ("usm_adaptive", "USM adaptive replay", "^") ):
        plt.scatter(arm_means.loc[arm, "e2e_p95_ms"], arm_means.loc[arm, "accuracy_percent"], label=label, marker=marker, s=70)
    plt.xlabel("P95 latency (ms); profile for subnet points, end-to-end for replay arms")
    plt.ylabel("Top-1 accuracy (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_p95_latency.png")
    plt.close()

    miss_plot = summary.groupby("arm")["deadline_miss_rate_percent"].agg(["mean", "std"])
    plt.figure()
    plt.bar(miss_plot.index, miss_plot["mean"], yerr=miss_plot["std"].fillna(0.0), capsize=4)
    plt.ylabel("Deadline miss rate (%)")
    plt.tight_layout()
    plt.savefig(output_dir / "deadline_miss_rate.png")
    plt.close()

    first_run_id = sorted(df["run_id"].unique())[0]
    adaptive_first = df[(df["run_id"] == first_run_id) & (df["arm"] == "usm_adaptive")].sort_values("sample_index")
    plt.figure()
    plt.step(adaptive_first["sample_index"], adaptive_first["selected_width"], where="post")
    plt.xlabel("Request index")
    plt.ylabel("Selected width")
    plt.yticks([0.25, 0.5, 0.75, 1.0])
    plt.tight_layout()
    plt.savefig(output_dir / "width_selection_over_requests.png")
    plt.close()

    plt.figure()
    for arm in ("usm_pinned", "usm_adaptive"):
        values = np.sort(df.loc[df["arm"] == arm, "end_to_end_latency_ms"].to_numpy(dtype=float))
        plt.plot(values, np.arange(1, len(values) + 1) / len(values), label=arm)
    plt.xlabel("End-to-end latency (ms)")
    plt.ylabel("Empirical CDF (descriptive pooled requests)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "latency_ecdf.png")
    plt.close()

    plt.figure()
    for condition, marker in (("baseline", "o"), ("contention", "x")):
        condition_rows = adaptive_first[adaptive_first["resource_condition"] == condition]
        plt.scatter(
            condition_rows["observed_cpu_pct"], condition_rows["selected_width"],
            label=f"{condition} condition", marker=marker, alpha=0.7,
        )
    plt.xlabel("CPU utilization (%)")
    plt.ylabel("Adaptive selected width")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "resource_conditions_width.png")
    plt.close()

    print(f"Paired report and plots generated in {output_dir}")

if __name__ == "__main__":
    main()
