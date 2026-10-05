import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from resource_control.qat_experiment import QAT_CANDIDATES


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q))


def paired_bootstrap(differences, rng, samples=10000):
    differences = np.asarray(differences, dtype=float)
    means = np.mean(rng.choice(differences, size=(samples, len(differences)), replace=True), axis=1)
    return {
        "mean_paired_difference_adaptive_minus_pinned": float(np.mean(differences)),
        "median_paired_difference": float(np.median(differences)),
        "ci_95_lower": float(np.percentile(means, 2.5)),
        "ci_95_upper": float(np.percentile(means, 97.5)),
        "n_paired_repetitions": int(len(differences)),
    }


def read_raw(raw_dir, run_id):
    records = []
    for path in sorted(Path(raw_dir).glob(f"*_{run_id}_rep*.jsonl")):
        with path.open("r", encoding="utf-8") as source:
            records.extend(json.loads(line) for line in source if line.strip())
    if not records:
        raise FileNotFoundError(f"No raw QAT results found for run {run_id}")
    return pd.DataFrame(records)


def create_report(run_id, run_root, output_dir=None):
    run_root = Path(run_root).resolve()
    analysis_dir = Path(output_dir).resolve() if output_dir else run_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    output_files = [
        "candidate_accuracy.csv", "candidate_latency.csv", "resource_condition_latency.csv",
        "summary.csv", "paired_comparison.csv", "selection_frequency.csv",
        "switching_analysis.csv", "deadline_analysis.csv", "controller_quality.json",
        "result_table.md", "configuration_selection.png", "accuracy_latency_pareto.png",
        "latency_cdf.png", "resource_state_configuration.png", "paired_differences.png",
    ]
    existing = [analysis_dir / name for name in output_files if (analysis_dir / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite QAT report artifact {existing[0]}")

    accuracy_doc = json.loads((run_root / "accuracy" / "candidate_accuracy.json").read_text(encoding="utf-8"))
    profiles = pd.read_csv(run_root / "profiles" / "resource_condition_profiles.csv")
    validation_path = run_root / "profiles" / "prediction_validation.json"
    if validation_path.exists():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
    else:
        profile_metadata = json.loads((run_root / "profiles" / "profile_metadata.json").read_text(encoding="utf-8"))
        validation = {"status": profile_metadata.get("prediction_validation", "not available")}
    workload_doc = json.loads((run_root / "workload" / "workload.json").read_text(encoding="utf-8"))
    df = read_raw(run_root / "raw", run_id)
    if set(df["arm"].unique()) != {"usm_pinned", "usm_adaptive"}:
        raise ValueError("Both pinned and adaptive arms must be present")

    expected = [
        (request["request_id"], int(request["sample_index"]), float(request["deadline_ms"]),
         request["resource_condition"], int(request["dataset_index"]))
        for request in workload_doc["requests"]
    ]
    for (group_id, repetition_id), run_df in df.groupby(["run_id", "repetition"], sort=True):
        arm_records = {}
        for arm in ("usm_pinned", "usm_adaptive"):
            arm_df = run_df[run_df["arm"] == arm].sort_values("sample_index")
            keys = list(zip(
                arm_df["request_id"], arm_df["sample_index"].astype(int),
                arm_df["deadline_ms"].astype(float), arm_df["scheduled_resource_condition"],
                arm_df["dataset_index"].astype(int),
            ))
            if keys != expected:
                raise ValueError(f"{group_id}/rep{int(repetition_id):03d}/{arm} does not match the frozen workload")
            arm_records[arm] = arm_df
        pinned_pairs = set(zip(arm_records["usm_pinned"]["selected_width"], arm_records["usm_pinned"]["selected_bit_width"]))
        if pinned_pairs != {(1.0, 32)}:
            raise ValueError(f"Pinned configuration was not exactly (1.0, 32) in {group_id}/rep{int(repetition_id):03d}")

    accuracy_df = pd.DataFrame(accuracy_doc["results"])
    accuracy_df.to_csv(analysis_dir / "candidate_accuracy.csv", index=False)
    profiles.to_csv(analysis_dir / "candidate_latency.csv", index=False)
    profiles.to_csv(analysis_dir / "resource_condition_latency.csv", index=False)

    per_run = []
    for repetition, run_df in df.groupby("repetition", sort=True):
        repetition = int(repetition)
        for arm, arm_df in run_df.groupby("arm"):
            end_to_end = arm_df["end_to_end_latency_ms"].to_numpy(dtype=float)
            model_latency = arm_df["model_latency_ms"].to_numpy(dtype=float)
            violation = arm_df["deadline_violation_ms"].to_numpy(dtype=float)
            misses = arm_df["missed_deadline"].to_numpy(dtype=bool)
            switch_ms = arm_df.loc[arm_df["switched"], "measured_switch_setting_ms"].to_numpy(dtype=float)
            switch_kinds = arm_df["switch_type"].value_counts().to_dict()
            selected_configs = list(zip(arm_df["selected_width"], arm_df["selected_bit_width"]))
            per_run.append({
                "run_id": f"{run_id}_rep{repetition:03d}",
                "repetition": repetition,
                "arm": arm,
                "n_requests": len(arm_df),
                "accuracy_percent": 100.0 * float(arm_df["correct"].mean()),
                "model_mean_ms": float(np.mean(model_latency)),
                "model_p50_ms": percentile(model_latency, 50),
                "model_p95_ms": percentile(model_latency, 95),
                "model_p99_ms": percentile(model_latency, 99),
                "e2e_mean_ms": float(np.mean(end_to_end)),
                "e2e_p50_ms": percentile(end_to_end, 50),
                "e2e_p95_ms": percentile(end_to_end, 95),
                "e2e_p99_ms": percentile(end_to_end, 99),
                "deadline_misses": int(misses.sum()),
                "deadline_miss_rate_percent": 100.0 * float(misses.mean()),
                "mean_violation_ms": float(np.mean(violation)),
                "p95_violation_ms": percentile(violation, 95),
                "max_violation_ms": float(np.max(violation)),
                "mean_queue_delay_ms": float(arm_df["queue_delay_ms"].mean()),
                "controller_overhead_mean_ms": float(arm_df["controller_overhead_ms"].mean()),
                "switch_count": int(arm_df["switched"].sum()),
                "switch_rate_percent": 100.0 * float(arm_df["switched"].mean()),
                "width_only_switches": int(switch_kinds.get("width_only", 0)),
                "bit_only_switches": int(switch_kinds.get("bit_only", 0)),
                "width_and_bit_switches": int(switch_kinds.get("width_and_bit", 0)),
                "switch_overhead_mean_ms": float(np.mean(switch_ms)) if len(switch_ms) else 0.0,
                "switch_overhead_p95_ms": percentile(switch_ms, 95) if len(switch_ms) else 0.0,
                "controller_feedback_overhead_ms": 0.0,
                "mean_selected_width": float(arm_df["selected_width"].mean()),
                "mean_selected_bit_width": float(arm_df["selected_bit_width"].mean()),
                "peak_process_rss_mb": float(arm_df["process_rss_mb"].max()),
                "peak_gpu_memory_mb": float(arm_df["gpu_peak_allocated_mb"].max()) if arm_df["gpu_peak_allocated_mb"].notna().any() else None,
                "max_gpu_memory_used_mb": float(arm_df["gpu_memory_used_mb"].dropna().max()) if arm_df["gpu_memory_used_mb"].notna().any() else None,
                "mean_cpu_utilization_percent": float(arm_df["observed_cpu_pct"].mean()),
                "minimum_available_memory_mb": float(arm_df["available_memory_mb"].min()),
                "mean_gpu_utilization_percent": float(arm_df["gpu_utilization_percent"].dropna().mean()) if arm_df["gpu_utilization_percent"].notna().any() else None,
                "mean_gpu_clock_mhz": float(arm_df["gpu_clock_mhz"].dropna().mean()) if arm_df["gpu_clock_mhz"].notna().any() else None,
                "max_gpu_temperature_c": float(arm_df["gpu_temperature_c"].dropna().max()) if arm_df["gpu_temperature_c"].notna().any() else None,
                "fallback_rate_percent": 100.0 * float(arm_df["fallback"].mean()) if arm == "usm_adaptive" else 0.0,
                "fallback_count": int(arm_df["fallback"].sum()) if arm == "usm_adaptive" else 0,
                "precision_fallback_count": int(arm_df["precision_fallback_used"].sum()) if arm == "usm_adaptive" and "precision_fallback_used" in arm_df else 0,
                "precision_fallback_rate_percent": 100.0 * float(arm_df["precision_fallback_used"].mean()) if arm == "usm_adaptive" and "precision_fallback_used" in arm_df else 0.0,
                "mean_fp32_feasible_candidate_count": float(arm_df["fp32_feasible_candidate_count"].dropna().mean()) if arm == "usm_adaptive" and "fp32_feasible_candidate_count" in arm_df and arm_df["fp32_feasible_candidate_count"].notna().any() else None,
                "feasible_candidate_rate_percent": 100.0 * float((arm_df["feasible_candidate_count"].fillna(0) > 0).mean()) if arm == "usm_adaptive" else None,
                "mean_feasible_candidate_count": float(arm_df["feasible_candidate_count"].dropna().mean()) if arm_df["feasible_candidate_count"].notna().any() else None,
            })
    summary = pd.DataFrame(per_run).sort_values(["repetition", "arm"])
    summary.to_csv(analysis_dir / "summary.csv", index=False)

    selection = df.groupby(["arm", "selected_width", "selected_bit_width"], dropna=False).agg(
        selections=("request_id", "count"),
        accuracy_percent=("correct", lambda x: 100.0 * x.mean()),
        mean_model_latency_ms=("model_latency_ms", "mean"),
        deadline_miss_rate_percent=("missed_deadline", lambda x: 100.0 * x.mean()),
    ).reset_index()
    full_index = pd.MultiIndex.from_product(
        [["usm_pinned", "usm_adaptive"], QAT_CANDIDATES],
        names=["arm", "config"],
    )
    selection["config"] = list(zip(selection["selected_width"], selection["selected_bit_width"]))
    selection = selection.drop(columns=["selected_width", "selected_bit_width"])
    selection = selection.set_index(["arm", "config"]).reindex(full_index).reset_index()
    selection["selected_width"] = selection["config"].map(lambda config: config[0])
    selection["selected_bit_width"] = selection["config"].map(lambda config: config[1])
    selection = selection.drop(columns=["config"])
    selection["selections"] = selection["selections"].fillna(0).astype(int)
    selection["selection_frequency_percent"] = selection.apply(
        lambda row: 100.0 * row["selections"] / max(1, int((df["arm"] == row["arm"]).sum())), axis=1
    )
    selection.to_csv(analysis_dir / "selection_frequency.csv", index=False)
    adaptive_rows = df[df["arm"] == "usm_adaptive"]
    adaptive_rows.groupby("selected_width").size().rename("selections").to_frame().assign(
        selection_frequency_percent=lambda table: 100.0 * table["selections"] / max(1, len(adaptive_rows))
    ).reset_index().to_csv(analysis_dir / "adaptive_width_distribution.csv", index=False)
    adaptive_rows.groupby("selected_bit_width").size().rename("selections").to_frame().assign(
        selection_frequency_percent=lambda table: 100.0 * table["selections"] / max(1, len(adaptive_rows))
    ).reset_index().to_csv(analysis_dir / "adaptive_bit_width_distribution.csv", index=False)

    switch_summary = df[df["arm"] == "usm_adaptive"].groupby("switch_type").agg(
        count=("request_id", "count"),
        mean_overhead_ms=("measured_switch_setting_ms", "mean"),
        p95_overhead_ms=("measured_switch_setting_ms", lambda x: percentile(x, 95)),
    ).reset_index()
    switch_summary.to_csv(analysis_dir / "switching_analysis.csv", index=False)
    deadline_summary = summary[[
        "run_id", "repetition", "arm", "deadline_misses", "n_requests",
        "deadline_miss_rate_percent", "mean_violation_ms", "p95_violation_ms", "max_violation_ms",
    ]]
    deadline_summary.to_csv(analysis_dir / "deadline_analysis.csv", index=False)

    paired_metrics = [
        "accuracy_percent", "model_p50_ms", "model_p95_ms", "model_p99_ms",
        "e2e_p50_ms", "e2e_p95_ms", "e2e_p99_ms", "deadline_miss_rate_percent",
        "mean_violation_ms", "p95_violation_ms", "max_violation_ms", "switch_rate_percent",
        "mean_selected_width", "mean_selected_bit_width", "controller_overhead_mean_ms",
        "switch_overhead_mean_ms", "mean_queue_delay_ms", "width_only_switches",
        "bit_only_switches", "width_and_bit_switches", "fallback_rate_percent",
    ]
    paired_rows = []
    paired_values = summary.pivot(index="repetition", columns="arm", values=paired_metrics)
    if paired_values.isna().any().any():
        raise ValueError("Missing one of the two arms from a complete paired repetition")
    rng = np.random.default_rng(20261002)
    for metric in paired_metrics:
        difference = paired_values[(metric, "usm_adaptive")] - paired_values[(metric, "usm_pinned")]
        boot = rng.choice(difference.to_numpy(dtype=float), size=(10000, len(difference)), replace=True).mean(axis=1)
        paired_rows.append({
            "metric": metric,
            "mean_paired_difference_adaptive_minus_pinned": float(difference.mean()),
            "median_paired_difference": float(difference.median()),
            "ci_95_lower": float(np.percentile(boot, 2.5)),
            "ci_95_upper": float(np.percentile(boot, 97.5)),
            "n_paired_repetitions": len(difference),
        })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(analysis_dir / "paired_comparison.csv", index=False)

    condition_summary = df.groupby(["scheduled_resource_condition", "arm"]).agg(
        n_requests=("request_id", "count"),
        accuracy_percent=("correct", lambda x: 100.0 * x.mean()),
        e2e_p95_ms=("end_to_end_latency_ms", lambda x: percentile(x, 95)),
        deadline_miss_rate_percent=("missed_deadline", lambda x: 100.0 * x.mean()),
        mean_selected_width=("selected_width", "mean"),
        mean_selected_bit_width=("selected_bit_width", "mean"),
        mean_cpu_percent=("observed_cpu_pct", "mean"),
        mean_gpu_percent=("gpu_utilization_percent", "mean"),
        minimum_available_memory_mb=("available_memory_mb", "min"),
        maximum_gpu_memory_used_mb=("gpu_memory_used_mb", "max"),
        maximum_gpu_temperature_c=("gpu_temperature_c", "max"),
    ).reset_index()
    condition_summary.to_csv(analysis_dir / "resource_condition_summary.csv", index=False)

    phase_summary = df.groupby(["phase", "arm"], dropna=False).agg(
        n_requests=("request_id", "count"),
        accuracy_percent=("correct", lambda x: 100.0 * float(x.mean())),
        model_p50_ms=("model_latency_ms", lambda x: percentile(x, 50)),
        model_p95_ms=("model_latency_ms", lambda x: percentile(x, 95)),
        e2e_p50_ms=("end_to_end_latency_ms", lambda x: percentile(x, 50)),
        e2e_p95_ms=("end_to_end_latency_ms", lambda x: percentile(x, 95)),
        deadline_miss_rate_percent=("missed_deadline", lambda x: 100.0 * float(x.mean())),
        mean_deadline_violation_ms=("deadline_violation_ms", "mean"),
        peak_process_rss_mb=("process_rss_mb", "max"),
        peak_gpu_memory_mb=("gpu_peak_allocated_mb", "max"),
        controller_overhead_mean_ms=("controller_overhead_ms", "mean"),
        switch_count=("switched", "sum"),
    ).reset_index()
    phase_summary.to_csv(analysis_dir / "phase_summary.csv", index=False)
    phase_switching = df[df["arm"] == "usm_adaptive"].groupby(["phase", "switch_type"], dropna=False).agg(
        switch_count=("switched", "sum"),
        request_count=("request_id", "count"),
    ).reset_index()
    phase_switching.to_csv(analysis_dir / "phase_switching.csv", index=False)

    quality = {
        "profile_holdout_validation": validation,
        "adaptive_fallback_rate_percent": float(df.loc[df["arm"] == "usm_adaptive", "fallback"].mean() * 100.0),
        "adaptive_precision_fallback_rate_percent": float(df.loc[df["arm"] == "usm_adaptive", "precision_fallback_used"].mean() * 100.0) if "precision_fallback_used" in df else None,
        "adaptive_mean_feasible_candidates": float(df.loc[df["arm"] == "usm_adaptive", "feasible_candidate_count"].mean()),
        "adaptive_selection_count": int((df["arm"] == "usm_adaptive").sum()),
    }
    (analysis_dir / "controller_quality.json").write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8")

    result_lines = [
        "# USM Full Versus Adaptive USM Results", "",
        "Paired repetitions are the independent units; request rows are not treated as independent replicates.",
        "Latency is PyTorch fake-quantized execution, not integer INT4/INT8 kernel performance.", "",
        "| Metric | USM Full fixed | USM Adaptive | Adaptive - Full (95% paired bootstrap CI) |",
        "| --- | ---: | ---: | ---: |",
    ]
    arm_means = summary.groupby("arm").mean(numeric_only=True)
    for _, row in paired.iterrows():
        metric = row["metric"]
        result_lines.append(
            f"| {metric} | {arm_means.loc['usm_pinned', metric]:.4f} | "
            f"{arm_means.loc['usm_adaptive', metric]:.4f} | "
            f"{row['mean_paired_difference_adaptive_minus_pinned']:.4f} "
            f"[{row['ci_95_lower']:.4f}, {row['ci_95_upper']:.4f}] |"
        )
    pinned_config = json.loads((run_root / "config.json").read_text(encoding="utf-8"))
    result_lines.extend([
        "", f"Pinned configuration (fixed, not calibrated): `({pinned_config['width_mult']:.2f}, {pinned_config['bit_width']})`.",
        (f"Held-out profile prediction validation: MAE={validation['mae_ms']:.4f} ms, RMSE={validation['rmse_ms']:.4f} ms."
         if "mae_ms" in validation else f"Held-out profile prediction validation: {validation['status']}.") ,
        "",
    ])
    (analysis_dir / "result_table.md").write_text("\n".join(result_lines), encoding="utf-8")

    adaptive_first = df[(df["repetition"] == 1) & (df["arm"] == "usm_adaptive")].sort_values("sample_index")
    label_map = {config: index for index, config in enumerate(QAT_CANDIDATES)}
    plt.figure(figsize=(12, 4))
    phase_colors = {"phase_1_relaxed": "#dcefe8", "phase_2_constrained": "#f6e3d8", "phase_3_recovery": "#e4e9f2"}
    for phase, subset in adaptive_first.groupby("phase", sort=False):
        if subset.empty:
            continue
        plt.axvspan(subset["sample_index"].min() - 0.5, subset["sample_index"].max() + 0.5,
                    color=phase_colors.get(phase, "#eeeeee"), alpha=0.75)
    plt.step(adaptive_first["sample_index"], [label_map[(w, b)] for w, b in zip(adaptive_first["selected_width"], adaptive_first["selected_bit_width"])], where="post", color="#172b4d", linewidth=1.5)
    plt.yticks(range(len(QAT_CANDIDATES)), [str(config) for config in QAT_CANDIDATES])
    plt.xlabel("Request index")
    plt.ylabel("Selected (width, bits)")
    plt.tight_layout()
    plt.savefig(analysis_dir / "configuration_selection.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 5))
    profile_df = pd.read_csv(run_root / "profiles" / "resource_condition_profiles.csv")
    config_names = [f"({width:g},{bits})" for width, bits in QAT_CANDIDATES]
    x_values = np.arange(len(QAT_CANDIDATES))
    conditions = list(profile_df["resource_condition"].drop_duplicates())
    bar_width = 0.8 / max(1, len(conditions))
    adaptive_selected = set(zip(adaptive_first["selected_width"], adaptive_first["selected_bit_width"]))
    for condition_index, condition in enumerate(conditions):
        values = profile_df[profile_df["resource_condition"] == condition].set_index(["width_mult", "bit_width"])
        offsets = x_values - 0.4 + bar_width * (condition_index + 0.5)
        p95_values = [float(values.loc[config, "p95_ms"]) for config in QAT_CANDIDATES]
        bars = ax.bar(offsets, p95_values, width=bar_width, label=condition)
        for bar, config in zip(bars, QAT_CANDIDATES):
            if config in adaptive_selected:
                bar.set_edgecolor("#c4472d")
                bar.set_linewidth(2.0)
    ax.set_xticks(x_values, config_names, rotation=45, ha="right")
    ax.set_ylabel("Measured model P95 latency (ms)")
    ax.set_xlabel("Candidate (width, bit-width); outlined bars were selected")
    ax.legend()
    fig.tight_layout()
    fig.savefig(analysis_dir / "candidate_latency_selections.png")
    plt.close(fig)

    overall = df.groupby("arm").agg(
        model_p95_ms=("model_latency_ms", lambda x: percentile(x, 95)),
        deadline_miss_rate_percent=("missed_deadline", lambda x: 100.0 * float(x.mean())),
    ).reindex(["usm_pinned", "usm_adaptive"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    labels = ["USM Full fixed", "USM Adaptive"]
    axes[0].bar(labels, overall["model_p95_ms"], color=["#486f73", "#d17b49"])
    axes[0].set_ylabel("Model P95 latency (ms)")
    axes[1].bar(labels, overall["deadline_miss_rate_percent"], color=["#486f73", "#d17b49"])
    axes[1].set_ylabel("Deadline miss rate (%)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(analysis_dir / "fixed_full_vs_adaptive.png")
    plt.close(fig)

    print(f"QAT paired report generated: {analysis_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate all report artifacts from archived USM QAT raw results.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    create_report(args.run_id, args.run_root, args.output_dir)


if __name__ == "__main__":
    main()