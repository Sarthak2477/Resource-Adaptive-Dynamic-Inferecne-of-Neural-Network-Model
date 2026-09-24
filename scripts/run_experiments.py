"""Run repeated and held-out controller evaluations with paired seeds."""

import argparse
import json
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--policies", nargs="+", default=["p95", "p99"])
    parser.add_argument("--traces", nargs="+", default=["sinusoidal", "heldout"])
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results = []
    for repetition in range(args.repetitions):
        seed = 12345 + repetition
        for policy in args.policies:
            for trace in args.traces:
                command = [
                    sys.executable,
                    os.path.join(project_root, "scripts", "evaluate.py"),
                    "--policy", policy,
                    "--trace", trace,
                    "--seed", str(seed),
                    "--threads", str(args.threads),
                ]
                print(f"Running repetition={repetition} policy={policy} trace={trace}")
                subprocess.run(command, cwd=project_root, check=True)
                result_path = os.path.join(
                    project_root,
                    "results",
                    f"evaluation_{policy}_{trace}_seed{seed}_threads{args.threads}.json",
                )
                with open(result_path, encoding="utf-8") as result_file:
                    result = json.load(result_file)
                results.append({
                    "repetition": repetition,
                    "seed": seed,
                    "policy": policy,
                    "trace": trace,
                    "miss_rate_percent": result["deadline_miss_rate_percent"],
                    "feasible_miss_rate_percent": result["feasible_budget_miss_rate_percent"],
                    "infeasible_rate_percent": result["infeasible_budget_rate_percent"],
                    "accuracy_percent": result["accuracy_percent"],
                    "p95_latency_ms": result["p95_latency_ms"],
                    "p99_latency_ms": result["p99_latency_ms"],
                    "cold_transition_misses": result["cold_transition_misses"],
                    "warm_misses": result["warm_misses"],
                })

    output_path = os.path.join(project_root, "results", "experiment_summary.json")
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)
        output_file.write("\n")
    print(f"Experiment summary written to: {output_path}")


if __name__ == "__main__":
    main()