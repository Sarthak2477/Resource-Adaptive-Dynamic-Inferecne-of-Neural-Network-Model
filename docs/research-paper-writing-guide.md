# Research Paper Writing Guide

## Purpose

This guide turns the current repository into a research-paper workflow. It is written for an IEEE-style technical paper, but the research and evidence guidance is venue-neutral. It is not a claim that the current experiments are publication-ready: it distinguishes results that exist from claims that still require controlled experiments.

The working topic is adaptive neural-network inference under changing latency budgets. The project explores choosing among width/precision configurations and correcting latency estimates online. Start with the technical formulation in [`formulation.md`](../formulation.md), then use this guide to audit the implementation, design the evaluation, and draft the paper.

## 1. State the Research Question

Write one question that can be answered by the planned experiments. A suitably narrow starting question is:

> Under changing inference-latency budgets, how does selecting the highest-accuracy feasible subnet using empirical latency profiles and per-configuration online correction affect accuracy, deadline misses, and latency compared with static and alternative selection policies?

This wording reflects the current prototype more accurately than claiming generalized, live hardware-aware prediction. The current saved GPU evaluations construct a `MeasuredFrontierLatencyModel` from the device profile; they do not evaluate the trained Random Forest as the runtime predictor. See [`scripts/evaluate.py`](../scripts/evaluate.py) and [`resource_control/controller.py`](../resource_control/controller.py).

### Define terms before using them

- **Configuration / subnet:** a width multiplier and bit-width pair, currently drawn from four widths and four bit widths. State the exact candidate set used in each experiment.
- **Latency budget:** the maximum allowed time per request, in milliseconds.
- **Deadline miss:** measured latency greater than the request budget. Define whether preprocessing, model/configuration switching, batch-normalization calibration, controller work, and data transfer are included.
- **Infeasible budget:** a budget below the best achievable latency under a clearly stated and independently applied feasibility rule. Do not define infeasibility solely from a policy's own prediction and then treat the resulting feasible-only miss rate as an independent guarantee.
- **Percentile policy:** state which latency distribution and observations produce P50/P95/P99, when that estimate is used, and whether cold transitions use the same quantile.
- **Online calibration:** identify the exact update equation, update interval, clipping bounds, initialization, and whether the factor is maintained per configuration.

### Separate hypotheses from contributions

A hypothesis is tested; a contribution is something the work provides. Keep both measurable.

Possible contribution framing, subject to completed experiments:

1. A controller that selects among deployable model configurations under explicit request-level latency budgets.
2. A per-configuration online latency correction rule and a feasibility-aware selection procedure.
3. An empirical evaluation comparing static, mean-profile, percentile-profile, and calibrated policies under controlled budget and load traces.

Do not claim that a contribution is novel until related-work review supports that statement. Do not claim statistical deadline guarantees based only on empirical profile quantiles.

## 2. Audit the Evidence Before Drafting Results

Create a results ledger before writing prose. Every result should be traceable to a command, configuration, machine-readable output, and experimental protocol.

### Current saved evaluation snapshots

The current P95 and P99 JSONs each evaluate 100 samples using the sinusoidal trace and seed 12345 on CUDA/GTX 1650. Their recorded device-speed fingerprints differ, so these are separate descriptive runs, not a paired policy comparison.

| Saved run | Accuracy | Mean latency | P95 latency | P99 latency | Prediction MAE | Total misses | Infeasible budgets | Feasible-tagged misses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P95 policy | 94.0% | 16.95 ms | 32.22 ms | 53.83 ms | 2.54 ms | 29/100 (29%) | 26/100 (26%) | 6/74 (8.11%) |
| P99 policy | 94.0% | 15.18 ms | 33.60 ms | 41.02 ms | 2.36 ms | 27/100 (27%) | 29/100 (29%) | 4/71 (5.63%) |

Sources: [`P95 evaluation JSON`](../results/evaluation_p95_sinusoidal_seed12345_threads1.json) and [`P99 evaluation JSON`](../results/evaluation_p99_sinusoidal_seed12345_threads1.json). These are rounded from the stored summaries. Use exact precision in analysis code, and sensible rounding only in paper tables.

Safe description at this stage:

> In two saved 100-sample GTX 1650 runs, the P95 and P99 configurations each achieved 94% subset accuracy. The observed overall deadline-miss rates were 29% and 27%, respectively. Because the recorded hardware-speed fingerprints differ and there is only one saved run per policy, these values are descriptive and do not establish a comparative effect.

Do not state that P99 outperforms P95, that either policy reduces misses, or that the controller meets deadlines. The feasible-only rates depend on the evaluation's own feasibility labels and should not be interpreted as a guarantee.

### Reconcile conflicting artifacts

Before using any project number in the paper:

1. Compare README headline results with the detailed evaluation JSONs and identify the exact run behind each value.
2. Compare README latency examples with the checked-in device profile. The README's GTX 1650 examples conflict with the profile values reported by the project audit; regenerate or correct them.
3. Verify the checkpoint, dataset split, model/configuration accuracy values, and profile used by each result. The evaluation currently uses a hard-coded configuration accuracy table; document the source and evaluation conditions for that table.
4. Record the exact source revision, command-line arguments, random seed, thread count, device, driver/runtime versions, and profile timestamp for every run.
5. Preserve raw per-sample outputs and scripts alongside aggregated tables. Never copy a number into the manuscript without its provenance.

The older CPU result in [`evaluation_results.json`](../results/evaluation_results.json) has a different schema and execution context. Do not combine it with the GPU table as if it were a controlled cross-device comparison.

### Implementation facts and limits to disclose

Check these details in code before describing the final system:

- [`scripts/evaluate.py`](../scripts/evaluate.py) evaluates a balanced 100-example subset, not the full CIFAR-10 test set. The saved 94% accuracy is therefore subset accuracy, not full-test accuracy.
- The current evaluation uses `MeasuredFrontierLatencyModel`, a lookup over the measured profile. The selected prediction does not adapt to changing telemetry in the way a validated learned surrogate should. The README's RF-backed description should not be repeated as an evaluated result until that path is actually used and tested.
- [`scripts/train_surrogate.py`](../scripts/train_surrogate.py) synthesizes load examples by multiplying measured latency by fixed factors and reports metrics on the training data. Those metrics are not held-out predictive validation.
- The evaluation simulates CPU/temperature contention labels; it does not inject and measure controlled GPU contention. Explain that distinction.
- The measured latency is the model forward pass; lazy batch-normalization recalibration occurs before the timer starts. Controller overhead and end-to-end request latency are not represented by that timing.
- Profiles use empirical observations. An observed P95/P99 is not a confidence bound or a statistical service-level guarantee. State the sample count, warm-up procedure, and uncertainty.
- In `controller.select`, feasibility is based on the minimum warm profile latency, while selection also applies a safety margin and switching penalty. In the evaluation, selected-candidate percentile handling differs for a candidate that is already active versus one requiring a switch. Resolve these definitions or explicitly report the existing behavior as a limitation before drawing policy conclusions.

## 3. Design the Experiments That Can Answer the Question

Do this before finalizing the Introduction. A paper's claims, baselines, and metrics must be determined by the experiment plan rather than inferred after inspecting favorable values.

### Comparison policies

At minimum, compare the proposed policy against:

1. **Static fastest:** always use the measured fastest valid configuration.
2. **Static highest accuracy:** always use the highest-accuracy configuration.
3. **Mean-profile selector:** choose the most accurate configuration feasible under mean latency estimates.
4. **Percentile selector:** choose using the specified empirical percentile, without online correction.
5. **Calibrated percentile selector:** proposed policy with per-configuration online correction.
6. **Oracle reference (optional):** choose using latency known from held-out measurements, clearly labeled as an upper-bound reference and not a deployable method.

Keep the input samples, request budgets, load trace, hardware, warm-up, and random seeds paired across policies. A policy comparison is only meaningful when the non-policy conditions are held constant. Ensure the evaluation script truly applies each named policy consistently to all candidates and switching states before interpreting its output.

### Workload and repetition design

- Use the full test set for final classification accuracy, or justify the subset and report uncertainty. A 100-sample subset is suitable for smoke testing, not a stable estimate of CIFAR-10 accuracy.
- Use several budget traces: sinusoidal, step changes, bursty budgets, and a held-out trace. Describe the formula and parameters, not only the trace name.
- Distinguish synthetic traces from measured workload traces. If making real-world contention claims, inject or measure controlled concurrent workloads and log actual telemetry.
- Run paired repetitions across multiple seeds and, where possible, multiple independent profiling sessions. Fix a thread count and device software configuration.
- Report hardware-specific findings separately. To claim generalization, evaluate on held-out devices or use a clearly specified cross-device protocol.
- Partition profile/training data and evaluation data so that latency model evaluation is not measured on the same observations used for fitting or tuning.
- Record cold-start, warm steady-state, and configuration-transition behavior separately. Decide whether the target application includes startup and model recalibration costs.

[`scripts/run_experiments.py`](../scripts/run_experiments.py) is intended to run multiple seeds, policies, and traces. Before launching a long run, check its output naming and overwrite behavior, confirm each evaluation actually exercises the intended policy, and save a manifest of command lines and environment details.

### Ablations

Run component ablations with all other conditions fixed:

- Remove online calibration.
- Replace percentile estimates with means.
- Remove switching penalty / dwell-time constraint.
- Compare lookup-based predictions with a genuinely evaluated learned predictor.
- Remove telemetry features or hold them constant to determine whether telemetry contributes useful predictive information.
- Compare cold-transition and warm-only evaluation.

An ablation supports a mechanism claim only if its results are repeated and uncertainty is reported.

## 4. Select Metrics and Define Them Precisely

Use metrics that jointly describe prediction, model quality, deadlines, and adaptation. Define each in the paper and ensure the analysis code uses exactly the same denominator and measurement boundary.

| Metric | Recommended definition / reporting |
|---|---|
| Accuracy | Correct predictions divided by evaluated examples; report test-set size and uncertainty. If selections vary across examples, state that this is controller-selected accuracy. |
| Latency | Mean, median, P95, and P99 measured latency; state whether forward-only or end-to-end. Include cold and warm values where relevant. |
| Overall deadline-miss rate | Count of measured latencies greater than request budgets divided by all evaluated requests. Report count and denominator as well as percent. |
| Infeasible-budget rate | Requests whose budgets are independently below a measured achievable minimum divided by all requests. Specify how the minimum is estimated. |
| Feasible-only miss rate | Misses among independently classified feasible requests divided by independently classified feasible requests. Do not use a policy-specific prediction as the only feasibility oracle. |
| Violation magnitude | For missed requests, report mean and distribution of `max(0, actual latency - budget)`, in milliseconds. Define whether infeasible requests are included. |
| Prediction error | MAE and optionally RMSE, computed on held-out or runtime predictions; report prediction sample count and separate cold/warm performance. |
| Switching rate | Configuration changes divided by transitions between consecutive requests. Also report number of switches and switching latency if measured. |
| Controller overhead | Selection/update time, memory, and any additional energy if the application makes resource use a claim. |
| Uncertainty | Across independent paired runs: mean and confidence interval or median and interquartile range; use an analysis appropriate to the experimental unit. |

For a per-request record $t$, define `miss_t = 1[latency_t > budget_t]`. Then overall miss rate is `sum(miss_t) / T`. For feasible-only miss rate, use a separately defined indicator `feasible_t`, and report `sum(miss_t * feasible_t) / sum(feasible_t)` together with both counts. Avoid calling an empirical percentile a guarantee.

## 5. Plan the Tables and Figures

Design the evidence display before drafting Results. Each figure should answer one question, and its caption should state the workload, hardware, sample count, and aggregation method.

### Suggested result tables

**Table A: Experimental environment.** Device model, CPU, memory, OS, runtime/library versions, precision support, thread count, profile date/sample count, model checkpoint, dataset split, and measurement boundary.

**Table B: Main policy comparison.** One row per policy and workload. Include accuracy, mean/P50/P95/P99 latency, total misses (`count / N`), independent infeasible rate, feasible-only misses (`count / feasible N`), prediction MAE, switching rate, and controller overhead. Include uncertainty across paired repetitions.

**Table C: Ablation study.** One row per component configuration; retain the primary deadline and accuracy outcomes and show paired deltas against the complete policy.

**Table D: Per-configuration profile.** Configuration, measured accuracy, profile observation count, mean/std/P50/P95/P99 latency, and cold-transition latency. Clearly identify which values are measured and which are estimated.

Do not fill a publication table with blank or invented values. Keep a working table with `TBD` fields until experiments complete, then replace those fields with computed outputs and uncertainty.

### Suggested figures

1. **Accuracy-latency frontier:** each deployable configuration as a point, with accuracy versus measured latency; identify the selected configurations.
2. **Budget and actual latency over time:** budget and measured latency on the same axis, with infeasible intervals and policy switches marked. Use representative traces only as illustrations; summarize all repetitions elsewhere.
3. **Miss rate versus budget regime:** compare policy miss rates across budget ranges and report counts/uncertainty.
4. **Prediction calibration:** predicted versus observed latency on held-out measurements, plus residuals; separate warm/cold and load conditions.
5. **Accuracy versus deadline compliance:** plot accuracy against miss rate for each policy and show run-to-run variation.
6. **Ablation effects:** paired change in miss rate and accuracy when each mechanism is removed.

Avoid overloaded plots, unexplained color, truncated axes that exaggerate effects, and plotting individual requests as independent experimental replicates when the run is the actual independent unit.

## 6. Draft the Paper in a Productive Order

A useful writing order differs from the reading order. Draft the Methods and Experimental Setup after the protocol is fixed, then Results, then Introduction and Abstract. This reduces the temptation to promise results the experiments do not test.

### 6.1 Title

Use a literal title that states the method and setting without asserting success. Possible working title:

> Percentile-Based Configuration Selection for Neural Inference Under Dynamic Latency Budgets

Revise it after the experiments establish the real contribution. Do not put “guaranteed,” “optimal,” or “general” in the title unless those terms are formally justified and empirically supported.

### 6.2 Abstract (write last)

Use approximately five pieces, adapted to the venue's word limit:

1. **Problem:** deployment must trade model quality against changing inference deadlines.
2. **Gap:** identify the narrowly evidenced limitation in the closest prior approaches, supported by related work.
3. **Method:** state the configuration set, latency estimator, online correction, and selection rule.
4. **Evaluation:** name datasets, hardware, traces, baselines, repetitions, and measurement boundary.
5. **Result and implication:** report the primary quantitative result with uncertainty and the scope it supports.

Do not write the final result sentence until the controlled experiment summary exists. The current 94% values are 100-example subset accuracy, not full-test accuracy.

### 6.3 Introduction

Draft four or five paragraphs:

1. Explain the practical setting: the inference budget varies and model configurations have different quality/cost.
2. Define the technical problem and why average latency may be insufficient for tail-sensitive deadlines.
3. Identify the literature gap only after reviewing and citing the relevant work.
4. State the research question and a short overview of the proposed controller.
5. List two or three contributions, each verifiable in a later section.

Close with a scope sentence that limits claims to the tested model, hardware, traces, and measurement boundary.

### 6.4 Related Work

Organize by concepts rather than one paragraph per paper:

- slimmable/dynamic neural networks and early-exit or adaptive-compute methods;
- quantization-aware training and multi-precision inference;
- latency estimation, hardware profiling, and performance modeling;
- deadline-aware scheduling/control and tail-latency methods.

For each group, explain the shared problem, representative approaches, their evaluation conditions, and the precise difference from this project. Use primary sources and verify bibliographic details. Do not describe a method as novel merely because it differs from this repository's code.

### 6.5 Method

Describe enough for an independent implementation:

1. Model architecture, training procedure, checkpoint, available configurations, and how per-configuration accuracy is obtained.
2. Configuration features and hardware/profile data schema.
3. Latency estimator, training data, preprocessing, train/test split, and inference path actually used in each experiment.
4. Online calibration equation, initialization, learning rate, clipping, per-configuration state, and update timing.
5. Selection rule, safety factor, risk multiplier, switching cost, dwell constraints, tie-breaking, and fallback when no candidate is feasible.
6. Independent feasibility classification and how infeasible requests are handled in every metric.
7. Runtime measurement boundary, synchronization, warm-up, and treatment of cold configuration transitions.

Use pseudocode for the control loop and equations for the key decision/update. Keep proposed variants separate from the implemented/evaluated method.

### 6.6 Experimental Setup

Provide a reproducibility table, then detail:

- dataset, preprocessing, exact test split/subset, class balance, and sample count;
- checkpoint and training/evaluation accuracy method;
- hardware and software versions, clock/power settings if controlled, and thread count;
- latency profile measurement procedure, sample counts, warm-up, and percentile estimator;
- budget traces and load generation, with held-out workload details;
- policy baselines, paired seeds, repetitions, and any hyperparameter selection process;
- metrics, independent experimental unit, uncertainty method, and measurement boundaries.

Explain exclusions. For example, if timing excludes batch-normalization recalibration, say so explicitly and report its cost separately if startup latency matters.

### 6.7 Results

Present results in this sequence:

1. Confirm profile/model validity and report held-out estimator performance.
2. Show the main paired policy comparison and uncertainty.
3. Explain accuracy-latency trade-offs and configuration choices.
4. Analyze total misses, independent feasibility, feasible-only misses, and violation magnitude.
5. Report cold transitions, switching rate, controller overhead, and trace behavior.
6. Present ablations and robustness/generalization results.

For each table/figure, write: (a) the observed result, (b) the size and uncertainty of the difference, and (c) the narrow interpretation supported. Avoid causal wording if only an uncontrolled association is measured. Do not repeat every table value in prose.

### 6.8 Discussion and Limitations

Discuss what the results mean for the deployment question and when a system operator might use this approach. Explicitly cover:

- empirical percentiles are not guarantees;
- simulated contention versus measured contention;
- device and model coverage;
- subset versus full-test accuracy;
- profile drift and cold-start behavior;
- timing that excludes controller and calibration overhead;
- limits of hard-coded accuracy values or synthetic surrogate training data;
- sensitivity to safety margin, profile size, and workload distribution.

Limitations should be concrete, not a generic statement that more work is needed.

### 6.9 Conclusion

Answer the research question in a few sentences, state the strongest measured finding with uncertainty, and name the setting where it applies. Do not introduce new results or broader claims. If the results remain preliminary, call them a prototype evaluation and specify the next validation step.

## 7. Claim-Support Rules

Use this checklist while drafting and again before submission:

- Every quantitative sentence points to a traceable experiment and table/figure.
- Every comparison is paired or its uncontrolled differences are disclosed.
- “Reduces” is used only for a supported comparison with uncertainty, not a comparison of unrelated runs.
- “Guarantees” is reserved for a formal guarantee with stated assumptions; empirical P95/P99 alone do not qualify.
- “Hardware-aware” or “telemetry-aware” is supported by experiments showing the runtime predictor consumes changing hardware state and benefits from doing so.
- “Generalizes” is supported by held-out hardware/workloads rather than multiple policies on one machine.
- “Real-time” is supported by end-to-end timing and a relevant system requirement.
- “Optimal” is reserved for a defined objective and demonstrated optimizer/oracle conditions; otherwise say “highest-accuracy among tested feasible configurations.”
- Accuracy is labeled subset or full-test accuracy, and its sample count is given.
- Infeasibility uses an independent, consistent definition; both its denominator and feasible-only miss denominator are visible.
- Any null, negative, or mixed result is reported rather than omitted.

## 8. Prioritized Work Plan

Complete these in order; avoid polishing prose around a result that may change.

1. **Freeze provenance:** identify the checkpoint, accuracy source, data split, profile, machine/software, and exact commands for each saved result.
2. **Reconcile documentation:** replace or annotate README figures that do not match machine-readable results and measured profiles.
3. **Fix experimental semantics:** make the selected percentile consistent for warm and cold candidates; align feasibility classification with safety margin, transition cost, and the independent feasibility measure used for reporting.
4. **Validate the actual predictor:** evaluate the RF on held-out measured latency data, or keep the paper's claims explicitly about measured lookup profiles. Do not use training-set metrics as test performance.
5. **Run paired experiments:** use the same hardware, profile, samples, budgets, and seeds across policies; collect multiple independent repetitions and preserve raw outputs.
6. **Broaden workload evidence:** evaluate held-out traces and controlled measured load. Add hardware only if cross-device claims are intended.
7. **Measure overhead and startup:** include controller, transfer, switching, and batch-normalization calibration costs or explicitly keep claims to forward-pass latency.
8. **Compute uncertainty and generate tables/figures from scripts:** avoid hand-calculated final values; archive code and manifests.
9. **Draft Methods and Setup, then Results, Discussion, Introduction, and Abstract.**
10. **Conduct a claim audit:** ask a colleague to map each abstract/introduction claim to a method, result, and limitation.

## 9. Detailed Experiment Execution Guide

This section is the experiment runbook. Run experiments in the order below: the integrity and profile checks determine whether the later controller comparison is interpretable. The commands assume PowerShell is open at the repository root and the project's Python environment is activated. Replace `python` with the selected interpreter command if the environment requires it. Do not launch the full matrix until the checkpoint, profile, and evaluation semantics have passed the gates below.

### Experiment 0: Artifact and Evaluation Smoke Check

**Question:** Does the evaluation use the intended trained checkpoint, data, hardware profile, and policy, and can a short run complete reproducibly?

**Why first:** `scripts/evaluate.py` warns and continues with randomly initialized weights if no checkpoint is found. It also selects a device profile by detected hardware name. Either condition can produce plausible-looking but invalid results.

**Procedure:**

1. Confirm that the checkpoint and expected profile exist. Inspect the checkpoint location rather than assuming the README's example path is present:

	```powershell
	Get-ChildItem models/checkpoint -File
	Get-ChildItem profiles -Filter '*_profile.csv'
	```

2. Confirm the selected Python environment has the project dependencies. In particular, the current controller tests import `psutil` and the evaluation requires PyTorch, NumPy, and the project's data/model dependencies. Install the project's documented dependencies from [`README.md`](../README.md) into the selected environment before proceeding; do not interpret an import failure as a controller test failure.
3. Run the controller unit tests:

	```powershell
	python -m unittest discover -s tests -p "test_resource_control.py" -v
	```

4. Run one small evaluation using the current supported arguments:

	```powershell
	python scripts/evaluate.py --policy p95 --trace sinusoidal --seed 12345 --threads 1
	```

5. Confirm in the console that the expected checkpoint was loaded, the intended device/profile was selected, the evaluation used measured deployable configurations, and no random-weight warning appeared.
6. Inspect the emitted JSON under `results/`. Verify its `device`, `hardware_profile`, `hardware_fingerprint`, `total_samples`, `seed`, `trace`, `latency_policy`, and `thread_count` fields. Retain this as a smoke artifact, not a result for the paper.
7. If any check fails, stop. Resolve missing data/checkpoint/profile or code-path mismatch before running subsequent experiments.

**Record:** source revision, checkpoint path and checksum, selected profile, command, machine/software details, and smoke JSON path.

### Experiment 1: Per-Configuration Accuracy Validation

**Question:** What classification accuracy does each deployable width/precision configuration achieve on a clearly defined evaluation set?

**Current limitation:** `scripts/evaluate.py` still embeds accuracy values in a constant table and evaluates a fixed balanced subset of 100 examples. The standalone [`scripts/evaluate_config_accuracy.py`](../scripts/evaluate_config_accuracy.py) now performs this experiment across every deployable configuration on the complete 10,000-image test split, while the controller evaluation remains a separate subset-based experiment.

**Procedure:**

1. Freeze the checkpoint and record its cryptographic checksum. Confirm the data split and preprocessing match the intended CIFAR-10 test protocol.
2. Run the dedicated evaluator from the repository root. Either pass the trained checkpoint explicitly or allow the script to search the project's known checkpoint locations:

	 ```powershell
	 python scripts/evaluate_config_accuracy.py `
		 --checkpoint models/checkpoint/us_resnet_epoch100_checkpoint.pt.zip `
		 --device cuda `
		 --batch-size 256 `
		 --bn-calibration-batches 100 `
		 --seed 12345 `
		 --output results/config_accuracy_seed12345.json
	 ```

	 Remove `--device cuda` to auto-select CUDA when available and otherwise use CPU. Use `--deterministic` to request deterministic PyTorch algorithms; the script will fail if an operation lacks a deterministic implementation. The evaluator refuses a non-10,000-image test set unless `--allow-nonstandard-test-size` is deliberately supplied, and it refuses to overwrite either output file.
3. The evaluator iterates over every pair in `FLAGS.deploy_configs`; it does not change configuration within a test-set pass. For each configuration it seeds the train loader, recalibrates BN using only the training split, then evaluates the test split in inference mode.
4. The script emits one JSON file with full per-class results and confusion matrices, plus a sibling CSV containing one row per configuration, top-1 accuracy, 95% Wilson interval, sample count, and per-class accuracy values.
5. Verify the JSON records exactly 10,000 test samples, all 16 expected configurations, the intended checkpoint path and SHA-256, class mapping, seed, device, BN calibration split/batch count, and software versions. Check that each configuration's per-class totals sum to the test-set size and that the confusion-matrix sum is 10,000.
6. Compare the measured values against `CONFIG_ACCURACIES` in `scripts/evaluate.py`. Resolve discrepancies and update the source table from the verified artifact, not by hand-editing the paper alone.
7. Run a second invocation with a different output filename if nondeterministic kernels or augmentation during BN calibration are being studied. Treat evaluations on the same fixed test set as repeated measurements of implementation variability, not independent test examples; the Wilson interval describes test-set binomial uncertainty, not variability across training/checkpoint seeds.
8. Use the JSON/CSV artifacts as the source for the accuracy-versus-latency plot and controller frontier. Preserve both with the model checkpoint and profile provenance.

The evaluator requires the repository's PyTorch, torchvision, NumPy, and related project dependencies. It fails rather than silently using randomly initialized weights if no checkpoint is available. Do not report the controller's current subset accuracy as full-test accuracy.

### Experiment 2: Repeated Device Latency Profiling

**Question:** What are the warm and transition latency distributions for every measured configuration on the target device?

**Current command:**

```powershell
python scripts/profile_hardware.py --threads 1
```

The profiler requires the trained checkpoint by default and writes unique artifacts under `profiles/raw/`; it does not overwrite the shared device profile. By default it records 200 warm observations per configuration, then measures all 240 ordered transitions between distinct configurations, with 10 first-call and immediate warm observations per ordered pair. The raw CSV contains every timed observation; separate CSVs contain per-configuration and per-transition distributions, and a JSON manifest records the checkpoint hash and available device/runtime metadata.

**Procedure:**

1. Close unrelated GPU/CPU workloads, connect the machine to its normal power source, and record power mode, device temperature, driver/runtime versions, and thread count. Keep these conditions fixed across the policy comparison.
2. Confirm Experiment 0's checkpoint is loaded by the profiler. Do not profile randomly initialized weights if the evaluated model is trained.
3. Preserve the checked-in or prior profile as a dated provenance artifact:

	```powershell
	New-Item -ItemType Directory -Force profiles/raw | Out-Null
	$stamp = Get-Date -Format yyyyMMdd_HHmmss
	Copy-Item profiles/nvidia_geforce_gtx_1650_profile.csv "profiles/raw/gtx1650_before_$stamp.csv"
	```

	Replace the filename with the actual detected profile name on the current machine.
4. Run three independent processes with unique session IDs. Keep the machine on the same power mode and power source, close unrelated workloads, and restart the process between sessions. Replace the example power-mode text with the actual OS/device setting:

	```powershell
python scripts/profile_hardware.py --threads 1 --session-id session01 --power-mode "AC, Best performance"
python scripts/profile_hardware.py --threads 1 --session-id session02 --power-mode "AC, Best performance"
python scripts/profile_hardware.py --threads 1 --session-id session03 --power-mode "AC, Best performance"
	```

	Each run prints exact paths for raw observations, config summaries, the transition matrix, and metadata. The metadata captures the checkpoint SHA-256, thread count, PyTorch/CUDA versions, and (when `nvidia-smi` is available) driver, GPU temperature, and power readings. The supplied power-mode string is recorded verbatim. Keep the four artifacts together for every session.
5. For every configuration, report warm sample count, mean, standard deviation, median, P95, P99, minimum, and maximum. The config summary also reports pooled incoming first-transition statistics. The transition summary has the same statistics separately for every `previous_config -> next_config` pair and phase (`first` or immediate `warm`).
6. Treat P99 from 200 warm samples, and especially P99 from 10 per-pair transition samples, as noisy empirical estimates, not service-level guarantees. Inspect device-speed fingerprint and distributions across sessions before pooling; if sessions vary materially, identify/control the cause or include session as a source of variation.
7. Select one frozen profile for the primary paired policy evaluation. State whether it is a single session or a predeclared aggregation of profiling sessions. For the existing controller/profile interface, use a selected session's `*_config_summary.csv` as the profile input; retain the raw and transition files alongside it. Do not profile on evaluation samples or tune policy parameters on held-out evaluation sessions.

**Interpretation:** A transition trial runs the source configuration, changes to the distinct target configuration, times the target's first call, then times one immediate warm call. The per-pair transition-warm distribution therefore has 10 observations by default; the 200-sample warm distribution per configuration is the better-supported estimate of steady-state latency. The destination's legacy `cold_latency_*` summary columns now describe pooled first calls from distinct source configurations, not same-config reapplications.

### Experiment 3: Held-Out Latency-Predictor Validation

**Question:** Does the latency estimator predict unseen measurements, including unseen load states or devices, accurately enough to support selection?

**Current limitation:** `scripts/train_surrogate.py` creates synthetic high-load targets by multiplying observed latency by fixed factors and reports training-set MAE/MAPE. This is not held-out validation and does not demonstrate response to real contention. The saved controller evaluations use a measured lookup model, not this Random Forest.

**Procedure:**

1. Collect raw latency observations across configuration, device, profiling session, CPU load, thermal state, memory state, and relevant hardware fingerprint. Current CSV profiles largely contain baseline telemetry and summaries, so additional raw-observation collection is required for a real load-conditioned model.
2. Split data by independent session or device, not by randomly splitting rows from the same profile. Keep all observations from a held-out session/device out of training and hyperparameter tuning.
3. Train the candidate estimator only on training partitions. Fit preprocessing and feature normalization on training data only. Save the trained model, training-data manifest, feature list, and random seed.
4. Evaluate on held-out raw measurements. Report MAE, RMSE, median absolute error, signed bias, and prediction-error quantiles. Stratify by configuration, load regime, device, and cold/warm state.
5. Assess deadline-relevant calibration: for predictions advertised as P95/P99, measure empirical coverage on held-out data and report confidence intervals. A nominal P95 prediction should not be described as reliable unless observed coverage supports it under the stated conditions.
6. Compare the learned estimator against simple baselines: per-configuration mean, per-configuration percentile lookup, and a device/configuration-only predictor. Report the exact paired test rows and do not use synthetic rows as independent observations.
7. Wire the validated predictor into a dedicated evaluation mode and confirm that input telemetry changes can change its output. The model used at runtime must be the same model evaluated here.

**Implementation required:** Add session-aware raw-profile collection and held-out train/evaluation code. Do not cite current in-training metrics from `scripts/train_surrogate.py` as evidence of generalization.

### Experiment 4: Primary Paired Policy Comparison

**Question:** Compared under the same requests and runtime conditions, what accuracy/deadline/latency trade-off does the proposed calibrated percentile selector achieve relative to relevant baselines?

**Required policies:** static fastest, static highest accuracy, mean-profile adaptive, percentile adaptive without calibration, percentile adaptive with calibration, and optionally an oracle using held-out measured latency as a labeled upper bound. Specify one primary percentile (for example P95) before execution; treat P50/P99 as planned secondary analyses, not a search for the most favorable outcome.

**Current script support:** `scripts/evaluate.py` accepts `avg`, `p50`, `p95`, and `p99`, plus `sinusoidal`, `step`, `bursty`, and `heldout` traces. `scripts/run_experiments.py` runs repeated seeds, policies, and traces, but its defaults cover only P95/P99 and sinusoidal/heldout. The present scripts do not implement static fastest/highest-accuracy policies or a switch to disable calibration. Also resolve the warm/cold percentile and feasibility inconsistencies described above before using this run as the primary result.

**Procedure:**

1. Complete Experiments 0-3 and fix the policy semantics: every candidate must use the same named quantile regardless of current/switch state; define safety margin, switching penalty, feasibility, fallback, and independent feasible-budget labels consistently.
2. Implement and unit-test the missing baseline/ablation switches. Add tests proving the policy selects the expected candidate on controlled frontiers, calibration can be disabled, and switching/cold latency is handled as specified.
3. Freeze checkpoint, profile, dataset sample order, policy parameters, device/software environment, and thread count. Derive one exact budget sequence per seed and reuse it across every policy. Save that sequence as an artifact rather than relying on separate processes to recreate it.
4. Stabilize or record the hardware fingerprint. The current runner starts a separate evaluation process for every policy, and the recorded device-speed score can differ between runs. Either use a frozen measured fingerprint/profile for all policies or rerun/stratify sessions so that observed fingerprint variation cannot be mistaken for a policy effect.
5. Smoke-test one seed and trace with each policy. Verify selected configurations, output filenames, prediction path, feasibility labels, and per-sample alignment before launching repetitions.
6. Run paired seeds and traces. Once the runner implements all policies and preserves per-run raw JSONs, a current-CLI example for the supported adaptive subset is:

	```powershell
	python scripts/run_experiments.py --repetitions 10 --threads 1 --policies avg p50 p95 p99 --traces sinusoidal step bursty heldout
	```

	This command does **not** include the static or uncalibrated baselines until those are added to the runner. Do not launch it before fixing evaluation semantics and confirming that it will not overwrite valuable existing result files.
7. Treat a complete evaluation run, not each individual image, as the independent replicate for uncertainty unless the experimental design justifies a different unit. Preserve all per-request results and all run summaries.
8. Report paired differences by seed/trace. Use confidence intervals or another predeclared uncertainty method; include absolute counts and denominators for miss rates. Report accuracy, latency quantiles, prediction error, independently defined infeasibility, feasible-only miss rate, violation magnitude, switching, and overhead.
9. Generate result tables/plots from the archived per-run files using a script. Record any failed or excluded run and its predeclared exclusion reason.

**Pass condition:** The selected policy shows a repeatable trade-off versus paired baselines, with uncertainty, and no conclusion depends on comparing runs with different data, profile, or machine state.

### Experiment 5: Controller Component Ablations

**Question:** Which controller component accounts for observed behavior?

**Ablations:** (A) complete calibrated percentile policy; (B) calibration disabled; (C) mean in place of percentile; (D) switching penalty disabled; (E) telemetry held constant or removed; (F) lookup estimator versus held-out validated learned estimator. Keep the candidate set, data, budget sequence, profile, and random seeds identical.

**Procedure:**

1. Implement each ablation as an explicit configuration option, not as an undocumented local code edit. Save all values in the result JSON and include them in the run manifest.
2. Add unit tests that assert the changed component is actually disabled/replaced and that other policy parameters remain unchanged.
3. Reuse the primary experiment's frozen seeds, budgets, profiles, and model checkpoint. Run the complete policy and each ablation in paired order; randomize execution order if thermal/time drift is a concern, while retaining the pairing key.
4. Repeat the same number of independent runs as the primary comparison. Do not use one run per ablation.
5. Compute paired deltas from the complete policy for miss rate, accuracy, latency quantiles, prediction error, switch rate, and controller overhead. Include uncertainty and report cases where removing a component has no measurable effect.
6. Present this as a mechanism analysis only if the ablation altered exactly one component and the policy semantics were held fixed.

**Implementation required:** Calibration is currently updated by `update_feedback`; there is no evaluated “calibration off” command-line option in the documented script interface. Add the option and tests before running this experiment.

### Experiment 6: Controlled Real-Load Evaluation

**Question:** Does the policy behave as intended when resource contention is measured rather than represented only by simulated telemetry values?

**Current limitation:** In `scripts/evaluate.py`, CPU and temperature scenarios are assigned synthetic values; this is not a controlled GPU/CPU contention experiment. The current latency measurement also covers the model forward pass, not the full serving request.

**Procedure:**

1. Define reproducible load conditions: idle, CPU contention, memory pressure if relevant, and GPU contention if supported. Specify the load generator, concurrency, duration, warm-up, safety limits, and stop conditions.
2. Record real telemetry continuously (CPU utilization, memory availability, GPU utilization/memory/temperature, clocks where available) and timestamp it against every inference request.
3. Use a separate driver process or service to generate load, and ensure the load generator itself is not included in controller inference time. Capture its configuration and logs.
4. Run a no-load baseline first. Verify that load is measurable, stable, and does not cause thermal/power behavior that makes repeated trials incomparable. Randomize or counterbalance policy order across independent load sessions.
5. Repeat each policy under the same load-generation schedule and budget trace. Save the actual observed telemetry trace, request budget, chosen configuration, predicted latency, and measured latency for every request.
6. Measure both model-forward latency and end-to-end request latency if the paper discusses deployment deadlines. State which latency drives controller decisions and which is reported as the service outcome.
7. Analyze each load regime separately, then report pooled results only with an explicit model for between-session/device variation. Never relabel the current simulated-load results as real contention.

**Safety:** Keep load bounded and monitor device temperature. Stop the workload if the machine reaches the hardware/operator's defined thermal or stability limit.

### Experiment 7: Cold Start, Switching, and Controller Overhead

**Question:** Do configuration transitions, calibration, controller computation, or startup costs change the apparent deadline result?

**Procedure:**

1. Define separate timing boundaries: (a) controller selection/update, (b) configuration switch, (c) first inference after a switch, (d) warm inference, and (e) end-to-end request.
2. Extend the profiler to measure every ordered transition between distinct configurations, with repeated samples and controlled warm-up. The current repeated-same-configuration cold loop is not a complete transition matrix.
3. Include lazy batch-normalization calibration as a separate startup cost. Decide whether the application pays this cost once per configuration or before every first use, then implement and report that exact behavior.
4. Instrument the controller using a monotonic high-resolution timer. Measure selection, feature construction, predictor calls, feedback update, and logging separately, with enough repetitions to report distributions.
5. Run workload traces with and without transitions while holding all other inputs fixed. Attribute misses to model execution, transitions, calibration, and controller overhead where possible.
6. Report both forward-only and end-to-end outcomes. Do not subtract overhead estimates from measured end-to-end latency unless the measurements demonstrate that the components do not overlap.

### Experiment 8: Cross-Device Generalization (Optional, Claim-Dependent)

Run this only if the paper intends to claim hardware portability or generalization.

1. Choose at least one held-out target device not used to tune model features, estimator hyperparameters, safety margins, or policy thresholds.
2. Repeat the same checkpoint/configuration accuracy validation and device profiling on the target. Confirm the target can execute every tested precision/configuration; exclude unsupported modes with an explicit reason.
3. Evaluate two settings separately: (a) device-specific profile available before deployment, and (b) zero-shot/unseen-device prediction if that is the claimed use case.
4. Reuse the budget traces and paired workload protocol, adjusting only hardware-specific factors that were predeclared. Record actual hardware/software fingerprints.
5. Report per-device outcomes and uncertainty; do not hide a weak device result in a pooled average. State whether online calibration is allowed to adapt on the target and how many observations it receives.
6. Limit the conclusion to the number and types of devices tested. A small device set does not establish universal portability.

### Recommended Run Order and Stop Gates

1. Experiment 0 smoke check and tests.
2. Experiment 1 accuracy validation and Experiment 2 repeated profiling.
3. Fix policy/feasibility semantics; validate the predictor in Experiment 3 if learned-predictor claims are planned.
4. Implement missing baselines and ablations; unit-test those paths.
5. Run Experiment 4 pilot, inspect paired alignment and output integrity, then run the full repetitions.
6. Run Experiment 5 ablations.
7. Run Experiments 6 and 7 for deployment/real-load claims; otherwise clearly scope them as future work and keep claims to the measured proxy.
8. Run Experiment 8 only when making cross-device claims.

At each gate, stop and repair the protocol if checkpoint loading, profile identity, trace alignment, policy identity, or result provenance is unclear. It is better to delay the comparison than to compute a precise summary from mismatched runs.

## 10. Reproducibility Checklist

Before submission, verify that the artifact package records:

- source revision and unmodified raw result files;
- model/checkpoint identifier and checksum;
- data source, split, preprocessing, and sample-selection procedure;
- configuration list and accuracy provenance;
- profile CSVs, sample counts, profiler settings, and profile timestamp;
- complete commands, seeds, traces, policy parameters, thread count, and repetitions;
- hardware, OS, accelerator driver, framework, CUDA/runtime, and relevant library versions;
- warm-up, synchronization, timing boundary, and cold-start procedure;
- scripts that transform raw per-run data into every reported table and figure;
- definitions for misses, infeasibility, feasible-only denominator, and uncertainty;
- known limitations and instructions to reproduce the main experiment.

## 11. Repository Evidence Map

- [`formulation.md`](../formulation.md): proposed problem, controller equations, metrics, and initial hypothesis.
- [`scripts/evaluate.py`](../scripts/evaluate.py): current sample selection, hard-coded accuracy table, budget/load trace, prediction path, timer, and summary metrics.
- [`scripts/evaluate_config_accuracy.py`](../scripts/evaluate_config_accuracy.py): full-test per-configuration accuracy evaluation, training-only BN recalibration, checkpoint hashing, confidence intervals, and JSON/CSV artifacts.
- [`resource_control/controller.py`](../resource_control/controller.py): measured-frontier lookup, calibration, candidate selection, percentile and feasibility behavior.
- [`scripts/profile_hardware.py`](../scripts/profile_hardware.py): hardware profiling procedure and sample collection.
- [`profiles/nvidia_geforce_gtx_1650_profile.csv`](../profiles/nvidia_geforce_gtx_1650_profile.csv): checked-in GTX 1650 profile data.
- [`scripts/train_surrogate.py`](../scripts/train_surrogate.py): synthetic load expansion and training-set metrics for the RF model.
- [`scripts/run_experiments.py`](../scripts/run_experiments.py): repeated policy/trace runner intended to create an experiment summary.
- [`tests/test_resource_control.py`](../tests/test_resource_control.py): controller behavior tests; these are not comparative performance evaluation.
- [`results/evaluation_p95_sinusoidal_seed12345_threads1.json`](../results/evaluation_p95_sinusoidal_seed12345_threads1.json) and [`results/evaluation_p99_sinusoidal_seed12345_threads1.json`](../results/evaluation_p99_sinusoidal_seed12345_threads1.json): the current GPU descriptive snapshots.
- [`results/evaluation_results.json`](../results/evaluation_results.json): older CPU result with a different schema and context.
- [`README.md`](../README.md): project overview and figures that should be reconciled against source artifacts.

## Final Submission Gate

Do not submit until the following sentence can be completed with evidence for every blank:

> On **[specified data]**, using **[model/checkpoint and candidate configurations]** on **[specified hardware]**, under **[measured or simulated workloads and budgets]**, the proposed **[fully specified policy]** changed **[primary outcome]** by **[effect with uncertainty]** relative to **[paired baseline]**, while **[accuracy/overhead trade-off]**; this conclusion is limited to **[scope and limitations]**.

If any field cannot be filled from a reproducible experiment, narrow the claim or run the missing experiment first.
