# Resource Adaptive Dynamic Inference of Neural Network Model

> **A universally slimmable & quantization-aware ResNet with a closed-loop, surrogate-backed runtime controller that dynamically selects the optimal subnet configuration to maximize accuracy while respecting fluctuating latency budgets.**

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Key Concepts](#2-key-concepts)
3. [System Architecture](#3-system-architecture)
4. [Project Structure](#4-project-structure)
5. [Results](#5-results)
6. [Prerequisites & Installation](#6-prerequisites--installation)
7. [Step-by-Step Usage Guide](#7-step-by-step-usage-guide)
8. [Component Deep Dive](#8-component-deep-dive)
9. [Configuration Reference](#9-configuration-reference)
10. [Running the Tests](#10-running-the-tests)
11. [Contributing](#11-contributing)
12. [Roadmap](#12-roadmap)

---

## 1. What This Project Does

Modern neural networks are deployed on hardware ranging from data-centre GPUs to embedded CPUs. In real deployments, the system is never idle — background processes, thermal throttling, and fluctuating workloads mean that the time available for a single inference pass changes constantly.

**AnyNet** solves this with a two-part system:

1. **A single trained model** (Universally Slimmable & Quantized ResNet-50) that can execute at any of 16 pre-defined configurations by adjusting its _channel width multiplier_ and _activation/weight bit-width_ at inference time — no re-training or separate model files needed.

2. **A runtime resource controller** that reads live hardware telemetry (CPU load, memory, thermals), predicts the latency of each candidate subnet using a machine-learned surrogate model, and selects the highest-accuracy subnet that is statistically guaranteed to finish within the current latency budget.

The loop is **closed**: after every inference step, the controller measures the actual latency and updates an online bias-correction factor (`k_adapt`) so predictions stay accurate even as the OS environment shifts.

### Achieved Results (NVIDIA GTX 1650, CIFAR-10)

| Metric                              | Value                                          |
| ----------------------------------- | ---------------------------------------------- |
| Overall Accuracy                    | **93.00%**                                     |
| Average Inference Latency           | **10.78 ms**                                   |
| Mean Latency Prediction Error (MAE) | **3.58 ms**                                    |
| Deadline Miss Rate                  | **21%** (sinusoidal budget + contention spike) |
| K_adapt Operating Range             | [0.633, 1.299]                                 |

---

## 2. Key Concepts

### Universally Slimmable Networks

Instead of training a separate model for each target hardware, a _universally slimmable_ network is trained once to operate at any width in a continuous range `[0.25, 1.0]`. During the forward pass, each convolutional layer only activates a _slice_ of its full channel count, controlled by `width_mult`.

### Quantization-Aware Training (QAT)

The model is trained with _fake quantization_ — activations and weights are clipped and rounded to simulate `4`, `8`, `16`, or `32`-bit integer arithmetic during the forward pass, but gradients still flow through for training. This gives near-lossless accuracy even at 4-bit.

### Sandwich Rule Training

To train a single model that works well at all widths simultaneously, each gradient step trains:

1. The **widest** (full-precision) configuration — learns from ground-truth labels.
2. The **narrowest** configuration — uses the widest model's soft outputs as a teacher (knowledge distillation).
3. Several **randomly sampled** intermediate configurations — also distilled from the widest model.

### Surrogate-Backed Controller

A `RandomForestRegressor` is trained offline on hardware profiles collected from multiple devices. At runtime it predicts the latency of each of the 16 (expanded to ~60 with interpolation) candidate configurations in microseconds, given the current CPU/thermal/memory telemetry and hardware fingerprint.

### Closed-Loop Online Adaptation

A scalar multiplier `k_adapt` corrects systematic prediction drift in real time:

```
error_ratio  = (actual_latency - predicted_latency) / predicted_latency
k_adapt     += lr * error_ratio          # lr = 0.05
k_adapt      = clip(k_adapt, 0.5, 3.0)
```

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Runtime Control Loop                      │
│                                                              │
│  ┌──────────────┐    ResourceState     ┌──────────────────┐  │
│  │ ResourceMonitor│ ─────────────────► │                  │  │
│  │  (telemetry) │                      │ SurrogateBackd   │  │
│  └──────────────┘                      │   Controller     │  │
│                                        │                  │  │
│  ┌──────────────┐    hw_fingerprint    │  ┌────────────┐  │  │
│  │  Hardware    │ ─────────────────► │  │  RF Surrog │  │  │
│  │  Profiler    │                      │  │  ate Model │  │  │
│  └──────────────┘                      │  └────────────┘  │  │
│                                        │  ┌────────────┐  │  │
│  ┌──────────────┐   latency_budget_ms  │  │  Pareto    │  │  │
│  │   Dynamic    │ ─────────────────► │  │  Frontier  │  │  │
│  │   Budget B(t)│                      │  └────────────┘  │  │
│  └──────────────┘                      └────────┬─────────┘  │
│                                                 │             │
│                              (width_mult, bit_width)          │
│                                                 ▼             │
│                                   ┌────────────────────────┐  │
│                                   │ Slimmable+Quant ResNet │  │
│                                   │       (inference)       │  │
│                                   └────────────┬───────────┘  │
│                                                │               │
│                              actual_latency_ms │               │
│                                                ▼               │
│                                   ┌────────────────────────┐  │
│                                   │  Feedback: update      │  │
│                                   │  k_adapt online        │  │
│                                   └────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Offline Pipeline (run once per new device)

```
profile_hardware.py  →  profiles/<device>_profile.csv
                                    │
                                    ▼
              scripts/train_surrogate.py  →  weights/surrogate_model.pkl
```

### Online Pipeline (runs at inference time)

```
scripts/evaluate.py
  ├── Load model checkpoint  (models/checkpoint/)
  ├── Load surrogate model   (weights/surrogate_model.pkl)
  ├── Load hardware profile  (profiles/<device>_profile.csv)
  └── For each sample:
        telemetry → controller.select() → set_model_width/bit_width → inference → controller.update_feedback()
```

---

## 4. Project Structure

```
anynet/
│
├── models/                          # Neural network definition
│   ├── __init__.py                  # Public API exports
│   ├── config.py                    # Global FLAGS: training hyperparameters & deploy configs
│   ├── ops.py                       # Slimmable conv/linear/BN + fake-quantization ops
│   ├── resnet.py                    # ResNet-50 Block + Model, set_model_width/bit_width
│   ├── train.py                     # Training loop (sandwich rule + KD), recalibrate_bn, evaluate
│   ├── anynet_model.ipynb           # Interactive training & visualization notebook
│   └── checkpoint/
│       └── us_resnet_epoch100_checkpoint.pt   # Pre-trained weights (not tracked by git)
│
├── resource_control/                # Runtime controller & telemetry
│   ├── controller.py                # HysteresisController, RuleBasedController,
│   │                                #   SurrogateBackedController, PhysicsSurrogateModel
│   ├── telemetry.py                 # ResourceState, ResourceMonitor, hardware_fingerprint
│   ├── calibration.py               # Multi-hardware profiling & leave-one-out cross-validation
│   ├── cost_table.py                # Pareto frontier helpers
│   └── trace_sim.py                 # Lightweight latency trace simulator
│
├── scripts/                         # Runnable entry-point scripts
│   ├── profile_hardware.py          # Step 1 – measure latency of all 16 configs on this device
│   ├── train_surrogate.py           # Step 2 – train RandomForest surrogate on profiles/
│   └── evaluate.py                  # Step 3 – closed-loop evaluation with sinusoidal budgets
│
├── profiles/                        # Auto-generated hardware latency databases (CSV)
│   ├── nvidia_geforce_gtx_1650_profile.csv
│   ├── tesla_t4_profile.csv
│   └── cpu_x86_64_profile.csv
│
├── weights/
│   └── surrogate_model.pkl          # Trained RandomForest surrogate (not tracked by git)
│
├── tests/
│   ├── test.py                      # Basic smoke tests
│   └── test_resource_control.py     # Scenario-based integration tests for the controller
│
├── data/                            # Placeholder for dataset utilities
├── cifar10/                         # Auto-downloaded CIFAR-10 dataset (not tracked by git)
├── RESOURCE_CONTROL_GUIDE.md        # Deep-dive math reference for the control system
└── README.md                        # This file
```

---

## 5. Results

### Accuracy vs. Configuration (CIFAR-10, full test set)

| Width Mult | Bit Width | Top-1 Accuracy |
| :--------: | :-------: | :------------: |
|    0.25    |     4     |     89.69%     |
|    0.25    |    32     |     90.59%     |
|    0.50    |     8     |     91.83%     |
|    0.75    |     8     |     92.12%     |
|    1.00    |     8     |     92.32%     |
|    1.00    |    32     |   **92.37%**   |

### Latency Profile (NVIDIA GTX 1650)

| Width Mult | Bit Width | Avg Latency | Std Dev |
| :--------: | :-------: | :---------: | :-----: |
|    0.25    |     4     |    ~5 ms    |  ~1 ms  |
|    0.50    |     8     |   ~11 ms    | ~1.5 ms |
|    0.75    |     8     |   ~18 ms    |  ~2 ms  |
|    1.00    |    32     |   ~26 ms    | ~2.5 ms |

### Closed-Loop Evaluation Summary

The evaluation in `scripts/evaluate.py` simulates 100 inference steps with:

- **Sinusoidal latency budget**: mean 20 ms, amplitude ±12 ms, period 40 steps
- **Contention spike** at steps 30–70: CPU load jumps to 85%, thermal to 80°C

The controller dynamically switched between configs — selecting the largest/most-accurate subnet when the budget was generous, falling back to `(0.25, 32)` during tight budget windows, without manual tuning.

---

## 6. Prerequisites & Installation

### Requirements

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- ~2 GB disk space for dataset + checkpoint

### Install

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/anynet.git
cd anynet

# 2. Create and activate a virtual environment
python -m venv env

# Windows
.\env\Scripts\activate

# Linux / macOS
source env/bin/activate

# 3. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install scikit-learn pandas numpy psutil
```

> **Note:** Replace `cu121` with your CUDA version (e.g. `cu118`, `cpu`). See [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the exact command.

### Verify Installation

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 7. Step-by-Step Usage Guide

All scripts must be run from the **project root** (`anynet/`).

### Step 0: Obtain a Pre-Trained Checkpoint

The pre-trained weights file `models/checkpoint/us_resnet_epoch100_checkpoint.pt` is not tracked by git (it is ~380 MB). You have two options:

**Option A — Download the provided checkpoint** _(link to be added by maintainer)_

**Option B — Train from scratch** (requires a GPU, ~4–8 hours on a single T4):

```bash
python -m models.train
```

This runs the full sandwich-rule + knowledge-distillation training loop for 100 epochs on CIFAR-10 and saves checkpoints every 5 epochs.

---

### Step 1: Profile Your Hardware

Generate a device-specific latency database. This takes ~5–10 minutes.

```bash
python scripts/profile_hardware.py
```

**What it does:**

- Detects your GPU or CPU automatically.
- Runs 20 warmup + 200 measurement passes for each of the 16 `(width_mult, bit_width)` configurations.
- Saves results to `profiles/<device_name>_profile.csv`.

**Example output:**

```
Detected Hardware Device: cuda (nvidia_geforce_gtx_1650)
[1/16] Profiling config: width=0.25, bits=4 ...  Avg: 5.12ms | Std: 0.83ms
[2/16] Profiling config: width=0.25, bits=8 ...  Avg: 5.48ms | Std: 0.91ms
...
Profiling complete! Results saved to: profiles/nvidia_geforce_gtx_1650_profile.csv
```

> **Why this step?** Latency varies significantly between devices (a GTX 1650 vs. a T4 vs. a laptop CPU). The surrogate model is device-aware, so it needs your hardware's measurements to be accurate.

---

### Step 2: Train the Surrogate Latency Model

```bash
python scripts/train_surrogate.py
```

**What it does:**

- Reads all CSV files from `profiles/`.
- Synthesizes additional training rows by simulating high-CPU (×1.3) and high-thermal (×1.25) load scenarios.
- Trains a `RandomForestRegressor` (100 estimators) on 12 input features.
- Saves the model to `weights/surrogate_model.pkl`.

**Feature set used:**

| Feature              | Description                                 |
| -------------------- | ------------------------------------------- |
| `width_mult`         | Active channel fraction (0.25–1.0)          |
| `bit_width`          | Quantization precision (4, 8, 16, 32)       |
| `approx_flops`       | Estimated FLOPs ∝ width_mult² × 10⁶         |
| `approx_params`      | Estimated parameter count                   |
| `flops_per_speed`    | FLOPs normalized by device throughput score |
| `cpu_pct`            | Current CPU utilization (%)                 |
| `mem_available_mb`   | Available RAM (MB)                          |
| `thermal_c`          | CPU/GPU temperature (°C)                    |
| `device_speed_score` | Matmul throughput proxy (matmuls/sec)       |
| `cpu_cores`          | Physical CPU core count                     |
| `ram_gb`             | Total system RAM (GB)                       |
| `has_cuda`           | 1.0 if CUDA GPU is active, else 0.0         |

> **Tip:** If you have profiles from multiple devices, place them all in `profiles/` before running this step. The surrogate will generalise better across hardware.

---

### Step 3: Run the Closed-Loop Evaluation

```bash
python scripts/evaluate.py
```

**What it does:**

- Loads the pre-trained ResNet-50 checkpoint.
- Loads the surrogate model and device profile.
- Selects 100 balanced CIFAR-10 test samples (10 per class).
- For each sample, simulates a dynamic latency budget and hardware contention scenario, asks the controller to pick the best config, runs inference, and feeds back the actual latency.
- Prints a per-sample log every 10 steps and a final summary.

**Example output:**

```
Sample 010/100 | Normal Load     | Budget: 28.3ms | Pred: 24.2ms | Actual: 25.1ms | K_adapt: 1.033 | Miss: False | Selected: (1.0, 32) | Correct: True
Sample 040/100 | Contention Spike| Budget: 14.7ms | Pred: 11.2ms | Actual: 13.8ms | K_adapt: 1.112 | Miss: False | Selected: (0.5, 8)  | Correct: True
...
Total Samples Evaluated: 100
Overall Accuracy: 93.00%
Average Inference Latency: 10.78 ms
Mean Absolute Prediction Error (MAE): 3.58 ms
Deadline Miss Rate: 21.00% (21/100 missed)
K_adapt Operating Range: [0.633, 1.299]
```

---

## 8. Component Deep Dive

### `models/ops.py` — Slimmable & Quantized Primitives

The core building blocks that make the model adaptable at runtime:

- **`ResnetConv2d`**: A `nn.Conv2d` that reads `self.width_mult` on every forward pass and slices `in_channels` and `out_channels` accordingly. Stem layers only slice the output.
- **`ResnetBatchNorm2d`**: Maintains a separate `(running_mean, running_var)` dict keyed by `(width_mult, bit_width)`. The correct stats are loaded before each forward pass and saved back after, preventing cross-subnet contamination.
- **`ActQuant`**: Fake-quantizes activations by clamping to `[-1, 1]` and rounding to `2^bit_width` levels. In 32-bit mode it is an identity.
- **`fake_quantize_weight`** / **`fake_quantize_act`**: Straight-through estimator (STE) quantization used during the training forward pass.

### `models/resnet.py` — The Backbone

A ResNet-50 with bottleneck blocks (`1×1 → 3×3 → 1×1`), adapted to:

- Accept a CIFAR-10 stem (3×3 conv, no max-pool) or an ImageNet stem (7×7 + max-pool).
- Dynamically compute channel sizes at the _maximum_ width, then slice at runtime.
- Expose `set_width(w)` and `set_bit_width(b)` methods that propagate to every layer.

### `models/train.py` — Training Pipeline

- **`train_one_epoch`**: Implements the sandwich rule. Each batch trains the max-width config (cross-entropy loss), then the min-width and randomly sampled widths (KL-divergence from the max-width soft outputs).
- **`recalibrate_bn`**: After training, BN statistics must be re-estimated for each `(width_mult, bit_width)` pair by running forward passes in `train()` mode. This is also called lazily at evaluation time when a new config is first encountered.
- **`evaluate`**: Loops over all 16 deploy configs, computing accuracy and loss on the full test set.
- **`get_dataloaders`**: Auto-downloads CIFAR-10 from fast.ai's S3 mirror if it is not present locally.

### `resource_control/telemetry.py` — Hardware Sensing

- **`ResourceState`**: A frozen dataclass capturing `(cpu_pct, mem_available_mb, battery_pct, thermal_c, timestamp)`.
- **`ResourceMonitor`**: Polls `psutil` for CPU/memory/battery. Reads the Linux thermal sysfs path for temperature (falls back gracefully on Windows/macOS). Applies **exponential smoothing** (`alpha=0.3`) to filter short transient spikes that would otherwise cause unnecessary config switches.
- **`hardware_fingerprint`**: Benchmarks the device with 30 repeated 64×256 matrix multiplications (after 5 warmup passes) to produce a `device_speed_score`. Combined with CPU count, RAM, and CUDA availability, this uniquely characterises a device for the surrogate.

### `resource_control/controller.py` — The Brain

Three controllers are provided, in order of increasing sophistication:

| Class                       | Description                                                               | Use When                   |
| --------------------------- | ------------------------------------------------------------------------- | -------------------------- |
| `HysteresisController`      | Simple budget-based selection with a minimum dwell time.                  | Baseline / ablation        |
| `RuleBasedController`       | Hand-tuned heuristics with thermal, memory, and CPU penalties.            | No training data available |
| `SurrogateBackedController` | ML-predicted latency + online `k_adapt` correction + uncertainty margins. | **Production use**         |

**`SurrogateBackedController.select()` logic:**

```
For each candidate config in frontier:
    1. Build 12-feature row (config + live telemetry + hw_fingerprint)
    2. pred_latency = surrogate_model.predict(row)
    3. pred_latency_adj = pred_latency * k_adapt
    4. std_latency = config.std_ms * k_adapt
    5. risk_score = pred_latency_adj + k_risk * std_latency
    6. if config != current_config: risk_score += switching_penalty_ms
    7. if risk_score <= budget * safety_margin AND acc > best_acc:
           candidate = this config

Return highest-accuracy feasible config (or lowest-latency fallback).
```

The **switching penalty** discourages unnecessary config changes — switching has a real overhead because `ResnetBatchNorm2d` must reload different running statistics and the CPU cache is disrupted.

### `resource_control/calibration.py` — Cross-Hardware Validation

- **`collect_profiling_data`**: Profiles any model on the current device with live telemetry capture.
- **`leave_one_hardware_out_eval`**: Trains the surrogate on all devices except one, then measures how well it generalizes to the held-out device with 0, 5, 10, or 25 few-shot calibration samples. This is the standard evaluation methodology for domain adaptation in hardware-aware ML.

---

## 9. Configuration Reference

All model and training hyperparameters live in `models/config.py` as a `FLAGS` object:

| Flag                | Default           | Description                                           |
| ------------------- | ----------------- | ----------------------------------------------------- |
| `dataset`           | `"cifar10"`       | Dataset name (`"cifar10"` or `"imagenet"`)            |
| `depth`             | `50`              | ResNet depth (`50`, `101`, `152`)                     |
| `width_mult_range`  | `(0.25, 1.0)`     | Continuous range for width multiplier during training |
| `bit_width_list`    | `[4, 8, 16, 32]`  | Quantization bit-widths to sample during training     |
| `num_sample_widths` | `2`               | Extra random widths sampled per sandwich step         |
| `width_divisor`     | `8`               | Channel counts are rounded to this multiple           |
| `batch_size`        | `256`             | Training batch size                                   |
| `num_epochs`        | `100`             | Total training epochs                                 |
| `lr`                | `0.1`             | Initial SGD learning rate                             |
| `momentum`          | `0.9`             | SGD momentum                                          |
| `weight_decay`      | `1e-4`            | L2 regularisation                                     |
| `deploy_configs`    | 16 `(w, b)` pairs | Configs profiled and evaluated at inference           |

**Controller parameters** (set directly in `scripts/evaluate.py`):

| Parameter              | Default | Description                                                     |
| ---------------------- | ------- | --------------------------------------------------------------- |
| `min_dwell_s`          | `0.0`   | Minimum seconds between config switches (0 = switch every step) |
| `k_risk`               | `1.2`   | Multiplier on std-dev for risk-averse safety margin             |
| `switching_penalty_ms` | `1.5`   | Extra latency penalty added to non-active configs               |
| `safety_margin`        | `0.9`   | Target budget fraction (avoids running right at the edge)       |
| `lr_adapt`             | `0.05`  | Online learning rate for `k_adapt` correction                   |

---

## 10. Running the Tests

```bash
# Integration test: controller scenario simulation
python tests/test_resource_control.py

# Basic smoke tests
python tests/test.py
```

`test_resource_control.py` runs four scenarios end-to-end:

1. **Generous budget, low load** — expects a large/accurate config.
2. **Tight budget, low load** — expects a narrower config.
3. **Tight budget + high CPU/thermal stress** — expects the surrogate to scale up predictions and select a smaller config.
4. **Extreme budget under stress** — tests the fallback to the smallest config.

---

## 11. Contributing

Contributions are welcome! Here is how to get started:

### Workflow

1. **Fork** the repository and clone your fork.
2. **Create a branch**: `git checkout -b feature/your-feature-name`
3. Make your changes, add tests if applicable.
4. **Run the tests** to confirm nothing is broken.
5. Open a **Pull Request** with a clear description of what you changed and why.

### Where to Contribute

- **New hardware profiles**: Run `scripts/profile_hardware.py` on your device and submit the generated CSV in `profiles/`. This helps the surrogate generalise better.
- **Controller policies**: Implement a new controller class in `resource_control/controller.py`. It should expose a `select(resource_state, latency_budget_ms, hw_fingerprint)` method.
- **New datasets / backbones**: The slimmable ops in `models/ops.py` can wrap any `nn.Conv2d` or `nn.Linear`.
- **Bug reports**: Please open an issue on GitHub with the full traceback and your `python --version` / `torch.__version__` output.

### Code Style

- Follow existing code conventions (no external formatter is enforced yet).
- Document new public functions with a docstring.
- Keep scripts runnable from the project root with `python scripts/<script>.py`.

---

## 12. Roadmap

- [ ] ImageNet support (replace CIFAR-10 stem + larger checkpoint)
- [ ] ONNX export for each subnet configuration
- [ ] Real-time telemetry dashboard (live `k_adapt`, config selection, latency plot)
- [ ] ARM / Apple Silicon hardware profiles
- [ ] Bayesian surrogate (Gaussian Process) to replace Random Forest for better uncertainty estimates
- [ ] `pip`-installable package

---

## License

This project is open source. See `LICENSE` for details _(to be added)_.
