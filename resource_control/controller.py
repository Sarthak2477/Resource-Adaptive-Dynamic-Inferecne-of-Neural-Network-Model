import json
import time

class HysteresisController:
    """Wraps a base policy with a minimum dwell time, so noisy telemetry
    doesn't cause rapid config thrashing."""

    def __init__(self, frontier, min_dwell_s=2.0, safety_margin=0.9):
        self.frontier = frontier  # sorted list of dicts, from pareto_frontier()
        self.min_dwell_s = min_dwell_s
        self.safety_margin = safety_margin   # target under budget, not exactly at it
        self.current_config = frontier[-1]['config']   # start at best-effort
        self._last_switch_time = time.time()

    def _best_for_budget(self, latency_budget_ms):
        target = latency_budget_ms * self.safety_margin
        feasible = [r for r in self.frontier if r['latency_ms'] <= target]
        return max(feasible, key=lambda r: r['acc']) if feasible else self.frontier[0]

    def select(self, resource_state, latency_budget_ms):
        now = time.time()
        candidate = self._best_for_budget(latency_budget_ms)

        if candidate['config'] == self.current_config:
            return self.current_config   # no change needed

        if (now - self._last_switch_time) < self.min_dwell_s:
            return self.current_config   # too soon to switch again, absorb the noise

        self.current_config = candidate['config']
        self._last_switch_time = now
        return self.current_config

    @staticmethod
    def resource_state_to_budget(state, base_budget_ms=30.0):
        budget = base_budget_ms
        if state.battery_pct is not None and state.battery_pct < 15:
            budget *= 0.5
        if state.thermal_c is not None and state.thermal_c > 70:
            budget *= 0.6
        if state.cpu_pct > 80:
            budget *= 0.7
        return budget

class RuleBasedController:
    """Simple hand-tuned heuristic policy: if thermal / memory tight, step down
    aggressively; otherwise stay on the current config or try to go up
    if there's room."""

    def __init__(self, frontier, baseline_config):
        self.frontier = frontier  # sorted list of dicts
        self.baseline_config = baseline_config
        self.current_config = baseline_config
        self._last_check = 0.0

    def select(self, resource_state, latency_budget_ms):
        now = time.time()
        if (now - self._last_check) < 2.0:
            return self.current_config
        self._last_check = now

        budget_ms = latency_budget_ms * 0.9
        budget_w = 2.0  #Watt, adjust based on your device

        # Get current perceived cost
        current_latency = next((r['latency_ms'] for r in self.frontier if r['config'] == self.current_config), None)
        current_acc = next((r['acc'] for r in self.frontier if r['config'] == self.current_config), 0.0)

        # Penalize for thermal violations
        thermal_penalty = 1.0
        if resource_state.thermal_c is not None and resource_state.thermal_c > 70:
            thermal_penalty *= 1.1 + min(0.5, (resource_state.thermal_c - 70) / 30.0)  # +10–60% cost

        # Penalize for memory pressure
        mem_penalty = 1.0
        if resource_state.mem_available_mb is not None and resource_state.mem_available_mb < 512:
            mem_penalty *= 1.1 + min(0.5, (512 - resource_state.mem_available_mb) / 256.0)

        # Penalize for CPU contention (if we're being squeezed by other apps)
        cpu_penalty = 1.0
        if resource_state.cpu_pct > 85:
            cpu_penalty *= 1.05 + min(0.3, (resource_state.cpu_pct - 85) / 15.0)

        # Apply penalties to latency
        if current_latency is not None:
            current_latency_adj = current_latency * thermal_penalty * mem_penalty * cpu_penalty
        else:
            current_latency_adj = float('inf')

        # Find the best feasible config
        candidate = None
        best_acc = -1.0

        for r in self.frontier:
            r_latency = r['latency_ms']
            r_mem = r.get('mem_mb', 0.0)
            r_acc = r['acc']
            r_config = r['config']

            # Check if config is feasible under current conditions
            feasible = True

            # Memory check (simple upper bound)
            if r_mem > 0 and resource_state.mem_available_mb is not None and r_mem > resource_state.mem_available_mb:
                feasible = False

            # Thermal check
            r_thermal_penalty = 1.0
            if r_config and len(r_config) >= 2:
                w = r_config[0]; bw = r_config[1]
                # deeper / wider models stress thermal more
                stress = (w / 32) * (bw / 16) * 1.5
                if resource_state.thermal_c is not None:
                    if resource_state.thermal_c > 70:
                        r_thermal_penalty *= 1.05 + min(0.5, (resource_state.thermal_c - 70) / 30.0) * stress

            # CPU contention
            r_cpu_penalty = 1.0
            if resource_state.cpu_pct > 85:
                r_cpu_penalty *= 1.05 + min(0.3, (resource_state.cpu_pct - 85) / 15.0) * stress

            # Apply per-config penalties
            r_latency_adj = r_latency * r_thermal_penalty * r_cpu_penalty

            # Don't pick a config that's clearly slower than what we're already running (unless it buys massive accuracy)
            if current_latency_adj != float('inf') and r_latency_adj > current_latency_adj * 1.5 and r_acc <= current_acc:
                continue

            # Budget compliance
            if r_latency_adj > budget_ms:
                continue

            # Keep best feasible
            if feasible and r_acc > best_acc:
                best_acc = r_acc
                candidate = r

        if candidate is not None:
            self.current_config = candidate['config']
            return self.current_config
        else:
            # Fallback: find something that fits with relaxed margins
            target = latency_budget_ms * 1.3
            feasible = [r for r in self.frontier if r['latency_ms'] <= target]
            if feasible:
                candidate = max(feasible, key=lambda r: r['acc'])
                self.current_config = candidate['config']
                return self.current_config
            else:
                # If nothing fits, go with baseline (it's already running, something must fit)
                return self.baseline_config

FEATURES = [
    "width_mult", "bit_width",
    "approx_flops", "approx_params",
    "flops_per_speed",
    "cpu_pct", "mem_available_mb", "thermal_c",
    "device_speed_score", "cpu_cores", "ram_gb", "has_cuda",
]

CANONICAL_LATENCY_POLICY = "p95"
CANONICAL_SAFETY_MARGIN = 0.9
CANONICAL_MIN_DWELL_S = 0.0
CANONICAL_SWITCHING_PENALTY_MS = 0.0
CANONICAL_CANDIDATE_WIDTHS = (0.25, 0.5, 0.75, 1.0)
CANONICAL_EXPERIMENT_PROTOCOL = {
    "precision_bits": 32,
    "candidate_widths": list(CANONICAL_CANDIDATE_WIDTHS),
    "latency_policy": CANONICAL_LATENCY_POLICY,
    "safety_margin": CANONICAL_SAFETY_MARGIN,
    "minimum_dwell_seconds": CANONICAL_MIN_DWELL_S,
    "switching_penalty_ms": CANONICAL_SWITCHING_PENALTY_MS,
    "fallback": "lowest adjusted predicted latency in the current measured resource condition when no candidate is feasible",
}

def extract_config_features(config):
    """Extracts width_mult, bit_width, approx_flops, approx_params
    from a config tuple or dict."""
    if isinstance(config, dict) and "config" in config:
        config = config["config"]

    width = 32.0
    bit_width = 16.0

    if isinstance(config, (list, tuple)):
        if len(config) >= 1:
            width = float(config[0])
        if len(config) >= 2:
            bit_width = float(config[1])
    elif isinstance(config, (int, float)):
        width = float(config)

    # Frontier configs store width multipliers directly, e.g. 0.25 or 1.0.
    width_mult = width

    # Approximate FLOPs and Params
    approx_flops = (width_mult ** 2) * 1e6
    approx_params = (width_mult ** 2) * 1e5

    return {
        "width_mult": width_mult,
        "bit_width": bit_width,
        "approx_flops": approx_flops,
        "approx_params": approx_params,
    }

class PhysicsSurrogateModel:
    """A fallback physics-based latency estimator when ML models are not yet trained."""
    
    def predict(self, X):
        preds = []
        for row in X:
            # Keep this positional fallback aligned with FEATURES.
            approx_flops = row[FEATURES.index("approx_flops")]
            device_speed = row[FEATURES.index("device_speed_score")]
            cpu_pct = row[FEATURES.index("cpu_pct")]
            thermal_c = row[FEATURES.index("thermal_c")]
            
            # latency (ms) ≈ (flops / speed) * 1000.0 * contention
            base_lat = (approx_flops / device_speed) * 1000.0
            contention = 1.0 + (cpu_pct / 100.0)
            if thermal_c > 70:
                contention *= 1.2
            preds.append(base_lat * contention)
        return preds


class MeasuredFrontierLatencyModel:
    """Look up measured latency percentiles for a configuration and condition."""

    def __init__(self, frontier, latency_policy="avg"):
        if latency_policy not in ("avg", "p50", "p95", "p99"):
            raise ValueError("latency_policy must be avg, p50, p95, or p99")
        self.latency_policy = latency_policy
        self._latencies = {}
        for row in frontier:
            config = (float(row["config"][0]), float(row["config"][1]))
            condition_profiles = row.get("condition_profiles")
            if condition_profiles:
                for condition, profile in condition_profiles.items():
                    self._latencies[(condition, config)] = self._row_latency(profile)
            else:
                self._latencies[("baseline", config)] = self._row_latency(row)

    def _row_latency(self, row):
        key = {
            "avg": "latency_ms",
            "p50": "latency_p50_ms",
            "p95": "latency_p95_ms",
            "p99": "latency_p99_ms",
        }[self.latency_policy]
        return float(row.get(key, row.get("latency_ms", row.get("avg_latency_ms", 0.0))))

    def predict(self, X):
        predictions = []
        for row in X:
            config = (float(row[FEATURES.index("width_mult")]),
                      float(row[FEATURES.index("bit_width")]))
            condition_code = int(row[len(FEATURES)]) if len(row) > len(FEATURES) else 0
            condition = "contention" if condition_code == 1 else "baseline"
            key = (condition, config)
            if key not in self._latencies:
                raise ValueError(f"No measured latency for configuration {config} in {condition} condition")
            predictions.append(self._latencies[key])
        return predictions

class SurrogateBackedController:
    """Uses a machine learning surrogate model to predict model config latencies
    under current hardware & resource state, selecting the best fit config."""

    def __init__(self, frontier, surrogate_model=None, baseline_config=None, safety_margin=CANONICAL_SAFETY_MARGIN, min_dwell_s=CANONICAL_MIN_DWELL_S, k_risk=0.0, switching_penalty_ms=CANONICAL_SWITCHING_PENALTY_MS, latency_policy=CANONICAL_LATENCY_POLICY, calibration_lr=0.05):
        if latency_policy not in ("avg", "p50", "p95", "p99"):
            raise ValueError("latency_policy must be avg, p50, p95, or p99")
        self.latency_policy = latency_policy
        self.frontier = sorted(frontier, key=lambda row: self._profile_latency(row, cold=False))
        self.surrogate_model = surrogate_model if surrogate_model is not None else PhysicsSurrogateModel()
        self.baseline_config = baseline_config if baseline_config is not None else self.frontier[0]['config']
        self.current_config = self.baseline_config
        self.safety_margin = safety_margin
        self._last_switch_time = time.time()
        self.min_dwell_s = min_dwell_s
        self.k_adapt = 1.0
        self.lr_adapt = 0.05
        self.last_predicted_latency = None
        self.k_risk = k_risk
        self.switching_penalty_ms = switching_penalty_ms
        self.calibration_lr = calibration_lr
        self.k_adapt_by_config = {row["config"]: 1.0 for row in frontier}
        self.last_selected_config = None
        self.last_prediction_is_cold = False
        self.last_budget_feasible = True
        self.last_selection_status = "uninitialized"
        self.last_min_safe_latency = None

    @staticmethod
    def _latency_key(policy):
        return {
            "avg": "latency_ms",
            "p50": "latency_p50_ms",
            "p95": "latency_p95_ms",
            "p99": "latency_p99_ms",
        }[policy]

    def _profile_latency(self, row, cold=False):
        if cold:
            key = f"cold_latency_{self.latency_policy}_ms"
            if key in row:
                return float(row[key])
        key = self._latency_key(self.latency_policy)
        return float(row.get(key, row.get("latency_ms", 0.0)))

    def update_feedback(self, actual_latency_ms, predicted_latency_ms, config=None):
        """
        Updates the prediction bias scale factor (k_adapt) on-the-fly.
        """
        if predicted_latency_ms <= 0:
            return
        
        config = config or self.last_selected_config or self.current_config
        # Calculate proportional error ratio and limit one OS interruption's influence.
        error_ratio = (actual_latency_ms - predicted_latency_ms) / predicted_latency_ms
        error_ratio = max(-0.5, min(1.0, error_ratio))
        
        updated = self.k_adapt_by_config.get(config, 1.0) + self.calibration_lr * error_ratio
        self.k_adapt_by_config[config] = max(0.5, min(updated, 3.0))
        self.k_adapt = self.k_adapt_by_config[config]
        
        # Clip to a safe operating range to prevent runaway feedback
        self.k_adapt = max(0.5, min(self.k_adapt, 3.0))

    def select(self, resource_state, latency_budget_ms, hw_fingerprint):
        now = time.time()

        # Dwell check to prevent rapid switching
        if (now - self._last_switch_time) < self.min_dwell_s:
            return self.current_config

        # Default values if resource state is None
        if resource_state is None:
            cpu_pct = 50.0
            mem_available_mb = 1024.0
            thermal_c = 45.0
        else:
            cpu_pct = resource_state.cpu_pct
            mem_available_mb = resource_state.mem_available_mb
            thermal_c = resource_state.thermal_c if resource_state.thermal_c is not None else 45.0

        speed = hw_fingerprint.get("device_speed_score", 1000.0)
        cpu_cores = hw_fingerprint.get("cpu_cores", 4)
        ram_gb = hw_fingerprint.get("ram_gb", 8.0)
        has_cuda = hw_fingerprint.get("has_cuda", 0.0)

        candidate = None
        best_acc = -1.0
        safe_latencies = []
        ranked_candidates = []
        self.last_min_safe_latency = None
        self.last_budget_feasible = False
        self.last_selection_status = "infeasible_budget"
        target_budget = latency_budget_ms * self.safety_margin
        selected_pred = None
        selected_is_cold = False

        for r in self.frontier:
            r_config = r['config']
            r_acc = r['acc']
            r_mem = r.get('mem_mb', 0.0)

            # Strict memory check
            if r_mem > 0 and resource_state is not None:
                if resource_state.mem_available_mb is not None and r_mem > resource_state.mem_available_mb:
                    continue

            # Feature Engineering
            cfg_feat = extract_config_features(r_config)
            flops_per_speed = cfg_feat["approx_flops"] / speed

            row = {
                "width_mult": cfg_feat["width_mult"],
                "bit_width": cfg_feat["bit_width"],
                "approx_flops": cfg_feat["approx_flops"],
                "approx_params": cfg_feat["approx_params"],
                "flops_per_speed": flops_per_speed,
                "cpu_pct": cpu_pct,
                "mem_available_mb": mem_available_mb,
                "thermal_c": thermal_c,
                "device_speed_score": speed,
                "cpu_cores": cpu_cores,
                "ram_gb": ram_gb,
                "has_cuda": has_cuda,
            }
            condition = getattr(resource_state, "resource_condition", None) or "baseline"
            condition_code = 1 if condition == "contention" else 0

            try:
                feature_values = [row[feat] for feat in FEATURES]
                if isinstance(self.surrogate_model, MeasuredFrontierLatencyModel):
                    feature_values.append(condition_code)
                pred_latency = float(self.surrogate_model.predict([feature_values])[0])
            except Exception:
                # Physics fallback if predict fails
                pred_latency = (cfg_feat["approx_flops"] / speed) * 1000.0
                pred_latency *= (1.0 + cpu_pct / 100.0)
                if thermal_c > 70:
                    pred_latency *= 1.3

            # Scale by configuration-specific online calibration.
            config_factor = self.k_adapt_by_config.get(r_config, 1.0)
            pred_latency_adj = pred_latency * config_factor

            is_cold = r_config != self.current_config
            safe_latencies.append(pred_latency_adj)

            # Fetch standard deviation and scale by k_adapt
            std_latency = r.get('std_ms', 1.0) * self.k_adapt

            # Calculate upper bound using risk multiplier
            risk_averse_pred = pred_latency_adj
            if self.latency_policy in ("avg", "p50"):
                risk_averse_pred += self.k_risk * std_latency

            # Apply switching cost if candidate config differs from currently active config
            if r_config != self.current_config:
                risk_averse_pred += self.switching_penalty_ms
            ranked_candidates.append((risk_averse_pred, pred_latency_adj, r_config))

            if risk_averse_pred <= target_budget:
                if r_acc > best_acc:
                    best_acc = r_acc
                    candidate = r_config
                    selected_pred = pred_latency_adj
                    selected_is_cold = is_cold

        self.last_min_safe_latency = min(safe_latencies) if safe_latencies else None
        self.last_budget_feasible = (
            self.last_min_safe_latency is not None
            and self.last_min_safe_latency <= latency_budget_ms
        )
        self.last_selection_status = "feasible_budget" if self.last_budget_feasible else "infeasible_budget"

        if candidate is not None:
            if candidate != self.current_config:
                self.current_config = candidate
                self._last_switch_time = now
            self.last_predicted_latency = selected_pred
            self.last_selected_config = self.current_config
            self.last_prediction_is_cold = selected_is_cold
            return self.current_config
        else:
            # Fallback uses the lowest predicted latency in the active condition.
            fallback_latency, fallback_pred, fallback_config = min(
                ranked_candidates, key=lambda item: item[0]
            )
            fallback_is_cold = fallback_config != self.current_config
            if fallback_config != self.current_config:
                self.current_config = fallback_config
                self._last_switch_time = now
            self.last_predicted_latency = fallback_pred
            self.last_selected_config = self.current_config
            self.last_prediction_is_cold = fallback_is_cold
            self.last_selection_status = "infeasible_budget" if not self.last_budget_feasible else "no_candidate_with_safety_margin"
            return self.current_config