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
    "width_mult", "bit_width", "depth_mult",
    "approx_flops", "approx_params",
    "flops_per_speed",
    "cpu_pct", "mem_available_mb", "thermal_c",
    "device_speed_score", "cpu_cores", "ram_gb", "has_cuda",
]

def extract_config_features(config):
    """Extracts width_mult, bit_width, depth_mult, approx_flops, approx_params
    from a config tuple or dict."""
    if isinstance(config, dict) and "config" in config:
        config = config["config"]

    width = 32.0
    bit_width = 16.0
    depth = 1.0

    if isinstance(config, (list, tuple)):
        if len(config) >= 1:
            width = float(config[0])
        if len(config) >= 2:
            bit_width = float(config[1])
        if len(config) >= 3:
            depth = float(config[2])
    elif isinstance(config, (int, float)):
        width = float(config)

    width_mult = width / 32.0
    depth_mult = depth

    # Approximate FLOPs and Params
    approx_flops = (width_mult ** 2) * depth_mult * 1e6
    approx_params = (width_mult ** 2) * depth_mult * 1e5

    return {
        "width_mult": width_mult,
        "bit_width": bit_width,
        "depth_mult": depth_mult,
        "approx_flops": approx_flops,
        "approx_params": approx_params,
    }

class PhysicsSurrogateModel:
    """A fallback physics-based latency estimator when ML models are not yet trained."""
    
    def predict(self, X):
        preds = []
        for row in X:
            approx_flops = row[3]
            device_speed = row[9]
            cpu_pct = row[6]
            thermal_c = row[8]
            
            # latency (ms) ≈ (flops / speed) * 1000.0 * contention
            base_lat = (approx_flops / device_speed) * 1000.0
            contention = 1.0 + (cpu_pct / 100.0)
            if thermal_c > 70:
                contention *= 1.2
            preds.append(base_lat * contention)
        return preds

class SurrogateBackedController:
    """Uses a machine learning surrogate model to predict model config latencies
    under current hardware & resource state, selecting the best fit config."""

    def __init__(self, frontier, surrogate_model=None, baseline_config=None, safety_margin=0.9, min_dwell_s=2.0):
        self.frontier = frontier
        self.surrogate_model = surrogate_model if surrogate_model is not None else PhysicsSurrogateModel()
        self.baseline_config = baseline_config if baseline_config is not None else frontier[0]['config']
        self.current_config = self.baseline_config
        self.safety_margin = safety_margin
        self._last_switch_time = time.time()
        self.min_dwell_s = min_dwell_s

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
        target_budget = latency_budget_ms * self.safety_margin

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
                "depth_mult": cfg_feat["depth_mult"],
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

            try:
                feature_values = [row[feat] for feat in FEATURES]
                pred_latency = float(self.surrogate_model.predict([feature_values])[0])
            except Exception:
                # Physics fallback if predict fails
                pred_latency = (cfg_feat["approx_flops"] / speed) * 1000.0
                pred_latency *= (1.0 + cpu_pct / 100.0)
                if thermal_c > 70:
                    pred_latency *= 1.3

            if pred_latency <= target_budget:
                if r_acc > best_acc:
                    best_acc = r_acc
                    candidate = r_config

        if candidate is not None:
            if candidate != self.current_config:
                self.current_config = candidate
                self._last_switch_time = now
            return self.current_config
        else:
            # Fallback: lowest latency config
            fallback_config = self.frontier[0]['config']
            if fallback_config != self.current_config:
                self.current_config = fallback_config
                self._last_switch_time = now
            return self.current_config