import time
import psutil
import platform
import torch
from dataclasses import dataclass, field
from collections import deque

@dataclass
class ResourceState:
    cpu_pct: float
    mem_available_mb: float
    battery_pct: float | None
    thermal_c: float | None
    timestamp: float
    resource_condition: str | None = None

def device_speed_score(device, matmul_size=256, n_warmup=5, n_reps=30):
    """A single scalar proxying this device's matmul throughput.
    Total wall time: well under 200ms on virtually any hardware."""
    if isinstance(device, str):
        device = torch.device(device)
    try:
        x = torch.randn(64, matmul_size, device=device)
        w = torch.randn(matmul_size, matmul_size, device=device)

        # Warmup: absorb CUDA context init, cuDNN algo selection, allocator
        # warmup, thread-pool spin-up -- none of which reflect steady-state speed
        for _ in range(n_warmup):
            _ = x @ w
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_reps):
            _ = x @ w
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / n_reps
        return 1.0 / max(elapsed, 1e-9)   # higher = faster device; matmuls/sec, roughly
    except Exception:
        # Fallback if torch or device is not working
        return 1000.0

def hardware_fingerprint(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    speed = device_speed_score(device)
    return {
        "device_speed_score": speed,
        "cpu_cores": psutil.cpu_count(logical=False) or 1,
        "cpu_cores_logical": psutil.cpu_count(logical=True) or 1,
        "ram_gb": psutil.virtual_memory().total / 1e9,
        "has_cuda": float(torch.cuda.is_available() and device.type == "cuda"),
        "arch": platform.machine(),   # e.g. "x86_64", "aarch64"
    }

class ResourceMonitor:
    """Tiered telemetry: cheap tiers read every call, expensive tiers
    read on a schedule and cached in between."""

    def __init__(self, live_refresh_s=0.5, thermal_path="/sys/class/thermal/thermal_zone0/temp",
                 smoothing_alpha=0.3, history_len=50):
        self.live_refresh_s = live_refresh_s
        self.thermal_path = thermal_path
        self.alpha = smoothing_alpha
        self._last_live_read = 0.0
        self._cached = None
        self._smoothed = None
        self.history = deque(maxlen=history_len)
        self._hw_fingerprint = None

    def get_hardware_fingerprint(self, device=None):
        if self._hw_fingerprint is None:
            self._hw_fingerprint = hardware_fingerprint(device)
        return self._hw_fingerprint

    def _read_live(self) -> ResourceState:
        cpu = psutil.cpu_percent(interval=None)          # non-blocking, uses last sample
        mem = psutil.virtual_memory().available / 1e6
        battery = psutil.sensors_battery()
        batt_pct = battery.percent if battery else None
        try:
            with open(self.thermal_path) as f:
                thermal_c = int(f.read().strip()) / 1000.0
        except (FileNotFoundError, PermissionError):
            thermal_c = None
        return ResourceState(cpu, mem, batt_pct, thermal_c, time.time())

    def read(self) -> ResourceState:
        now = time.time()
        # Tier 1/2: reuse the cached reading if it's still fresh
        if self._cached is None or (now - self._last_live_read) >= self.live_refresh_s:
            self._cached = self._read_live()
            self._last_live_read = now

        # Exponential smoothing (prevents short spikes from triggering switches)
        if self._smoothed is None:
            self._smoothed = self._cached
        else:
            a = self.alpha
            self._smoothed = ResourceState(
                cpu_pct=a * self._cached.cpu_pct + (1 - a) * self._smoothed.cpu_pct,
                mem_available_mb=a * self._cached.mem_available_mb + (1 - a) * self._smoothed.mem_available_mb,
                battery_pct=self._cached.battery_pct,   # discrete, don't smooth
                thermal_c=(a * self._cached.thermal_c + (1 - a) * self._smoothed.thermal_c)
                            if self._cached.thermal_c is not None else None,
                timestamp=now,
            )

        self.history.append(self._smoothed)
        return self._smoothed

    def recalibrate_latency(self, model, sample_input, config):
        core = model.module if hasattr(model, 'module') else model
        core.set_width(config[0]); core.set_bit_width(config[1])
        core.eval()
        with torch.no_grad():
            start = time.perf_counter()
            core(sample_input)
            torch.cuda.synchronize() if sample_input.is_cuda else None
        return (time.perf_counter() - start) * 1000  # ms

    def estimate_thermal_throttle(self):
        """
        Heuristic: if CPU is pegged and temp is rising, apply a penalty.
        This is a simplified proxy for hardware throttling.
        """
        if len(self.history) < 3: return 0.0
        # very basic: if CPU > 90% and temp is going up
        recent = list(self.history)[-3:]
        cpu_avg = sum(r.cpu_pct for r in recent) / len(recent)
        temp_now = recent[-1].thermal_c
        temp_prev = recent[-2].thermal_c
        if cpu_avg > 90 and temp_now is not None and temp_prev is not None and temp_now > temp_prev:
            return 1.1 + min(0.5, (temp_now - 75) / 25.0)   # up to +50%
        return 1.0

if __name__ == "__main__":
    monitor = ResourceMonitor()
    print("Read smoothed:", monitor.read())
    print("Read live:", monitor._read_live())
    print("Hardware Fingerprint:", monitor.get_hardware_fingerprint())