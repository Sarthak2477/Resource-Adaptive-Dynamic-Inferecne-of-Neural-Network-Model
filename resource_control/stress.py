import threading
import time
import weakref

import torch


class ResourceStress:
    """Generate repeatable CPU and optional GPU compute contention."""

    def __init__(self, device, cpu=True, gpu=True, matrix_size=512, idle_s=0.0):
        self.device = torch.device(device)
        self.cpu = cpu
        self.gpu = gpu and self.device.type == "cuda"
        self.matrix_size = matrix_size
        self.idle_s = max(0.0, float(idle_s))
        self._stop = threading.Event()
        self._enabled = threading.Event()
        self._ready = threading.Event()
        self._ready_lock = threading.Lock()
        self._ready_count = 0
        self._condition_lock = threading.Lock()
        self._condition = "baseline"
        self._threads = []
        self._schedule_thread = None
        weakref.finalize(self, self._stop.set)

    def start(self):
        if not self._threads:
            self._stop.clear()
            self._ready.clear()
            self._ready_count = 0
            if self.cpu:
                self._threads.append(threading.Thread(target=self._run_cpu, daemon=True))
            if self.gpu:
                self._threads.append(threading.Thread(target=self._run_gpu, daemon=True))
            for thread in self._threads:
                thread.start()
            if not self._ready.wait(timeout=5.0):
                self.close()
                raise RuntimeError("Resource stress workers failed to initialize")

    def set_enabled(self, enabled):
        if enabled:
            self.start()
            self._enabled.set()
        else:
            self._enabled.clear()

    def set_condition(self, condition):
        if condition not in ("baseline", "contention"):
            raise ValueError(f"Unsupported stress condition: {condition}")
        self.set_enabled(condition == "contention")
        with self._condition_lock:
            self._condition = condition

    def current_condition(self):
        with self._condition_lock:
            return self._condition

    def schedule(self, requests, replay_origin):
        if self._schedule_thread is not None:
            raise RuntimeError("Resource condition schedule is already running")

        def apply_schedule():
            for request_index, request in enumerate(requests):
                target = replay_origin + float(request.get("arrival_s", request_index * 0.01))
                while not self._stop.is_set():
                    remaining = target - time.perf_counter()
                    if remaining <= 0:
                        break
                    self._stop.wait(min(remaining, 0.01))
                if self._stop.is_set():
                    return
                scenario = request.get("resource_scenario", "baseline")
                scenario = {
                    "synthetic_baseline": "baseline",
                    "synthetic_contention": "contention",
                }.get(scenario, scenario)
                self.set_condition(scenario)

        self._schedule_thread = threading.Thread(target=apply_schedule, daemon=True)
        self._schedule_thread.start()

    def wait_for_schedule(self):
        if self._schedule_thread is not None:
            self._schedule_thread.join(timeout=5.0)
            if self._schedule_thread.is_alive():
                raise RuntimeError("Resource condition scheduler did not finish")
            self._schedule_thread = None

    def _run_cpu(self):
        size = self.matrix_size
        left = torch.randn(size, size, device="cpu")
        right = torch.randn(size, size, device="cpu")
        self._mark_ready()
        while not self._stop.is_set():
            if self._enabled.wait(timeout=0.01):
                torch.mm(left, right)
                if self.idle_s:
                    self._stop.wait(self.idle_s)

    def _run_gpu(self):
        size = self.matrix_size
        left = torch.randn(size, size, device=self.device)
        right = torch.randn(size, size, device=self.device)
        self._mark_ready()
        while not self._stop.is_set():
            if self._enabled.wait(timeout=0.01):
                torch.mm(left, right)
                torch.cuda.synchronize(self.device)
                if self.idle_s:
                    self._stop.wait(self.idle_s)

    def _mark_ready(self):
        with self._ready_lock:
            self._ready_count += 1
            if self._ready_count == len(self._threads):
                self._ready.set()

    def close(self):
        self._stop.set()
        self._enabled.set()
        if self._schedule_thread is not None:
            self._schedule_thread.join(timeout=5.0)
            self._schedule_thread = None
        for thread in self._threads:
            thread.join(timeout=5.0)
            if thread.is_alive():
                raise RuntimeError("Resource stress worker did not stop")
        self._threads.clear()
        self._enabled.clear()
        self._stop.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()