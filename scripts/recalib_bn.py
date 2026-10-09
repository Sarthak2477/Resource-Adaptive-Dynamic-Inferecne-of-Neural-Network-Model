"""Per-configuration BN statistics cache for the USM width/bit-width model.

Problem this solves
-------------------
`recalibrate_bn` writes into one shared set of BN buffers. Calibrating every
(width, bit) candidate in a loop leaves only the *last* candidate's statistics
in the model, so every later configuration is served with stale statistics.

What this module does
---------------------
1. Calibrates each candidate configuration once and snapshots the model's
   registered buffers right after that calibration (BNBank.build).
2. Switches configurations by re-pointing the module buffers at the cached
   tensors (BNBank.activate). No tensor data is copied or recalibrated.
3. Optionally persists the bank to disk, keyed by everything that determines
   the statistics, so later runs skip calibration entirely.

Which buffers are cached
------------------------
Every entry in each module's `_buffers` (BN running_mean / running_var /
num_batches_tracked, plus any activation-quantizer observer buffers). This is
deliberately broad: it needs no knowledge of how ResnetBatchNorm2d or ActQuant
are implemented, and the snapshot is taken immediately after calibration of a
single config, so it is exactly the state that config needs.

Rules for correct use
---------------------
* Keep the model in eval() mode while serving. In train mode BN updates its
  buffers in place, which would corrupt the shared cached tensors.
* Call `activate(config)` AFTER `set_model_width` / `set_model_bit_width`.
"""

import hashlib
import json
import random
import time
from pathlib import Path

import torch


def config_key(config):
    return (float(config[0]), int(config[1]))


def _buffer_slots(model):
    """(qualified_name, module, buffer_name) for every registered buffer."""
    slots = []
    for module_name, module in model.named_modules():
        for buffer_name, tensor in module._buffers.items():
            if tensor is not None:
                qualified = f"{module_name}.{buffer_name}" if module_name else buffer_name
                slots.append((qualified, module, buffer_name))
    return slots


def make_cache_key(checkpoint_sha256, bn_indices, batches, seed, configs):
    payload = json.dumps({
        "checkpoint": checkpoint_sha256,
        "bn_indices": [int(index) for index in bn_indices],
        "batches": int(batches),
        "seed": int(seed),
        "configs": [list(config_key(c)) for c in configs],
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class BNBank:
    """Cached buffer snapshots, one per (width, bit_width) configuration."""

    def __init__(self, model, snapshots):
        self._slots = _buffer_slots(model)
        self._names = [name for name, _, _ in self._slots]
        for key, snapshot in snapshots.items():
            if set(snapshot) != set(self._names):
                raise ValueError(f"Cached snapshot for {key} does not match the model's buffers")
        self.snapshots = snapshots
        self.active = None

    @property
    def configs(self):
        return sorted(self.snapshots)

    # ----- building ---------------------------------------------------
    @classmethod
    def build(cls, model, loader, configs, device, recalibrate_fn, batches, seed,
              verbose=True):
        """Calibrate each config once and snapshot buffers right afterwards."""
        slots = _buffer_slots(model)
        snapshots = {}
        for index, config in enumerate(configs):
            width, bits = config_key(config)
            candidate_seed = seed + index
            random.seed(candidate_seed)
            torch.manual_seed(candidate_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(candidate_seed)
            started = time.perf_counter()
            recalibrate_fn(model, loader, width, bits, device, num_batches=batches)
            snapshots[(width, bits)] = {
                name: module._buffers[buffer_name].detach().clone()
                for name, module, buffer_name in slots
            }
            if verbose:
                print(f"[bn_cache] calibrated width={width:g} bits={bits} "
                      f"in {time.perf_counter() - started:.2f}s")
        model.eval()
        return cls(model, snapshots)

    # ----- switching --------------------------------------------------
    def activate(self, config):
        """Point every module buffer at this config's cached tensors."""
        key = config_key(config)
        snapshot = self.snapshots[key]
        for name, module, buffer_name in self._slots:
            module._buffers[buffer_name] = snapshot[name]
        self.active = key

    # ----- diagnostics ------------------------------------------------
    def difference_report(self):
        """How many floating-point buffers differ from the reference config (the
        last in sorted order, i.e. the largest width / highest bits). If everything is identical,
        the statistics do not depend on the config and the original bug was
        not BN; if they differ, the bank is doing real work."""
        keys = self.configs
        reference = self.snapshots[keys[-1]]
        report = {}
        for key in keys:
            differing = sum(
                1 for name in self._names
                if reference[name].dtype.is_floating_point
                and not torch.equal(reference[name], self.snapshots[key][name])
            )
            report[key] = differing
        return report

    # ----- persistence ------------------------------------------------
    def save(self, path, cache_key):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_key": cache_key,
            "snapshots": {
                key: {name: tensor.cpu() for name, tensor in snapshot.items()}
                for key, snapshot in self.snapshots.items()
            },
        }
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, model, path, cache_key, device):
        """Return a bank, or None if the file is missing, stale, or mismatched."""
        path = Path(path)
        if not path.exists():
            return None
        payload = torch.load(path, map_location="cpu")
        if payload.get("cache_key") != cache_key:
            return None
        snapshots = {
            key: {name: tensor.to(device) for name, tensor in snapshot.items()}
            for key, snapshot in payload["snapshots"].items()
        }
        try:
            return cls(model, snapshots)
        except ValueError:
            return None


def build_or_load_bn_bank(model, loader, configs, device, recalibrate_fn, batches,
                          seed, cache_dir=None, checkpoint_sha256="", bn_indices=(),
                          verbose=True):
    """Load the bank from disk when valid, otherwise calibrate and save it."""
    cache_key = make_cache_key(checkpoint_sha256, bn_indices, batches, seed, configs)
    path = Path(cache_dir) / f"bn_bank_{cache_key}.pt" if cache_dir else None
    if path is not None:
        bank = BNBank.load(model, path, cache_key, device)
        if bank is not None:
            if verbose:
                print(f"[bn_cache] loaded cached BN statistics from {path}")
            return bank
    bank = BNBank.build(model, loader, configs, device, recalibrate_fn, batches, seed,
                        verbose=verbose)
    if path is not None:
        bank.save(path, cache_key)
        if verbose:
            print(f"[bn_cache] saved BN statistics to {path}")
    return bank