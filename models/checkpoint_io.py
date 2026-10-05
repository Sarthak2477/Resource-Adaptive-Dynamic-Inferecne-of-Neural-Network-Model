import hashlib
from pathlib import Path

import torch


def resolve_usm_checkpoint(path=None):
    if path is not None:
        checkpoint_path = Path(path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    project_root = Path(__file__).resolve().parent.parent
    candidates = (
        project_root / "models" / "checkpoint" / "us_resnet_epoch100_checkpoint.pt.zip",
        project_root / "models" / "checkpoint" / "us_resnet_epoch100_checkpoint.pt",
    )
    for checkpoint_path in candidates:
        if checkpoint_path.is_file():
            return checkpoint_path

    raise FileNotFoundError(
        "No trained USM checkpoint was found. Pass --checkpoint PATH; "
        "randomly initialized weights are not accepted for this experiment."
    )


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_checkpoint(model, path, device):
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint has no model state dictionary: {checkpoint_path}")

    cleaned_state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(cleaned_state_dict, strict=True)
    checkpoint_metadata = {}
    if isinstance(checkpoint, dict):
        for key in ("training_mode", "architecture", "width_mult", "bit_width", "epochs", "seed"):
            if key in checkpoint:
                checkpoint_metadata[key] = checkpoint[key]

    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "checkpoint_metadata": checkpoint_metadata,
    }


def load_fixed_width_checkpoint(model, path, device, width_mult):
    provenance = load_model_checkpoint(model, path, device)
    metadata = provenance["checkpoint_metadata"]
    if metadata.get("training_mode") != "separately_trained_fixed_width_fp32":
        raise ValueError("Checkpoint is not a separately trained fixed-width FP32 model")
    if float(metadata.get("width_mult", -1.0)) != float(width_mult):
        raise ValueError("Fixed-width checkpoint does not match the selected width")
    if int(metadata.get("bit_width", -1)) != 32:
        raise ValueError("Fixed-width checkpoint must use FP32 (bit_width=32)")
    return provenance