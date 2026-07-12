from __future__ import annotations

import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch.amp import GradScaler

from gptff.trainer.config import TrainingConfig
from gptff.trainer.scheduler import Scheduler

CHECKPOINT_VERSION: Final = 1
TRAINING_CHECKPOINT_TYPE: Final = "training_resume"
INFERENCE_CHECKPOINT_TYPE: Final = "inference"


def save_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Scheduler | None,
    scaler: GradScaler,
    config: TrainingConfig,
    *,
    epoch: int,
    best_energy_mae: float,
    best_force_mae: float,
    is_best_energy: bool,
    is_best_force: bool,
    rng_states: Sequence[Mapping[str, Any]],
    world_size: int,
    data_state: Mapping[str, int | None],
) -> None:
    model_state = _model_checkpoint(
        model,
        config,
        epoch=epoch,
        best_energy_mae=best_energy_mae,
        best_force_mae=best_force_mae,
    )
    training_state = {
        **model_state,
        "checkpoint_type": TRAINING_CHECKPOINT_TYPE,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "amp_scaler_enabled": bool(scaler.is_enabled()),
        "rng_states": list(rng_states),
        "world_size": int(world_size),
        "data_state": dict(data_state),
    }
    inference_state = {
        **model_state,
        "checkpoint_type": INFERENCE_CHECKPOINT_TYPE,
    }
    if is_best_energy:
        _atomic_torch_save(inference_state, output_dir / "bestE.pt")
    if is_best_force:
        _atomic_torch_save(inference_state, output_dir / "bestF.pt")
    # Commit last.pt only after any matching best checkpoints are durable. If a
    # best-file write fails, the previous last.pt remains the resume boundary.
    _atomic_torch_save(training_state, output_dir / "last.pt")


def load_training_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device,
) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device(device),
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Training checkpoint must contain a mapping: {checkpoint_path}")
    checkpoint_type = checkpoint.get("checkpoint_type")
    if checkpoint_type != TRAINING_CHECKPOINT_TYPE:
        actual = "legacy/inference" if checkpoint_type is None else repr(checkpoint_type)
        raise ValueError(
            f"Checkpoint {checkpoint_path} has type {actual} and cannot resume training. "
            "Use a training_resume checkpoint, typically last.pt."
        )
    version = checkpoint.get("checkpoint_version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported training checkpoint version {version!r}; "
            f"expected {CHECKPOINT_VERSION}."
        )
    required = {
        "epoch",
        "state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "amp_scaler_enabled",
        "best_energy_mae",
        "best_force_mae",
        "training_config",
        "model_config",
        "rng_states",
        "world_size",
        "data_state",
    }
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(f"Training checkpoint is missing required fields: {missing}.")
    return checkpoint


def capture_local_rng_state(
    generators: Mapping[str, torch.Generator],
    *,
    device: str | torch.device,
) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    torch_device = torch.device(device)
    cuda_state = None
    if torch_device.type == "cuda":
        cuda_state = torch.cuda.get_rng_state(torch_device).cpu()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tobytes(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        # Object collectives cannot reliably pickle Tensor storage on every
        # supported PyTorch/backend combination. Bytes are compact, safe for
        # weights_only checkpoint loading, and portable across ranks.
        "torch_cpu": _rng_tensor_to_bytes(torch.get_rng_state()),
        "torch_cuda": None if cuda_state is None else _rng_tensor_to_bytes(cuda_state),
        "data_loader_generators": {
            name: _rng_tensor_to_bytes(generator.get_state())
            for name, generator in generators.items()
        },
    }


def restore_local_rng_state(
    state: Mapping[str, Any],
    generators: Mapping[str, torch.Generator],
    *,
    device: str | torch.device,
) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.frombuffer(numpy_state["keys"], dtype=np.uint32).copy(),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(_rng_bytes_to_tensor(state["torch_cpu"]))
    torch_device = torch.device(device)
    cuda_state = state.get("torch_cuda")
    if torch_device.type == "cuda":
        if cuda_state is None:
            raise ValueError("Checkpoint does not contain CUDA RNG state for CUDA resume.")
        torch.cuda.set_rng_state(_rng_bytes_to_tensor(cuda_state), torch_device)
    elif cuda_state is not None:
        raise ValueError("Checkpoint contains CUDA RNG state but resume device is not CUDA.")

    saved_generators = state["data_loader_generators"]
    if set(saved_generators) != set(generators):
        raise ValueError(
            "DataLoader generator names do not match checkpoint: "
            f"saved={sorted(saved_generators)}, current={sorted(generators)}."
        )
    for name, generator in generators.items():
        generator.set_state(_rng_bytes_to_tensor(saved_generators[name]))


def resume_config_differences(
    checkpoint: Mapping[str, Any],
    config: TrainingConfig,
    *,
    world_size: int,
    data_state: Mapping[str, int | None],
) -> list[tuple[str, Any, Any]]:
    saved = checkpoint["training_config"]
    current = config.checkpoint_dict()
    differences: list[tuple[str, Any, Any]] = []

    _append_difference(
        differences,
        "model_config",
        checkpoint["model_config"],
        config.model.to_dict(),
    )
    for section in ("optimizer", "loss"):
        _append_difference(differences, section, saved[section], current[section])
    _append_difference(
        differences,
        "element_references",
        saved["element_references"],
        current["element_references"],
    )
    for key in (
        "dataset_path",
        "dataset_format",
        "validation_fraction",
        "test_fraction",
        "split_seed",
        "group_by_material",
    ):
        _append_difference(
            differences,
            f"data.{key}",
            saved["data"].get(key),
            current["data"].get(key),
        )
    for key in (
        "epochs",
        "batch_size",
        "amp",
        "grad_clip_norm",
        "seed",
        "deterministic",
    ):
        _append_difference(
            differences,
            f"training.{key}",
            saved["training"].get(key),
            current["training"].get(key),
        )
    _append_difference(
        differences,
        "world_size",
        int(checkpoint["world_size"]),
        int(world_size),
    )
    _append_difference(
        differences,
        "data_state",
        checkpoint["data_state"],
        dict(data_state),
    )
    return differences


def format_resume_config_error(
    differences: Sequence[tuple[str, Any, Any]],
) -> str:
    lines = ["Configuration mismatch prevents resume:"]
    lines.extend(
        f"  - {field}: saved={saved!r}, current={current!r}"
        for field, saved, current in differences
    )
    lines.append("Use the original training configuration or start a separate training run.")
    return "\n".join(lines)


def _model_checkpoint(
    model: torch.nn.Module,
    config: TrainingConfig,
    *,
    epoch: int,
    best_energy_mae: float,
    best_force_mae: float,
) -> dict[str, Any]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "epoch": int(epoch),
        "state_dict": model.state_dict(),
        "best_energy_mae": float(best_energy_mae),
        "best_force_mae": float(best_force_mae),
        "best_validation_metric": float(best_force_mae),
        "training_config": config.checkpoint_dict(),
        "model_name": "GPTFF",
        "model_config": config.model.to_dict(),
    }


def _atomic_torch_save(state: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.{os.getpid()}.",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as file:
            torch.save(dict(state), file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.close(file_descriptor)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise


def _append_difference(
    differences: list[tuple[str, Any, Any]],
    field: str,
    saved: Any,
    current: Any,
) -> None:
    if saved != current:
        differences.append((field, saved, current))


def _rng_tensor_to_bytes(state: torch.Tensor) -> bytes:
    return state.detach().cpu().contiguous().numpy().tobytes()


def _rng_bytes_to_tensor(state: bytes) -> torch.Tensor:
    return torch.from_numpy(np.frombuffer(state, dtype=np.uint8).copy())
