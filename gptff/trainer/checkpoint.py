from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch.cuda.amp import GradScaler

from gptff.trainer.config import TrainingConfig
from gptff.trainer.scheduler import CosineAnnealingWarmupRestarts


@dataclass(frozen=True)
class LoadedCheckpoint:
    epoch: int
    best_mae_error: float
    training_config: Optional[Dict[str, Any]]
    model_config: Optional[Dict[str, Any]]
    label_config: Optional[Dict[str, Any]]


def resolve_checkpoint_path(config: TrainingConfig, output_dir: Path) -> Path:
    if config.checkpoint_path:
        return Path(config.checkpoint_path)
    return output_dir / "curr_checkpoint.pth"


def load_training_checkpoint(
    checkpoint_path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
) -> LoadedCheckpoint:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(state["state_dict"])
    optimizer.load_state_dict(state["optimizer"])

    epoch = int(state.get("epoch", 0))
    if scheduler is not None:
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        else:
            scheduler.step(epoch)
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])

    return LoadedCheckpoint(
        epoch=epoch,
        best_mae_error=float(state.get("best_mae_error", 1e12)),
        training_config=state.get("training_config", state.get("cfg")),
        model_config=state.get("model_config"),
        label_config=state.get("label_config"),
    )


def save_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    *,
    epoch: int,
    best_mae_error: float,
    is_best: bool,
    scheduler: Optional[CosineAnnealingWarmupRestarts] = None,
    scaler: Optional[GradScaler] = None,
) -> None:
    model_state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "best_mae_error": best_mae_error,
        "optimizer": optimizer.state_dict(),
        "cfg": config.checkpoint_dict(),
        "training_config": config.checkpoint_dict(),
        "label_config": asdict(config.to_label_config()),
        "model_name": "tModLodaer_t" if config.transformer_activate else "GPTFFNet",
        "model_config": config.to_model_config().to_dict(),
    }
    if scheduler is not None:
        model_state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        model_state["scaler"] = scaler.state_dict()
    current_path = output_dir / "curr_checkpoint.pth"
    torch.save(model_state, current_path)
    if is_best:
        shutil.copyfile(current_path, output_dir / "best_checkpoint.pth")
