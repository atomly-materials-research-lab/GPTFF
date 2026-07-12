from __future__ import annotations

import shutil
from pathlib import Path

import torch

from gptff.trainer.config import TrainingConfig


def save_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    config: TrainingConfig,
    *,
    epoch: int,
    best_energy_mae: float,
    best_force_mae: float,
    is_best_energy: bool,
    is_best_force: bool,
) -> None:
    model_state = {
        "epoch": int(epoch),
        "state_dict": model.state_dict(),
        "best_energy_mae": float(best_energy_mae),
        "best_force_mae": float(best_force_mae),
        "best_validation_metric": float(best_force_mae),
        "training_config": config.checkpoint_dict(),
        "model_name": "GPTFF",
        "model_config": config.model.to_dict(),
    }
    current_path = output_dir / "last.pt"
    torch.save(model_state, current_path)
    if is_best_energy:
        shutil.copyfile(current_path, output_dir / "bestE.pt")
    if is_best_force:
        shutil.copyfile(current_path, output_dir / "bestF.pt")
