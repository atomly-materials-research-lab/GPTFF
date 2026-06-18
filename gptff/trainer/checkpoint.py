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
    best_validation_metric: float,
    is_best: bool,
) -> None:
    model_state = {
        "epoch": int(epoch),
        "state_dict": model.state_dict(),
        "best_validation_metric": float(best_validation_metric),
        "training_config": config.checkpoint_dict(),
        "model_name": "GPTFF",
        "model_config": config.to_model_config().to_dict(),
    }
    current_path = output_dir / "last.pt"
    torch.save(model_state, current_path)
    if is_best:
        shutil.copyfile(current_path, output_dir / "best.pt")
