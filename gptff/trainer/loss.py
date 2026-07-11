from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from gptff.inference import predict_energy_forces_stress
from gptff.trainer.config import TrainingConfig
from gptff.utils.labels import EV_PER_ANG3_TO_GPA


@dataclass
class BatchLoss:
    loss: torch.Tensor
    energy_mae: torch.Tensor | None
    force_mae: torch.Tensor | None
    stress_mae: torch.Tensor | None
    batch_size: int
    force_count: int = 0
    stress_count: int = 0


def mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(target - prediction))


def loss_weight_active(weight: float) -> bool:
    return float(weight) > 0.0


def validate_required_labels(batch, config: TrainingConfig) -> None:
    if not loss_weight_active(config.energy_loss_weight):
        raise ValueError("energy_loss_weight must be positive.")
    if not loss_weight_active(config.force_loss_weight):
        raise ValueError("force_loss_weight must be positive.")
    if batch.energy is None:
        raise ValueError("energy labels are required.")
    if batch.forces is None:
        raise ValueError("force labels are required.")
    if loss_weight_active(config.stress_loss_weight) and batch.stress is None:
        raise ValueError("stress labels are required when stress_loss_weight > 0.")


def compute_batch_loss(
    model: torch.nn.Module,
    batch,
    criterion: nn.Module,
    config: TrainingConfig,
    *,
    create_graph: bool,
) -> BatchLoss:
    validate_required_labels(batch, config)
    compute_stress = loss_weight_active(config.stress_loss_weight)

    energy_pred, force_pred, stress_pred = predict_energy_forces_stress(
        model,
        batch,
        create_graph=create_graph,
        compute_stress=compute_stress,
    )
    loss = energy_pred.new_zeros(())
    energy_mae = None
    force_mae = None
    stress_mae = None
    force_count = 0
    stress_count = 0

    num_atoms = batch.num_atoms.to(dtype=energy_pred.dtype)
    energy_per_atom = energy_pred.view(-1) / num_atoms
    target_energy = batch.energy.view(-1) / num_atoms
    energy_loss = criterion(energy_per_atom, target_energy)
    loss = loss + config.energy_loss_weight * energy_loss
    energy_mae = mae(energy_per_atom.detach(), target_energy.detach())

    force_loss = criterion(force_pred.reshape(-1), batch.forces.reshape(-1))
    loss = loss + config.force_loss_weight * force_loss
    force_mae = mae(force_pred.detach().reshape(-1), batch.forces.detach().reshape(-1))
    force_count = int(batch.forces.numel())

    if compute_stress:
        # Dataset stress labels and reported stress metrics use GPa.
        stress_pred_gpa = stress_pred * EV_PER_ANG3_TO_GPA
        stress_loss = criterion(stress_pred_gpa.reshape(-1), batch.stress.reshape(-1))
        loss = loss + config.stress_loss_weight * stress_loss
        stress_mae = mae(
            stress_pred_gpa.detach().reshape(-1),
            batch.stress.detach().reshape(-1),
        )
        stress_count = int(batch.stress.numel())

    return BatchLoss(
        loss=loss,
        energy_mae=energy_mae,
        force_mae=force_mae,
        stress_mae=stress_mae,
        batch_size=int(batch.num_atoms.shape[0]),
        force_count=force_count,
        stress_count=stress_count,
    )
