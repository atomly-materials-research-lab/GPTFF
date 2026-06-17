from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from gptff.model.prediction import predict_energy_forces_stress
from gptff.trainer.config import TrainingConfig


@dataclass
class BatchLoss:
    loss: torch.Tensor
    energy_mae: Optional[torch.Tensor]
    force_mae: Optional[torch.Tensor]
    stress_mae: Optional[torch.Tensor]
    batch_size: int
    force_count: int = 0
    stress_count: int = 0


def mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(target - prediction))


def loss_weight_active(weight: float) -> bool:
    return float(weight) > 0.0


def validate_required_labels(batch, config: TrainingConfig) -> None:
    active_terms = [
        loss_weight_active(config.w1),
        loss_weight_active(config.w2),
        loss_weight_active(config.w3),
    ]
    if not any(active_terms):
        raise ValueError("At least one loss weight must be positive.")
    if active_terms[0] and batch.energy is None:
        raise ValueError("energy labels are required when weight_energy > 0.")
    if active_terms[1] and batch.forces is None:
        raise ValueError("force labels are required when weight_force > 0.")
    if active_terms[2] and batch.stress is None:
        raise ValueError("stress labels are required when weight_stress > 0.")


def compute_batch_loss(
    model: torch.nn.Module,
    batch,
    criterion: nn.Module,
    config: TrainingConfig,
    *,
    create_graph: bool,
) -> BatchLoss:
    validate_required_labels(batch, config)
    compute_forces = loss_weight_active(config.w2)
    compute_stress = loss_weight_active(config.w3)

    energy_pred, force_pred, stress_pred = predict_energy_forces_stress(
        model,
        batch,
        unit_trans=config.unit_trans,
        create_graph=create_graph,
        compute_forces=compute_forces,
        compute_stress=compute_stress,
    )
    loss = energy_pred.new_zeros(())
    energy_mae = None
    force_mae = None
    stress_mae = None
    force_count = 0
    stress_count = 0

    if loss_weight_active(config.w1):
        num_atoms = batch.num_atoms.to(dtype=energy_pred.dtype)
        energy_per_atom = energy_pred.view(-1) / num_atoms
        target_energy = batch.energy.view(-1) / num_atoms
        energy_loss = criterion(energy_per_atom, target_energy)
        loss = loss + config.w1 * energy_loss
        energy_mae = mae(energy_per_atom.detach(), target_energy.detach())

    if compute_forces:
        force_loss = criterion(force_pred.reshape(-1), batch.forces.reshape(-1))
        loss = loss + config.w2 * force_loss
        force_mae = mae(force_pred.detach().reshape(-1), batch.forces.detach().reshape(-1))
        force_count = int(batch.forces.numel())

    if compute_stress:
        stress_loss = criterion(stress_pred.reshape(-1), batch.stress.reshape(-1))
        loss = loss + config.w3 * stress_loss
        stress_mae = mae(stress_pred.detach().reshape(-1), batch.stress.detach().reshape(-1))
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
