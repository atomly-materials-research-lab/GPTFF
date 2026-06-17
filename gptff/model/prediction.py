from __future__ import annotations

import torch


def predict_energy_forces_stress(
    model,
    batch,
    unit_trans=160.21766208,
    create_graph=True,
    *,
    compute_forces=True,
    compute_stress=True,
):
    graph = batch.with_geometry(
        positions_requires_grad=compute_forces,
        strain_requires_grad=compute_stress,
    )
    energy = model(graph).squeeze(-1)

    if not compute_forces and not compute_stress:
        return energy, None, None

    zero_energy = energy.sum() * 0.0 if energy.requires_grad else None

    grad_inputs = []
    if compute_forces:
        grad_inputs.append(graph.positions)
    if compute_stress:
        grad_inputs.append(graph.strain)

    if energy.requires_grad:
        gradients = torch.autograd.grad(
            energy,
            grad_inputs,
            torch.ones_like(energy),
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=True,
        )
    else:
        gradients = tuple(None for _ in grad_inputs)

    grad_iter = iter(gradients)
    if compute_forces:
        forces = -_optional_gradient(next(grad_iter), graph.positions, zero_energy)
    else:
        forces = None
    if compute_stress:
        stress_grad = _optional_gradient(next(grad_iter), graph.strain, zero_energy)
        stress = stress_grad / graph.volumes[:, None, None] * unit_trans
    else:
        stress = None
    return energy, forces, stress


def _optional_gradient(gradient, reference, zero_energy):
    if gradient is not None:
        return gradient
    zeros = torch.zeros_like(reference)
    if zero_energy is None:
        return zeros
    return zeros + zero_energy
